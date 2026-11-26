"""Independently check sampled Q1 labels against the original BTS ZIP records."""

import argparse
import csv
from decimal import Decimal
import gzip
from io import TextIOWrapper
import json
from pathlib import Path
from zipfile import ZipFile

from q1_reliability import AIRPORTS, ROOT, THRESHOLDS


DELAYS = {-5, 0, 14, 15, 44, 45, 59, 60, 89, 90}
BOUNDARY_FIELDS = ("FlightDate", "Origin", "Dest", "Cancelled", "Diverted", "ArrDelay",
                   "DivAirportLandings", "DivReachedDest", "DivArrDelay")


def numeric(value):
    return Decimal(value) if value.strip() else None


def samples(folder):
    found = {}
    expected = set(DELAYS) | {"cancelled", "diverted_and_reached"}
    with gzip.open(folder / "flights.csv.gz", "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["is_selected_origin"] != "True":
                continue
            code = row["outcome_code"]
            if code in {"L", "M", "T"}:
                case = numeric(row["ArrDelay"])
            elif code == "C":
                case = "cancelled"
            elif code == "V" and row["final_reached"] == "True":
                case = "diverted_and_reached"
            else:
                continue
            if case in expected:
                found[int(row["source_row"])] = (str(case), row)
                expected.remove(case)
                if not expected:
                    break
    if expected:
        raise ValueError(f"Missing boundary sample in {folder.name}: {expected}")
    with (folder / "exceptions.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["Origin"] in AIRPORTS and row["outcome_code"] == "U":
                found[int(row["source_row"])] = ("unknown", row)
    return found


def verify_row(raw, prepared, case):
    for field in BOUNDARY_FIELDS:
        assert raw[field] == prepared[field], (case, field, raw[field], prepared[field])
    code = prepared["outcome_code"]
    expected = {"True", "False"}
    if case == "cancelled":
        assert code == "C" and numeric(raw["Cancelled"]) == 1 and not raw["ArrDelay"].strip()
        assert prepared["final_reached"] == "False"
    elif case == "diverted_and_reached":
        assert code == "V" and numeric(raw["Cancelled"]) == 0 and numeric(raw["Diverted"]) == 1
        assert numeric(raw["DivReachedDest"]) == 1 and not raw["ArrDelay"].strip()
        assert prepared["final_reached"] == "True"
        delay = numeric(raw["DivArrDelay"])
        if delay is not None:
            assert prepared["final_late_60"] == str(delay >= 60)
    elif case == "unknown":
        assert code == "U" and prepared["audit_disposition"] == "included_review"
        if raw["FlightDate"] == "2023-10-04" and raw["Origin"] == "ORD" and raw["Dest"] == "DCA":
            assert numeric(raw["Cancelled"]) == 0 and numeric(raw["Diverted"]) == 0
            assert numeric(raw["DivAirportLandings"]) == 9 and numeric(raw["ArrDelay"]) == 231
            assert all(not prepared[f"severe_{threshold}"] for threshold in THRESHOLDS)
        return
    else:
        delay = numeric(raw["ArrDelay"])
        assert delay == Decimal(case) and numeric(raw["Cancelled"]) == numeric(raw["Diverted"]) == 0
        assert code == ("T" if delay < 15 else "M" if delay < 60 else "L")
        assert prepared["undiverted_on_time"] == str(delay < 15)
        for threshold in THRESHOLDS:
            assert prepared[f"severe_{threshold}"] == str(delay >= threshold)
        return
    assert all(prepared[f"severe_{threshold}"] in expected for threshold in THRESHOLDS)
    assert all(prepared[f"severe_{threshold}"] == "True" for threshold in THRESHOLDS)


def verify_month(folder):
    pending = samples(folder)
    audit = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    results = []
    with ZipFile(audit["source"]) as archive:
        with archive.open(audit["source_member"]) as stream:
            with TextIOWrapper(stream, encoding="utf-8-sig", newline="") as source:
                for number, raw in enumerate(csv.DictReader(source), start=2):
                    if number in pending:
                        case, prepared = pending.pop(number)
                        verify_row(raw, prepared, case)
                        results.append({"month": folder.name, "source_row": number, "case": case,
                                        **{field: raw[field] for field in BOUNDARY_FIELDS},
                                        "outcome_code": prepared["outcome_code"]})
                    if not pending:
                        break
    if pending:
        raise ValueError(f"Missing original rows: {folder.name} {sorted(pending)}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/问题一计算结果")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for item in manifest["inputs"]:
        results.extend(verify_month(Path(item["flight_file"]).parent))
    path = args.output / "source_boundary_checks.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Independently checked {len(results)} raw records in {len(manifest['inputs'])} months.")


if __name__ == "__main__":
    main()
