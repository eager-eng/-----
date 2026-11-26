"""Build the eight-airport analysis files from explicitly selected BTS months."""

import argparse
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote

import pandas as pd

import bts_monthly
from bts_monthly import KEY, airport_roles, candidate_keys, process_month, sha256


AIRPORTS = ("ATL", "DEN", "DFW", "MCO", "ORD", "PDX", "RDU", "SFO")
MONTH_PATTERN = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])")
PACKAGE_PATTERN = re.compile(
    r"On_Time_Reporting_Carrier_On_Time_Performance_(?:1987_present|\(1987_present\))_"
    r"(\d{4})_(\d{1,2})\.zip", re.IGNORECASE,
)
SOURCE_BASE = "https://www.transtats.bts.gov/PREZIP/"
PROVENANCE = ["source_file_month", "source_row"]
ISSUE_COLUMNS = KEY + PROVENANCE + ["audit_disposition", "cross_month_status"]
EVENT_FLAGS = ["undiverted_on_time", "ever_diverted", "final_reached", "final_late_60",
               "severe_45", "severe_60", "severe_90"]


def package_months(directory):
    if not directory.is_dir():
        raise ValueError(f"BTS package directory does not exist: {directory}")
    packages = {}
    for path in sorted(directory.iterdir()):
        match = PACKAGE_PATTERN.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        year, number = match.groups()
        month = f"{year}-{int(number):02d}"
        if MONTH_PATTERN.fullmatch(month) is None:
            continue
        if month in packages:
            raise ValueError(f"Multiple BTS packages for {month}: {packages[month].name}, {path.name}")
        packages[month] = path
    return packages


def month_result(source, folder, month, airports):
    if folder.exists():
        with (folder / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        expected = (month, list(airports), sha256(source), sha256(Path(bts_monthly.__file__)))
        actual = tuple(manifest.get(field) for field in ("month", "analysis_airports", "source_sha256", "script_sha256"))
        if actual != expected:
            raise ValueError(f"Cached data for {month} has different inputs or code. Use a new output directory.")
        for name in ("flights.csv.gz", "exceptions.csv", "airport_summary.csv"):
            if not (folder / name).is_file():
                raise ValueError(f"Cached data for {month} is missing {name}. Use a new output directory.")
        return manifest
    return process_month(source, folder, month, SOURCE_BASE + quote(source.name), airports=airports)


def read_csv(path, columns):
    return pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)


def cross_month_issues(output, months, airports):
    """Link off-month records before treating the included monthly keys as unique."""
    wrong_month = []
    for month in months:
        columns = KEY + PROVENANCE + ["Origin", "Dest", "audit_disposition", "audit_invalid_or_outside_month"]
        exceptions = read_csv(output / month / "exceptions.csv", columns)
        scope = airport_roles(exceptions, airports).any(axis=1)
        candidates = exceptions.loc[scope & exceptions.audit_invalid_or_outside_month.eq("True")]
        if not candidates.empty:
            wrong_month.append(candidate_keys(candidates).join(candidates[PROVENANCE + ["audit_disposition"]]))
    if not wrong_month:
        return pd.DataFrame(columns=ISSUE_COLUMNS), 0

    off_month = pd.concat(wrong_month, ignore_index=True)
    complete = off_month[KEY].notna().all(axis=1) & off_month[KEY].ne("").all(axis=1)
    candidates = off_month.loc[complete].copy()
    canonical_month = candidates.FlightDate.str[:7]
    included_matches = []
    # A retained row passed its source-month check. An identical valid key can
    # therefore only match it in the month identified by the key's FlightDate.
    for month, group in candidates.groupby(canonical_month):
        if month not in months:
            continue
        included = read_csv(output / month / "flights.csv.gz", KEY + PROVENANCE)
        keys = candidate_keys(included).join(included[PROVENANCE])
        matches = keys.merge(group[KEY].drop_duplicates(), on=KEY, validate="one_to_one")
        if not matches.empty:
            included_matches.append(matches.assign(audit_disposition="included"))

    off_month["cross_month_status"] = "unresolved_candidate_key"
    off_month.loc[complete, "cross_month_status"] = "outside_month_unmatched"
    off_month.loc[complete & ~off_month.FlightDate.str[:7].isin(months), "cross_month_status"] = "canonical_month_not_loaded"
    issues = pd.concat([off_month, *included_matches], ignore_index=True)
    known_months = issues.groupby(KEY, dropna=True).source_file_month.transform("nunique")
    same_key = known_months.gt(1).fillna(False)
    issues.loc[same_key, "cross_month_status"] = "same_key_across_months"
    return issues[ISSUE_COLUMNS].sort_values(PROVENANCE), len(issues.loc[same_key, KEY].drop_duplicates())


def monthly_statistics(folder, manifest):
    columns = ["Origin", "DOT_ID_Reporting_Airline", "outcome_code",
               "is_selected_origin", "is_selected_destination", *EVENT_FLAGS]
    rows = read_csv(folder / "flights.csv.gz", columns)
    primary = rows.is_selected_origin.eq("True")
    inbound = rows.is_selected_destination.eq("True")
    counts = manifest["counts"]
    stats = {
        "source_file_month": manifest["month"], "bts_included": counts["included"],
        "identity_unresolved": counts["identity_unresolved"], "exact_duplicates": counts["exact_duplicates"],
        "selected_union": len(rows), "selected_origin": int(primary.sum()),
        "selected_destination": int(inbound.sum()), "selected_both": int((primary & inbound).sum()),
    }
    assert (primary | inbound).all() and len(rows) == manifest["exported_rows"]
    scope = rows.loc[primary, ["Origin", "DOT_ID_Reporting_Airline"]].copy()
    scope["n"] = 1
    scope["n_U"] = rows.loc[primary, "outcome_code"].eq("U").astype(int)
    for flag in EVENT_FLAGS:
        scope[f"n_{flag}_unknown"] = rows.loc[primary, flag].eq("").astype(int)
    grouped = scope.groupby(["Origin", "DOT_ID_Reporting_Airline"], as_index=False).sum()
    grouped.insert(0, "source_file_month", manifest["month"])
    grouped["unknown_primary_rate"] = grouped.n_U / grouped.n

    origin_by_carrier = rows.loc[primary].DOT_ID_Reporting_Airline.value_counts()
    coverage = [{"source_file_month": manifest["month"], "DOT_ID_Reporting_Airline": carrier,
                 "selected_origin_rows": int(origin_by_carrier.get(carrier, 0))}
                for carrier in manifest["reporting_carrier_ids"]]
    return stats, grouped, coverage


def run_batch(raw_dir, output, months, airports=AIRPORTS):
    months = sorted(set(months))
    airports = tuple(sorted(set(airports)))
    if not months or any(MONTH_PATTERN.fullmatch(month) is None for month in months) or not airports:
        raise ValueError("Specify at least one month in YYYY-MM and at least one airport")
    packages = package_months(raw_dir)
    missing = [month for month in months if month not in packages]
    if missing:
        raise ValueError(f"Missing BTS packages for: {', '.join(missing)}")
    output.mkdir(parents=True, exist_ok=True)
    cached_months = {path.name for path in output.iterdir() if path.is_dir() and MONTH_PATTERN.fullmatch(path.name)}
    if cached_months.difference(months):
        raise ValueError("Output has month folders outside the request. Use a new output directory.")

    monthly = []
    unknown = []
    carriers = []
    for month in months:
        folder = output / month
        manifest = month_result(packages[month], folder, month, airports)
        stats, groups, coverage = monthly_statistics(folder, manifest)
        monthly.append(stats)
        unknown.append(groups)
        carriers.extend(coverage)
        print(f"{month}: {stats['selected_union']} retained, {stats['selected_origin']} departures", flush=True)
    cross_month, duplicates = cross_month_issues(output, months, airports)
    result = {
        "months": months, "years": sorted({int(month[:4]) for month in months}), "airports": list(airports),
        "selected_union_rows": sum(row["selected_union"] for row in monthly),
        "selected_origin_rows": sum(row["selected_origin"] for row in monthly),
        "selected_destination_rows": sum(row["selected_destination"] for row in monthly),
        "cross_month_issue_rows": len(cross_month), "same_key_cross_month_groups": duplicates,
        "batch_script_sha256": sha256(Path(__file__)), "command_argv": [sys.executable, *sys.argv],
        "monthly_manifests": [f"{month}/manifest.json" for month in months],
    }
    with TemporaryDirectory(dir=output, prefix=".batch-") as directory:
        staging = Path(directory)
        pd.DataFrame(monthly).to_csv(staging / "monthly_coverage.csv", index=False)
        pd.concat(unknown, ignore_index=True).to_csv(staging / "unknown_by_origin_carrier_month.csv", index=False)
        pd.DataFrame(carriers).to_csv(staging / "carrier_coverage.csv", index=False)
        cross_month.to_csv(staging / "cross_month_issues.csv", index=False)
        (staging / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        for path in sorted(staging.iterdir(), key=lambda path: path.name == "manifest.json"):
            path.replace(output / path.name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument("--months", nargs="+", help="Explicit month list, YYYY-MM")
    period.add_argument("--years", nargs="+", type=int, help="Require all twelve months in each specified year")
    args = parser.parse_args()
    months = args.months or [f"{year:04d}-{number:02d}" for year in args.years for number in range(1, 13)]
    try:
        manifest = run_batch(args.raw_dir, args.output, months)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
