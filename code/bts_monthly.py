"""Audit one BTS Reporting Carrier month and produce flight and airport CSVs."""

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd


KEY = ["FlightDate", "DOT_ID_Reporting_Airline", "Flight_Number_Reporting_Airline",
       "OriginAirportID", "DestAirportID", "CRSDepTime"]
DIVERSION_AIRPORTS = [f"Div{n}Airport" for n in range(1, 6)]
REQUIRED = KEY + [
    "Reporting_Airline", "Origin", "Dest", "CRSArrTime", "Flights", "Cancelled",
    "CancellationCode", "Diverted", "DivAirportLandings", "DivReachedDest",
    "DivArrDelay", "ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrDel15",
]
ANALYSIS_FIELDS = [
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    "CRSElapsedTime", "ActualElapsedTime", "Distance", "DayOfWeek",
]
OPTIONAL = DIVERSION_AIRPORTS + ["DepTimeBlk"] + ANALYSIS_FIELDS
OUTCOMES = ("C", "V", "L", "M", "T", "U")
THRESHOLDS = (45, 60, 90)
DICTIONARY = "https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ"


def clean(series):
    return series.astype("string").fillna("").str.strip()


def integer(series):
    raw = clean(series)
    whole = raw.str.fullmatch(r"[+-]?\d+(?:\.0+)?")
    return pd.to_numeric(raw.where(whole), errors="coerce").astype("Int64")


def normalize_clock(series):
    values = integer(series)
    midnight = values.eq(2400).fillna(False)
    valid = ((values.between(0, 2359) & values.mod(100).lt(60)).fillna(False) | midnight)
    clocks = values.where(valid).mask(midnight, 0).astype("string").str.zfill(4)
    day_offsets = midnight.astype("Int64").where(valid)
    return pd.DataFrame({"hhmm": clocks, "day_offset": day_offsets})


def candidate_keys(frame):
    key = frame[KEY].apply(clean)
    key["FlightDate"] = pd.to_datetime(key.FlightDate, format="%Y-%m-%d", errors="coerce").dt.strftime("%Y-%m-%d")
    clock = normalize_clock(key.CRSDepTime)
    key["CRSDepTime"] = clock.hhmm.mask(clock.day_offset.eq(1).fillna(False), "2400")
    for column in ("DOT_ID_Reporting_Airline", "OriginAirportID", "DestAirportID"):
        key[column] = integer(key[column]).where(lambda value: value.gt(0)).astype("string")
    return key


def airport_roles(frame, airports):
    return pd.DataFrame({
        "is_selected_origin": clean(frame.Origin).isin(airports),
        "is_selected_destination": clean(frame.Dest).isin(airports),
    }, index=frame.index)


def classify(frame):
    """Derive the primary categorical outcome and separate nullable event flags."""
    cancelled, diverted, div_reached, on_time_indicator = (
        integer(frame[column]).where(lambda value: value.isin([0, 1]))
        for column in ("Cancelled", "Diverted", "DivReachedDest", "ArrDel15")
    )
    delay, delay_nonnegative, div_delay, landings = (
        integer(frame[column]) for column in ("ArrDelay", "ArrDelayMinutes", "DivArrDelay", "DivAirportLandings")
    )
    delay_nonnegative = delay_nonnegative.where(delay_nonnegative.ge(0))
    arrived_clock = normalize_clock(frame["ArrTime"]).hhmm
    cancel_reported = cancelled.eq(1).fillna(False)
    diversion_reported = diverted.eq(1).fillna(False)
    return_cancel = landings.eq(9).fillna(False)
    diversion_airports = frame.reindex(columns=DIVERSION_AIRPORTS, fill_value="").apply(clean)
    known_stops = diversion_airports.ne("")
    stops_elsewhere = (known_stops & diversion_airports.ne(clean(frame.Origin), axis=0)).any(axis=1)
    numeric_diversion = landings.between(1, 5).fillna(False)
    evidence_of_diversion = (diversion_reported | numeric_diversion | stops_elsewhere) & ~return_cancel
    evidence_of_diversion |= return_cancel & stops_elsewhere
    direct_status = cancelled.eq(0).fillna(False) & diverted.eq(0).fillna(False)

    cancel_arrival_conflict = cancel_reported & (
        clean(frame.ArrDelay).ne("") | arrived_clock.notna() | div_reached.eq(1).fillna(False)
    )
    status_conflict = cancel_arrival_conflict | (return_cancel & cancelled.eq(0).fillna(False))
    status_conflict |= direct_status & (evidence_of_diversion | div_reached.eq(1).fillna(False))
    status_conflict |= cancelled.eq(0).fillna(False) & diversion_reported & clean(frame.ArrDelay).ne("")
    audited_cancel = cancel_reported & ~status_conflict
    audited_diversion = cancelled.eq(0).fillna(False) & diversion_reported & ~status_conflict

    audit_fields = {
        "Cancelled": cancelled, "Diverted": diverted, "DivReachedDest": div_reached,
        "ArrDel15": on_time_indicator, "ArrDelay": delay, "ArrDelayMinutes": delay_nonnegative,
        "DivArrDelay": div_delay, "DivAirportLandings": landings, "ArrTime": arrived_clock,
        "CRSDepTime": normalize_clock(frame.CRSDepTime).hhmm,
        "CRSArrTime": normalize_clock(frame.CRSArrTime).hhmm,
    }
    invalid = pd.Series(False, index=frame.index)
    for column, parsed in audit_fields.items():
        invalid |= clean(frame[column]).ne("") & parsed.isna()
    delay_conflict = (delay.clip(lower=0).ne(delay_nonnegative) & delay.notna() & delay_nonnegative.notna()).fillna(False)
    combined_delay = delay.fillna(delay_nonnegative)
    delay_conflict |= (combined_delay.ge(15).ne(on_time_indicator.eq(1)) & combined_delay.notna()
                       & on_time_indicator.notna()).fillna(False)
    combined_delay = combined_delay.mask(delay_conflict)
    arrived_direct = direct_status & ~status_conflict & (
        arrived_clock.notna() | delay.notna() | delay_nonnegative.notna() | on_time_indicator.notna()
    )
    direct_on_time = (combined_delay.lt(15) | (combined_delay.isna() & on_time_indicator.eq(0))).fillna(False)
    direct_on_time &= arrived_direct & ~delay_conflict
    main = pd.Series("U", index=frame.index, dtype="string")
    main.loc[audited_cancel] = "C"
    main.loc[audited_diversion] = "V"
    main.loc[direct_on_time] = "T"
    main.loc[arrived_direct & combined_delay.between(15, 59).fillna(False)] = "M"
    main.loc[arrived_direct & combined_delay.ge(60).fillna(False)] = "L"

    undiverted_on_time = main.eq("T").astype("boolean").mask(main.eq("U"))
    known_late = arrived_direct & ~delay_conflict & on_time_indicator.eq(1).fillna(False)
    undiverted_on_time.loc[known_late] = False

    ever_diverted = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    ever_diverted.loc[diverted.eq(0).fillna(False) & ~evidence_of_diversion & ~return_cancel] = False
    ever_diverted.loc[return_cancel & known_stops.any(axis=1) & ~stops_elsewhere] = False
    ever_diverted.loc[evidence_of_diversion] = True
    final_reached = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    final_reached.loc[audited_cancel] = False
    final_reached.loc[arrived_direct] = True
    diversion_reach_conflict = audited_diversion & div_reached.eq(0).fillna(False) & div_delay.notna()
    final_reached.loc[audited_diversion & ~diversion_reach_conflict] = div_reached.astype("boolean")
    final_late = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    final_late.loc[final_reached.eq(False).fillna(False) | direct_on_time] = False
    final_late.loc[arrived_direct & combined_delay.notna()] = combined_delay.ge(60)
    final_late.loc[audited_diversion & final_reached.fillna(False) & div_delay.notna()] = div_delay.ge(60)

    result = pd.DataFrame({
        "outcome_code": main, "undiverted_on_time": undiverted_on_time,
        "ever_diverted": ever_diverted, "final_reached": final_reached,
        "final_late_60": final_late, "return_cancel_code_9": return_cancel,
        "quality_status_conflict": status_conflict, "quality_delay_conflict": delay_conflict,
        "quality_diversion_reach_conflict": diversion_reach_conflict, "quality_invalid_field": invalid,
        "quality_missing_final_div_delay": audited_diversion & final_reached.fillna(False) & div_delay.isna(),
        "raw_cancelled_and_diverted": cancel_reported & diversion_reported,
    })
    for threshold in THRESHOLDS:
        severe = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        severe.loc[audited_cancel | audited_diversion] = True
        severe.loc[direct_on_time] = False
        severe.loc[arrived_direct & combined_delay.notna()] = combined_delay.ge(threshold)
        result[f"severe_{threshold}"] = severe
    return result


@contextmanager
def source_csv(path):
    if path.suffix.lower() != ".zip":
        with path.open("rb") as stream:
            yield stream, path.name
        return
    with ZipFile(path) as archive:
        names = [member for member in archive.namelist() if member.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path.name}, found {len(names)}")
        with archive.open(names[0]) as stream:
            yield stream, names[0]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(frame, origin):
    n = len(frame)
    outcomes = {status: int(frame.outcome_code.eq(status).sum()) for status in OUTCOMES}
    assert sum(outcomes.values()) == n
    on_time, unknown = outcomes["T"], outcomes["U"]
    possible_on_time = int(frame.undiverted_on_time.isna().sum())
    result = {
        "origin": origin, "n": n, **{f"n_{key}": count for key, count in outcomes.items()},
        "unknown_primary_rate": ratio(unknown, n), "on_time_confirmed_rate": ratio(on_time, n),
        "on_time_known_outcome_rate": ratio(on_time, n - unknown),
        "on_time_undiverted_arrival_rate": ratio(on_time, sum(outcomes[key] for key in "TML")),
        "on_time_conservative_upper": ratio(on_time + unknown, n),
        "on_time_evidence_upper": ratio(on_time + possible_on_time, n),
        "cancel_confirmed_rate": ratio(outcomes["C"], n),
        "diverted_not_cancelled_rate": ratio(outcomes["V"], n),
    }
    for event in ("ever_diverted", "final_reached", "final_late_60", *(f"severe_{value}" for value in THRESHOLDS)):
        confirmed = int(frame[event].sum())
        missing = int(frame[event].isna().sum())
        result[f"n_{event}"] = confirmed
        result[f"n_{event}_unknown"] = missing
        result[f"{event}_confirmed_rate"] = ratio(confirmed, n)
        result[f"{event}_upper"] = ratio(confirmed + missing, n)
    return result


def process_month(source, output, month, source_url=None, downloaded_on=None, *, airports=None):
    period = pd.Period(month, freq="M")
    if str(period) != month:
        raise ValueError("Month must use YYYY-MM")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    with source_csv(source) as (stream, member):
        raw = pd.read_csv(stream, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = REQUIRED if airports is None else REQUIRED + ANALYSIS_FIELDS
    missing = sorted(set(required).difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    source_columns = list(raw.columns)
    row_numbers = pd.Series(raw.index + 2, index=raw.index)

    key = candidate_keys(raw)
    dates = pd.to_datetime(key.FlightDate, format="%Y-%m-%d", errors="coerce")
    dates_in_month = dates.dt.to_period("M").eq(period)
    missing_key = key.isna().any(axis=1) | key.eq("").any(axis=1)
    one_flight = integer(raw.Flights).eq(1).fillna(False)
    valid_identity = dates_in_month & ~missing_key & one_flight & clean(raw.Origin).ne("")

    candidate_repeat = key.loc[valid_identity].duplicated(keep=False)
    exact_duplicates = raw.loc[candidate_repeat[candidate_repeat].index].duplicated()
    duplicate_index = exact_duplicates[exact_duplicates].index
    surviving_keys = key.loc[valid_identity].drop(index=duplicate_index)
    conflicting = surviving_keys.duplicated(keep=False)
    conflict_index = conflicting[conflicting].index
    included_index = surviving_keys.index.difference(conflict_index, sort=False)
    counts = {"raw_rows": len(raw), "exact_duplicates": len(duplicate_index),
              "identity_unresolved": int((~valid_identity).sum()) + len(conflict_index), "included": len(included_index)}
    assert counts["raw_rows"] == sum(value for name, value in counts.items() if name != "raw_rows")

    data_columns = REQUIRED + [column for column in OPTIONAL if column in raw]
    data = raw.loc[included_index, data_columns].copy()
    data.insert(0, "source_row", row_numbers.loc[included_index])
    data.insert(0, "source_file_month", month)
    data = data.join(classify(data))
    for field in ("CRSDepTime", "CRSArrTime"):
        data = data.join(normalize_clock(data[field]).add_prefix(f"{field}_"))

    flagged = data.filter(regex=r"^(quality_|raw_cancelled_and_diverted$|return_cancel_code_9$)").any(axis=1)
    outcome_exceptions = data.loc[flagged | data.outcome_code.eq("U")].copy()
    outcome_exceptions["audit_disposition"] = "included_review"
    excluded = raw.loc[raw.index.difference(included_index, sort=False), data_columns].copy()
    excluded.insert(0, "source_row", row_numbers.loc[excluded.index])
    excluded.insert(0, "source_file_month", month)
    excluded["audit_disposition"] = "identity_unresolved"
    excluded.loc[duplicate_index, "audit_disposition"] = "exact_duplicate"
    excluded["audit_bad_flights"] = ~one_flight.loc[excluded.index]
    excluded["audit_invalid_or_outside_month"] = ~dates_in_month.loc[excluded.index]
    excluded["audit_missing_key_or_origin"] = missing_key.loc[excluded.index] | clean(excluded.Origin).eq("")
    excluded["audit_candidate_key_collision"] = excluded.index.isin(conflict_index)
    exceptions = pd.concat([outcome_exceptions, excluded], ignore_index=True).sort_values("source_row")
    summaries = pd.DataFrame([summarize(data, "ALL")] + [
        summarize(group, origin) for origin, group in data.groupby("Origin", sort=True)
    ])
    assert int(summaries.iloc[1:].n.sum()) == counts["included"]
    export = data
    if airports is not None:
        roles = airport_roles(data, airports)
        scope = roles.any(axis=1)
        export = data.loc[scope].join(roles.loc[scope])
    manifest = {
        "month": month, "source": str(source.resolve()), "source_url": source_url,
        "downloaded_on": downloaded_on,
        "source_member": member, "source_sha256": sha256(source),
        "dictionary_url": DICTIONARY, "dictionary_checked_on": "2026-11-26",
        "run_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
        "python_executable": sys.executable, "pandas": pd.__version__,
        "script_sha256": sha256(Path(__file__)), "command_argv": [sys.executable, *sys.argv],
        "source_columns": source_columns, "counts": counts, "exception_rows": len(exceptions),
        "analysis_airports": list(airports) if airports is not None else None,
        "exported_rows": len(export),
        "flights_values": {str(key): int(value) for key, value in raw.Flights.value_counts().items()},
        "diversion_landings_values": {str(key): int(value) for key, value in raw.DivAirportLandings.value_counts().items()},
        "reporting_carrier_ids": sorted(set(clean(raw.DOT_ID_Reporting_Airline)) - {""}),
        "quality_counts": {key: int(value) for key, value in data.filter(regex=r"^(quality_|raw_cancelled_and_diverted$|return_cancel_code_9$)").sum().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".audit-") as directory:
        staging = Path(directory) / "result"
        staging.mkdir()
        export.to_csv(staging / "flights.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        summaries.to_csv(staging / "airport_summary.csv", index=False)
        exceptions.to_csv(staging / "exceptions.csv", index=False)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="BTS Reporting Carrier ZIP or CSV")
    parser.add_argument("--month", required=True, help="Flight month, YYYY-MM")
    parser.add_argument("--output", required=True, type=Path, help="New result directory")
    parser.add_argument("--source-url", help="Original download URL for provenance")
    parser.add_argument("--downloaded-on", type=date.fromisoformat, help="Acquisition date, YYYY-MM-DD")
    args = parser.parse_args()
    downloaded_on = args.downloaded_on.isoformat() if args.downloaded_on else None
    manifest = process_month(args.source, args.output, args.month, args.source_url, downloaded_on)
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
