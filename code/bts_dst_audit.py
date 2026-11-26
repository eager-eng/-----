"""Check scheduled clocks on real eight-airport flights around US DST transitions."""

import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from bts_batch import AIRPORTS, read_csv
from bts_monthly import integer


AIRPORT_ZONES = {
    airport: ZoneInfo(zone)
    for airport, zone in {
        "ATL": "America/New_York", "DEN": "America/Denver", "DFW": "America/Chicago",
        "MCO": "America/New_York", "ORD": "America/Chicago", "PDX": "America/Los_Angeles",
        "RDU": "America/New_York", "SFO": "America/Los_Angeles",
    }.items()
}
AUDIT_FIELDS = ["source_file_month", "source_row", "FlightDate", "Origin", "Dest",
                "CRSDepTime", "CRSArrTime", "CRSElapsedTime"]
READ_FIELDS = AUDIT_FIELDS + ["CRSDepTime_hhmm", "CRSDepTime_day_offset", "CRSArrTime_hhmm"]
HOUR = timedelta(hours=1)
DAY = timedelta(hours=24)


def transition_date(year, month):
    first = date(year, month, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + (7 * DAY if month == 3 else timedelta())


def utc_candidates(local, zone):
    candidates = {local.replace(tzinfo=zone, fold=fold).astimezone(UTC) for fold in (0, 1)}
    return sorted(value for value in candidates if value.astimezone(zone).replace(tzinfo=None) == local)


def check_flight(row):
    result = {field: getattr(row, field) for field in AUDIT_FIELDS}
    result.update(status="missing_scheduled_time", dep_utc="", cutoff_utc="", arr_utc="",
                  arr_local="", cutoff_local="", cutoff_wall_hours="")
    if not row.CRSDepTime_hhmm or not row.CRSArrTime_hhmm or pd.isna(row.elapsed_minutes):
        return result
    origin_zone = AIRPORT_ZONES[row.Origin]
    destination_zone = AIRPORT_ZONES[row.Dest]
    local = datetime.fromisoformat(f"{row.FlightDate}T{row.CRSDepTime_hhmm}")
    local += timedelta(days=int(row.CRSDepTime_day_offset))
    departures = utc_candidates(local, origin_zone)
    if not departures:
        result["status"] = "nonexistent_departure_clock"
        return result
    arrivals = [(dep, dep + timedelta(minutes=int(row.elapsed_minutes))) for dep in departures]
    matches = [(dep, arr) for dep, arr in arrivals
               if arr.astimezone(destination_zone).strftime("%H%M") == row.CRSArrTime_hhmm]
    if len(matches) != 1:
        result["status"] = "ambiguous_departure_clock" if matches else "scheduled_arrival_clock_mismatch"
        return result
    dep, arr = matches[0]
    cutoff = dep - DAY
    cutoff_local = cutoff.astimezone(origin_zone)
    wall_hours = (local - cutoff_local.replace(tzinfo=None)) / HOUR
    result.update(status="matched_unique" if len(departures) == 1 else "fold_matched_using_arrival",
                  dep_utc=dep.isoformat(), cutoff_utc=cutoff.isoformat(), arr_utc=arr.isoformat(),
                  arr_local=arr.astimezone(destination_zone).isoformat(),
                  cutoff_local=cutoff_local.isoformat(), cutoff_wall_hours=wall_hours)
    return result


def check_transition(batch, year, month):
    path = batch / f"{year}-{month:02}" / "flights.csv.gz"
    shift = transition_date(year, month)
    dates = [(shift + day * DAY).isoformat() for day in (-1, 0, 1)]
    raw = read_csv(path, READ_FIELDS)
    scope = raw.FlightDate.isin(dates) & raw.Origin.isin(AIRPORTS) & raw.Dest.isin(AIRPORTS)
    rows = raw.loc[scope].copy()
    rows["elapsed_minutes"] = integer(rows.CRSElapsedTime).where(lambda value: value.gt(0))
    if set(rows.Origin) != set(AIRPORTS):
        raise ValueError(f"Incomplete DST audit airport coverage in {path}")
    return pd.DataFrame.from_records(check_flight(row) for row in rows.itertuples(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--years", required=True, type=int, nargs="+")
    args = parser.parse_args()
    rows = pd.concat([check_transition(args.batch, year, month)
                      for year in sorted(set(args.years)) for month in (3, 11)], ignore_index=True)
    summary = rows.groupby(["source_file_month", "status", "cutoff_wall_hours"], as_index=False).size()
    rows.to_csv(args.batch / "dst_clock_checks.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary.to_csv(args.batch / "dst_clock_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
