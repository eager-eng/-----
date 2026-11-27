"""Acquire public BTS release metadata, later-period archives and weather inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
import json
import re
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import bts_fetch_development as downloader

CATALOG = "https://transtats.bts.gov/PREZIP/"
MOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
PACKAGE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}:\d{2}\s+[AP]M\s+(\d+)\s+"
    r'<a[^>]+>(On_Time_Reporting_Carrier_On_Time_Performance_(?:1987_present|\(1987_present\))_'
    r'(\d{4})_(\d{1,2})\.zip)</a>', re.IGNORECASE)


def fetch(url, timeout=25):
    with urlopen(Request(url, headers={"User-Agent": "Q3 historical model research"}), timeout=timeout) as stream:
        return stream.read()


def catalog(directory):
    entries = []
    html = fetch(CATALOG).decode("utf-8", errors="replace")
    for date, size, name, year, month in PACKAGE.findall(html):
        if 2023 <= int(year) <= 2026:
            entries.append({"month": f"{year}-{int(month):02d}", "directory_date": pd.Timestamp(date).date(),
                            "bytes": int(size), "archive": unescape(name), "source": CATALOG})
    frame = pd.DataFrame(entries).sort_values("month").drop_duplicates("month", keep="last")
    required = {f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)}
    if not required.issubset(frame.month):
        raise ValueError("Incomplete published BTS catalog")
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "bts_release_calendar.csv", index=False)
    return frame


def flight_archives(calendar, years, workers):
    tasks = calendar.loc[calendar.month.str[:4].astype(int).isin(years)]
    path = ROOT / "data/raw/bts"
    for year in years:
        sizes = tasks.loc[tasks.month.str.startswith(str(year))].set_index("month").bytes
        downloader.SIZES[year] = tuple(int(sizes.get(f"{year}-{m:02d}", 0)) for m in range(1,13))
    with ThreadPoolExecutor(workers) as executor:
        futures = {executor.submit(downloader.download, path, int(m[:4]), int(m[5:])): m for m in tasks.month}
        failures = []
        for item in as_completed(futures):
            month = futures[item]
            try:
                result = item.result()
                print(f"BTS {month}: {result.stat().st_size:,} bytes", flush=True)
            except Exception as error:
                failures.append(month)
                print(f"BTS {month}: {error}", flush=True)
    if failures:
        raise RuntimeError("Unavailable BTS months: " + ", ".join(sorted(failures)))


def mos_file(directory, station, year):
    day0 = f"{year}-01-01T00:00Z"
    day1 = f"{year}-12-31T18:00Z"
    source = f"{MOS}?"+urlencode(dict(station=station, model="GFS", sts=day0, ets=day1, format="csv"))
    path = directory / f"{station}_{year}.csv"
    if path.exists():
        return {"station": station, "year": year, "status": "cached", "bytes": path.stat().st_size, "source": source}
    for attempt in range(3):
        try:
            # Windows curl uses the Windows certificate store for the IEM endpoint.
            data = subprocess.run(["curl.exe","-fLsS","--max-time","45",source],
                                  check=True,capture_output=True).stdout
            if not data.startswith(b"runtime,ftime,"):
                raise ValueError("Unexpected MOS schema")
            staging = path.with_suffix(".part")
            staging.write_bytes(data)
            staging.replace(path)
            status = "available" if data.count(b"\n")>1 else "empty"
            return {"station": station, "year": year, "status": status, "bytes": len(data), "source": source}
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            if attempt == 2:
                message=error.stderr.decode("utf-8",errors="replace").strip() if getattr(error,"stderr",None) else str(error)
                return {"station": station, "year": year, "status": message, "bytes": 0, "source": source}


def weather(root, stations, years, workers):
    directory = root/"mos"
    directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as executor:
        tasks = [executor.submit(mos_file, directory, s, y) for s in stations for y in years]
        manifest = []
        for task in as_completed(tasks):
            row = task.result()
            manifest.append(row)
            if len(manifest) % 20 == 0 or len(manifest) == len(tasks):
                print(f"MOS {len(manifest)}/{len(tasks)}; {sum(r['bytes']>1000 for r in manifest)} nonempty", flush=True)
    out = root/f"mos_acquisition_{min(years)}_{max(years)}.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["bts","mos"])
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stations", nargs="+", default=[])
    parser.add_argument("--station-file", type=Path)
    args = parser.parse_args()
    if args.task == "bts":
        flight_archives(catalog(ROOT/"data/audit"), args.years, args.workers)
    else:
        stations = args.stations
        if args.station_file:
            stations += args.station_file.read_text(encoding="utf-8").split()
        if not stations:
            parser.error("Supply weather stations")
        weather(ROOT/"data/raw/iem", sorted(set(stations)), args.years,args.workers)


if __name__ == "__main__":
    main()
