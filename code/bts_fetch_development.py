"""Resume downloads of the 2023–2024 BTS Reporting Carrier monthly archives."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


# Published archive sizes in the BTS PREZIP index, January through December.
SIZES = {
    2023: (27068766, 24730977, 28954576, 27994592, 28333534, 29075972,
           30208166, 29539237, 28047499, 29452902, 27573257, 28818502),
    2024: (27573265, 25086854, 29313663, 28540589, 31139902, 31226761,
           34427174, 30854511, 28433978, 29418453, 28149354, 30150370),
}
HOSTS = ("www.transtats.bts.gov", "transtats.bts.gov")
MIB = 1024 * 1024


def download(directory, year, month):
    size = SIZES[year][month - 1]
    suffix = "(1987_present)" if (year, month) == (2023, 12) else "1987_present"
    target = directory / f"On_Time_Reporting_Carrier_On_Time_Performance_{suffix}_{year}_{month}.zip"
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError(f"Existing archive differs from the BTS size: {target}")
        return target

    staging = target.with_suffix(".zip.part")
    failed_reads = 0
    while True:
        start = staging.stat().st_size if staging.exists() else 0
        if start == size:
            break
        if start > size:
            raise ValueError(f"Partial archive exceeds the BTS size: {staging}")
        end = min(size, start + MIB) - 1
        url = f"https://{HOSTS[failed_reads % len(HOSTS)]}/PREZIP/{quote(target.name)}"
        request = Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urlopen(request, timeout=15) as response:
                expected = f"bytes {start}-{end}/{size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise OSError(f"Expected BTS range {expected}, got {response.headers.get('Content-Range')}")
                with staging.open("ab") as stream:
                    while chunk := response.read1(64 * 1024):
                        stream.write(chunk)
            if staging.stat().st_size != end + 1:
                raise OSError("Connection ended before the requested range arrived")
        except OSError as error:
            received = staging.stat().st_size if staging.exists() else 0
            failed_reads = 0 if received > start else failed_reads + 1
            if failed_reads >= 4:
                raise OSError(f"Unable to resume {target.name} after {received:,} bytes: {error}") from error
        else:
            failed_reads = 0

    with ZipFile(staging) as archive:
        if not any(name.lower().endswith(".csv") for name in archive.namelist()) or archive.testzip() is not None:
            raise ValueError(f"ZIP validation failed: {staging}")
    staging.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 13))
    months = [(year, month) for year in SIZES for month in (3, 11)]
    months += [(year, month) for year in SIZES for month in range(1, 13) if month not in (3, 11)]
    parser.add_argument("--months", nargs="+", choices=[f"{y}-{m:02d}" for y, m in months])
    args = parser.parse_args()
    if args.months:
        months = [(year, month) for year, month in months if f"{year}-{month:02d}" in args.months]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = {executor.submit(download, args.raw_dir, year, month): f"{year}-{month:02d}"
                 for year, month in months}
        for task in as_completed(tasks):
            try:
                archive = task.result()
                print(f"{tasks[task]} ready: {archive.stat().st_size:,} bytes", flush=True)
            except (OSError, ValueError, BadZipFile) as error:
                failed = True
                print(f"{tasks[task]} failed: {error}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
