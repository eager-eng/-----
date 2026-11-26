import csv
import gzip
import json
from zipfile import ZipFile

from test_bts_monthly import flight
from bts_batch import package_months, run_batch


def package(directory, month, rows, with_parentheses=False):
    year, number = month.split("-")
    vintage = "(1987_present)" if with_parentheses else "1987_present"
    name = f"On_Time_Reporting_Carrier_On_Time_Performance_{vintage}_{year}_{int(number)}"
    source = directory / f"{name}.zip"
    fields = list(rows[0])
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    with ZipFile(source, "w") as output:
        output.writestr(f"{name}.csv", buffer.getvalue())
    return source


def analysis_flight(number, **updates):
    return flight(number, **({
        "CarrierDelay": "0", "WeatherDelay": "", "NASDelay": "15", "SecurityDelay": "0",
        "LateAircraftDelay": "0", "CRSElapsedTime": "120", "ActualElapsedTime": "135",
        "Distance": "802", "DayOfWeek": "5",
    } | updates))


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_batch_preserves_fields_airport_roles_and_wrong_month_keys(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    jan = [analysis_flight(1), analysis_flight(2, Origin="SEA", OriginAirportID="14747"),
           analysis_flight(3, Dest="SEA", DestAirportID="14747", ArrDelay="", ArrDelayMinutes="", ArrDel15="1"),
           analysis_flight(4, Origin="SEA", OriginAirportID="14747", Dest="LAX", DestAirportID="12892")]
    feb = [analysis_flight(1, CRSDepTime="0005"), analysis_flight(5),
           analysis_flight(6, FlightDate="2024-02-12", Dest="SEA", DestAirportID="14747")]
    package(raw, "2024-01", jan, with_parentheses=True)
    package(raw, "2024-02", feb)
    output = tmp_path / "analysis"
    run_batch(raw, output, ["2024-01", "2024-02"], ("DFW", "ORD"))

    compact = read_rows(output / "2024-01" / "flights.csv.gz")
    assert len(compact) == 3
    assert [(row["is_selected_origin"], row["is_selected_destination"]) for row in compact] == [
        ("True", "True"), ("False", "True"), ("True", "False")]
    assert compact[0]["CarrierDelay"] == "0" and compact[0]["WeatherDelay"] == ""
    assert compact[0]["ActualElapsedTime"] == "135" and compact[0]["DayOfWeek"] == "5"

    issues = read_rows(output / "cross_month_issues.csv")
    assert {row["Flight_Number_Reporting_Airline"] for row in issues} == {"1", "5"}
    same_key = [row for row in issues if row["Flight_Number_Reporting_Airline"] == "1"]
    assert {row["source_file_month"] for row in same_key} == {"2024-01", "2024-02"}
    assert {row["cross_month_status"] for row in same_key} == {"same_key_across_months"}
    assert len(read_rows(output / "2024-02" / "flights.csv.gz")) == 1

    unknown = read_rows(output / "unknown_by_origin_carrier_month.csv")
    assert (unknown[0]["source_file_month"], unknown[0]["n"], unknown[0]["n_U"], unknown[0]["unknown_primary_rate"]) == (
        "2024-01", "2", "1", "0.5")
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["same_key_cross_month_groups"] == 1
    run_batch(raw, output, ["2024-01", "2024-02"], ("DFW", "ORD"))


def test_only_explicit_years_and_months_are_loaded(tmp_path):
    package(tmp_path, "2024-01", [analysis_flight(1)], with_parentheses=True)
    package(tmp_path, "2025-01", [analysis_flight(1)])
    months = package_months(tmp_path)
    assert set(months) == {"2024-01", "2025-01"}
    output = tmp_path / "analysis"
    run_batch(tmp_path, output, ["2024-01"], ("DFW", "ORD"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["months"] == ["2024-01"]
    assert not (output / "2025-01").exists()
