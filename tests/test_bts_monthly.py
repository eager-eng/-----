import csv
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from bts_monthly import classify, normalize_clock, process_month


def flight(number, **updates):
    row = {
        "FlightDate": "2024-01-05", "DOT_ID_Reporting_Airline": "19805",
        "Reporting_Airline": "AA", "Flight_Number_Reporting_Airline": str(number),
        "OriginAirportID": "11298", "DestAirportID": "13930",
        "Origin": "DFW", "Dest": "ORD", "CRSDepTime": "5", "CRSArrTime": "2400",
        "Cancelled": "0", "CancellationCode": "", "Diverted": "0",
        "DivAirportLandings": "0", "Div1Airport": "", "Div2Airport": "",
        "DivReachedDest": "", "DivArrDelay": "", "ArrTime": "2359",
        "ArrDelay": "0", "ArrDelayMinutes": "0", "ArrDel15": "0", "Flights": "1",
    }
    row.update(updates)
    return row


@pytest.mark.parametrize("raw,clock,offset", [("5", "0005", 0), ("2400", "0000", 1), ("1260", None, None)])
def test_local_clock(raw, clock, offset):
    result = normalize_clock(pd.Series([raw]))
    if clock is None:
        assert result.iloc[0].isna().all()
    else:
        assert tuple(result.iloc[0]) == (clock, offset)


def test_thresholds_and_partial_labels():
    delays = (-5, 0, 14, 15, 44, 45, 59, 60, 89, 90)
    rows = [flight(i, ArrDelay=f"{d}.0", ArrDelayMinutes=str(max(d, 0)), ArrDel15=str(int(d >= 15)))
            for i, d in enumerate(delays)]
    rows += [flight(100, ArrDelay="", ArrDelayMinutes="", ArrDel15="1"),
             flight(101, ArrDelay="15.1", ArrDelayMinutes="", ArrDel15="")]
    labels = classify(pd.DataFrame(rows))
    assert "".join(labels.outcome_code) == "TTTMMMMLLLUU"
    assert labels.severe_45.sum() == 5
    assert labels.severe_60.sum() == 3
    assert labels.severe_90.sum() == 1
    assert labels.iloc[-2].final_reached
    assert pd.isna(labels.iloc[-2].severe_60)
    assert labels.iloc[-1].quality_invalid_field


def test_diversion_and_cancel_preserve_process_and_partial_outcome():
    labels = classify(pd.DataFrame([
        flight(1, Cancelled="1", CancellationCode="A", Diverted="0", DivAirportLandings="9",
               Div1Airport="STL", Div2Airport="DFW", ArrTime="", ArrDelay="", ArrDelayMinutes="", ArrDel15=""),
        flight(2, Diverted="1", DivReachedDest="1", Div1Airport="STL", DivAirportLandings="1",
               ArrDelay="", ArrDelayMinutes="", ArrDel15=""),
        flight(3, Cancelled="1", ArrDelay="10"),
    ]))
    assert list(labels.outcome_code) == ["C", "V", "U"]
    assert labels.iloc[0].ever_diverted
    assert labels.iloc[0].return_cancel_code_9
    assert labels.iloc[1].final_reached
    assert pd.isna(labels.iloc[1].final_late_60)
    assert labels.iloc[2].quality_status_conflict


def test_monthly_files_and_counts(tmp_path):
    known = flight(1)
    rows = [known, known.copy(), flight(2, Flights="2"), flight(3, ArrDelay="60", ArrDelayMinutes="60", ArrDel15="1")]
    raw = tmp_path / "flights.csv"
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(known))
        writer.writeheader()
        writer.writerows(rows)
    archive = tmp_path / "flights.zip"
    with ZipFile(archive, "w") as output:
        output.write(raw, raw.name)
        output.writestr("readme.txt", "fixture")

    target = tmp_path / "output"
    process_month(archive, target, "2024-01")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    totals = pd.read_csv(target / "airport_summary.csv").query("origin == 'ALL'").iloc[0]
    assert manifest["counts"] == {"raw_rows": 4, "exact_duplicates": 1, "identity_unresolved": 1, "included": 2}
    assert totals.n == totals.n_T + totals.n_L == 2
    assert totals.on_time_confirmed_rate == 0.5
    exceptions = pd.read_csv(target / "exceptions.csv")
    assert list(exceptions.audit_disposition) == ["exact_duplicate", "identity_unresolved"]
