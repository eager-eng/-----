"""Score branch, marginal and high-risk-tail probabilities from frozen predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from q3_inputs import CLASSES, ROOT
from q3_models import intercept_slope, score_event


def matched_predictions(output, audit, end_month=None):
    columns = ["source_row", "is_selected_origin", "outcome_code"]
    records = []
    for forecast in sorted((output / "predictions").glob("*.csv.gz")):
        month = forecast.name[:7]
        if end_month is not None and month > end_month:
            continue
        labels = pd.read_csv(audit / month / "flights.csv.gz", usecols=columns)
        labels = labels.loc[labels.is_selected_origin, ["source_row", "outcome_code"]]
        frame = pd.read_csv(forecast).merge(labels, left_on="row", right_on="source_row", validate="one_to_one")
        if len(frame) != len(labels):
            raise ValueError(f"Frozen diagnostics require a resolved cutoff for every plan record in {month}")
        records.append(frame.loc[frame.outcome_code.isin(CLASSES)])
    return pd.concat(records, ignore_index=True)


def diagnose(output, audit, model, end_month=None):
    frame = matched_predictions(output, audit, end_month)
    known = frame.outcome_code.to_numpy()
    calibrated = frame[["p_" + name for name in CLASSES]].to_numpy()
    if end_month is not None:
        y = np.isin(known, CLASSES[:3])
        risk = calibrated[:, :3].sum(axis=1)
        result = score_event(y, risk)
        intercept, slope = intercept_slope(y, risk)
        result.update(first_month=str(frame.month.min()), last_month=str(frame.month.max()),
                      calibration_intercept=intercept, calibration_slope=slope,
                      predicted_minus_observed=result["predicted"] - result["observed"])
        pd.DataFrame([result]).to_csv(output / "same_period_comparison.csv", index=False, encoding="utf-8-sig")
        return
    raw = calibrated ** model["calibration"][model["primary_info"]]
    raw /= raw.sum(axis=1, keepdims=True)

    rows, tails = [], []
    all_rows = np.ones(len(known), dtype=bool)
    not_cancelled = known != "C"
    normal_arrival = np.isin(known, CLASSES[2:])
    for version, p in (("raw", raw), ("calibrated", calibrated)):
        for indices, label, eligibility, denominator in (
                ((0,), "C", all_rows, None), ((1,), "V", all_rows, None),
                ((2,), "L60", all_rows, None), ((0, 1, 2), "D60", all_rows, None),
                ((1,), "V_given_not_C", not_cancelled, p[:, 1:].sum(axis=1)),
                ((2,), "L60_given_arrival", normal_arrival, p[:, 2:].sum(axis=1)),
                ((3,), "M60_given_arrival", normal_arrival, p[:, 2:].sum(axis=1)),
                ((4,), "T_given_arrival", normal_arrival, p[:, 2:].sum(axis=1))):
            risk = p[:, indices].sum(axis=1)
            if denominator is not None:
                risk = risk / denominator
            y = np.isin(known[eligibility], np.asarray(CLASSES)[list(indices)])
            risk = risk[eligibility]
            intercept, slope = intercept_slope(y, risk)
            rows.append({"version": version, "event": label, **score_event(y, risk),
                         "calibration_intercept": intercept, "calibration_slope": slope})
            if denominator is None:
                order = np.argsort(-risk, kind="stable")
                for fraction in (.01, .05):
                    high = order[:int(np.ceil(fraction * len(order)))]
                    tails.append({"version": version, "event": label, "top_fraction": fraction,
                                  "n": len(high), "observed": y[high].mean(), "predicted": risk[high].mean()})
    pd.DataFrame(rows).to_csv(output / "branch_calibration.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tails).to_csv(output / "high_risk_calibration.csv", index=False, encoding="utf-8-sig")
    frame["severe"] = np.isin(known, CLASSES[:3])
    groups = []
    for field in ("carrier", "dest"):
        group = frame.groupby(["origin", field], observed=True).agg(
            n=("severe", "size"), events=("severe", "sum"), predicted=("p_D60", "mean")).reset_index()
        group = group.rename(columns={field: "group_value"})
        group["group_field"] = field
        group["observed"] = group.events / group.n
        group["residual"] = group.observed - group.predicted
        groups.append(group)
    pd.concat(groups, ignore_index=True).to_csv(output / "carrier_route_calibration.csv", index=False,
                                                encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", type=int, choices=[2025, 2026])
    parser.add_argument("--end-month")
    settings = parser.parse_args()
    root = ROOT / "outputs/问题三计算结果"
    batch = "q3_holdout2025" if settings.year == 2025 else "q3_later2026"
    freeze = json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))
    diagnose(root / str(settings.year), ROOT / "data/analysis" / batch, freeze, settings.end_month)
