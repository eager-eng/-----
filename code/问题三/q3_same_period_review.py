"""Audit frozen Info-A scores over the same nine months and continuing carriers."""

import numpy as np
import pandas as pd

from q3_diagnostics import matched_predictions
from q3_inputs import CLASSES, ROOT
from q3_models import intercept_slope, reliability


OUTPUT = ROOT / "outputs/问题三计算结果"
AUDITS = {2025: ROOT / "data/analysis/q3_holdout2025", 2026: ROOT / "data/analysis/q3_later2026"}


def continuing_carriers():
    monthly_sets = []
    for year, folder in AUDITS.items():
        coverage = pd.read_csv(folder / "carrier_coverage.csv")
        by_month = coverage.loc[coverage.selected_origin_rows > 0].groupby("source_file_month")
        for month in range(1, 10):
            records = by_month.get_group(f"{year}-{month:02}")
            monthly_sets.append(set(records.DOT_ID_Reporting_Airline))
    return set.intersection(*monthly_sets)


def summarize(flights, total):
    event = flights.outcome_code.isin(CLASSES[:3]).to_numpy()
    risk = flights.p_D60.to_numpy()
    observed = event.mean()
    uncertainty = observed * (1 - observed)
    brier = np.mean((risk - event) ** 2)
    bins = reliability(event, risk)
    calibration_loss = np.average((bins.predicted - bins.observed) ** 2, weights=bins.n)
    resolution = np.average((bins.observed - observed) ** 2, weights=bins.n)
    intercept, slope = intercept_slope(event, risk)
    return dict(n=len(flights), events=int(event.sum()), coverage=len(flights) / total,
                carriers=flights.carrier.nunique(), observed=observed, predicted=risk.mean(),
                brier=brier, uncertainty=uncertainty, brier_minus_uncertainty=brier - uncertainty,
                skill_vs_period_mean=1 - brier / uncertainty, reliability=calibration_loss,
                resolution=resolution, within_bin_residual=brier - uncertainty - calibration_loss + resolution,
                calibration_intercept=intercept, calibration_slope=slope)


def main():
    carriers = continuing_carriers()
    rows = []
    for year, folder in AUDITS.items():
        period = OUTPUT / str(year)
        flights = matched_predictions(period, folder, f"{year}-09")
        full = summarize(flights, len(flights))
        published = pd.read_csv(period / "same_period_comparison.csv").iloc[0]
        for field in ("n", "events", "brier", "calibration_intercept", "calibration_slope"):
            np.testing.assert_allclose(full[field], published[field], rtol=0, atol=1e-8)
        common = summarize(flights.loc[flights.carrier.isin(carriers)], len(flights))
        rows.extend([dict(year=year, scope="all", **full),
                     dict(year=year, scope="present_every_month_both_years", **common)])
    pd.DataFrame(rows).to_csv(OUTPUT / "same_period_brier_and_carriers.csv", index=False)


if __name__ == "__main__":
    main()
