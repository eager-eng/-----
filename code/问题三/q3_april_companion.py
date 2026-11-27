"""Evaluate the pre-specified April companion after all 2024 labels were public."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from q3_inputs import ROOT, INVALID_TIME, airport_metadata, historical, history_tables, load_period, release_registry
from q3_models import Forecast, calibrate, score_event, temperature
from q3_predeparture_prediction import block_draws, eligible, file_hash, fold_outcomes


def main():
    output = ROOT / "outputs/问题三计算结果"
    holdout = output / "2025"
    frozen = json.loads((output / "frozen_manifest.json").read_text(encoding="utf-8"))
    primary_path = output / frozen["model_file"]
    if file_hash(primary_path) != frozen["model_sha256"] or frozen["primary_info"] != "A":
        raise ValueError("The planned April comparison requires the verified Info-A freeze")
    primary, primary_temperature = joblib.load(primary_path)["models"]["A"]
    registry, metadata = release_registry(), airport_metadata()
    development = load_period(ROOT / "data/analysis/development", metadata)
    development["label_release"] = development.month.map(registry.eligible_sec).astype("int64")
    history = history_tables(development, registry)
    development, baseline = historical(development, history, registry)
    method = frozen["model"]
    folds, _ = fold_outcomes(development, baseline, registry, "A", [method])
    q4 = development.y.ge(0) & development.cutoff.ne(INVALID_TIME) & development.month.astype(str).ge("2024-10")
    q4_fit = eligible(development, registry, int(development.loc[q4, "cutoff"].min()))
    fourth = Forecast(method, "A").fit(development.loc[q4_fit])
    labels = [frame.y.to_numpy() for frame, _ in folds.values()] + [development.loc[q4, "y"].to_numpy()]
    predictions = [scores[method] for _, scores in folds.values()] + [fourth.predict(development.loc[q4])]
    update = int(pd.Timestamp("2025-03-16T00:00:00Z").timestamp())
    train = eligible(development, registry, update)
    if development.loc[train, "month"].astype(str).max() != "2024-12":
        raise ValueError("Companion training cutoff does not include December 2024")
    companion = Forecast(method, "A").fit(development.loc[train])
    companion_temperature = calibrate(np.concatenate(labels), np.vstack(predictions))
    n_train = int(train.sum())
    del development, labels, predictions, folds
    model_path = output / "april_companion.joblib"
    joblib.dump((companion, companion_temperature, history), model_path, compress=3)

    data = load_period(ROOT / "data/analysis/q3_holdout2025", metadata)
    data = data.loc[data.month.astype(str).ge("2025-04") & data.y.ge(0) & data.cutoff.ge(update)].reset_index(drop=True)
    revised, _ = historical(data.copy(), history, registry, "2024-12")
    original, _ = historical(data, joblib.load(primary_path)["history_tables"], registry, "2024-09")
    p = temperature(primary.predict(original), primary_temperature)[:, :3].sum(axis=1)
    q = temperature(companion.predict(revised), companion_temperature)[:, :3].sum(axis=1)
    y = original.y.le(2).to_numpy()
    scores = pd.DataFrame([{"model": name, **score_event(y, forecast)}
                           for name, forecast in (("year_start", p), ("april_companion", q))])
    scores.to_csv(holdout / "april_companion.csv", index=False, encoding="utf-8-sig")
    dates = pd.date_range(original.date.min(), original.date.max())
    day = dates.get_indexer(original.date)
    weights = block_draws(dates)
    errors = np.bincount(day, weights=(q-y)**2-(p-y)**2, minlength=len(dates))
    counts = np.bincount(day, minlength=len(dates))
    low, high = np.quantile((weights @ errors) / (weights @ counts), [.025, .975])
    manifest = {"first_service_date": str(dates[0].date()), "last_service_date": str(dates[-1].date()),
                "fit_n": n_train, "test_n": len(y), "temperature": companion_temperature,
                "brier_difference": float(errors.sum()/len(y)), "lo": float(low), "hi": float(high),
                "model_sha256": file_hash(model_path), "primary_model_sha256": frozen["model_sha256"],
                "script_sha256": file_hash(Path(__file__))}
    (holdout / "april_companion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
