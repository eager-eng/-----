"""Freeze and score the delay-threshold and schedule-density sensitivity models."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from q3_inputs import ROOT, airport_metadata, historical, load_period, release_registry
from q3_models import Forecast, score_event
from q3_open_meteo import enrich
from q3_predeparture_prediction import FREEZE_AT, eligible, file_hash


def threshold_labels(frame, minutes):
    labels = frame.y.to_numpy(copy=True)
    arrived = np.isin(labels, [2, 3])
    if not np.isfinite(frame.delay.to_numpy()[arrived]).all():
        raise ValueError("A delayed arrival has no recorded delay in minutes")
    labels[arrived] = np.where(frame.delay.to_numpy()[arrived] >= minutes, 2, 3)
    return labels


def inputs(batch, registry, metadata, package, info):
    frame = load_period(batch, metadata)
    frame, _ = historical(frame, package["history_tables"], registry, package["history_last"])
    if info == "B":
        if package["weather_source"] != "open_meteo":
            raise ValueError("Sensitivity run requires the same archived temperature features as the frozen model")
        frame = enrich(frame, buffer_hours=package["weather_buffer"])
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["freeze", "evaluate"])
    parser.add_argument("--development", type=Path, default=ROOT / "data/analysis/development")
    parser.add_argument("--test", type=Path, default=ROOT / "data/analysis/q3_holdout2025")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/问题三计算结果")
    args = parser.parse_args()
    base_path = args.output / "year_start_frozen.joblib"
    freeze = json.loads((args.output / "frozen_manifest.json").read_text(encoding="utf-8"))
    if file_hash(base_path) != freeze["model_sha256"]:
        raise ValueError("Main model hash differs")
    main_model = joblib.load(base_path)
    info = freeze["primary_info"]
    registry, metadata = release_registry(), airport_metadata()
    artifact = args.output / "sensitivity_frozen.joblib"
    manifest = args.output / "sensitivity_manifest.json"
    if args.task == "freeze":
        frame = inputs(args.development, registry, metadata, main_model, info)
        train = eligible(frame, registry, int(FREEZE_AT.timestamp()))
        frame = frame.loc[train].copy()
        frame["label_release"] = frame.month.map(registry.eligible_sec).astype("int64")
        if len(frame) != freeze["fit_n"]:
            raise ValueError("Sensitivity training cohort differs")
        no_load = Forecast(freeze["model"], info, load=False).fit(frame)
        models = {"D60_without_density": no_load}
        for minutes in (45, 90):
            response = frame.assign(y=threshold_labels(frame, minutes))
            models[f"D{minutes}"] = Forecast(freeze["model"], info).fit(response)
        joblib.dump(models, artifact, compress=3)
        manifest.write_text(json.dumps({"main_model_sha256": freeze["model_sha256"],
                                        "sensitivity_sha256": file_hash(artifact), "n_train": len(frame),
                                        "source_sha256": file_hash(Path(__file__)), "info": info,
                                        "thresholds": [45, 90], "calibration": "identity"}, indent=2),
                            encoding="utf-8")
        print(f"Frozen {len(models)} sensitivity models from {len(frame):,} records", flush=True)
        return

    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record["main_model_sha256"] != freeze["model_sha256"] or record["sensitivity_sha256"] != file_hash(artifact):
        raise ValueError("Sensitivity model is inconsistent with the freeze")
    frame = inputs(args.test, registry, metadata, main_model, info)
    frame = frame.loc[frame.y.ge(0) & frame.cutoff.ge(int(FREEZE_AT.timestamp()))]
    year = str(frame.date.dt.year.min())
    folder = args.output / year
    evaluation = json.loads((folder / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if len(frame) != evaluation["known"]:
        raise ValueError("Sensitivity and main evaluation cohorts differ")
    models = joblib.load(artifact)
    rows = []
    for event, model in models.items():
        minutes = 60 if event == "D60_without_density" else int(event[1:])
        label = threshold_labels(frame, minutes) <= 2
        risk = model.predict(frame)[:, :3].sum(axis=1)
        rows.append({"model": event, "delay_threshold": minutes, **score_event(label, risk)})
    pd.DataFrame(rows).to_csv(folder / "sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"Scored {len(rows)} sensitivity models on {len(frame):,} {year} records", flush=True)


if __name__ == "__main__":
    main()
