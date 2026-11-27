"""Verify the frozen Q3 forecasts and their separation from holdout labels."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from q3_inputs import ROOT, CLASSES


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def verify(output,year):
    freeze=json.loads((output/"frozen_manifest.json").read_text(encoding="utf-8"))
    if any(month>"2024-09" for month in freeze["input_months"]):
        raise ValueError("Year-start model references unavailable outcome packages")
    directory=output/str(year)
    audit=json.loads((directory/"evaluation_manifest.json").read_text(encoding="utf-8"))
    if audit["frozen_model_sha256"]!=freeze["model_sha256"]:
        raise ValueError("Holdout model is inconsistent with the frozen model")
    if set(audit["monthly_inputs"]) & set(freeze["input_months"]):
        raise ValueError("Training and holdout inputs overlap")
    n=0
    for filename,expected in audit["prediction_sha256"].items():
        path=directory/"predictions"/filename
        if digest(path)!=expected:
            raise ValueError(f"Prediction hash differs: {filename}")
        frame=pd.read_csv(path)
        if "y" in frame or frame[["month","row"]].duplicated().any():
            raise ValueError(f"Labels or duplicate keys in frozen prediction file: {filename}")
        if not (frame.dep_utc-frame.cutoff).eq(86400).all():
            raise ValueError(f"Forecast cutoff differs from 24 real hours: {filename}")
        p=frame[["p_"+state for state in CLASSES]].to_numpy()
        if (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=3e-9,rtol=0):
            raise ValueError(f"Five state probability invalid: {filename}")
        if not np.allclose(p[:,:3].sum(axis=1),frame.p_D60,atol=3e-9,rtol=0):
            raise ValueError(f"Severe disruption probability invalid: {filename}")
        n+=len(frame)
    if n!=audit["eligible"]:
        raise ValueError("Prediction row count does not reconcile")
    table=pd.read_csv(directory/"metrics.csv")
    for scope,rows in table.groupby("scope"):
        if rows.n.nunique()!=1 or (scope=="all" and rows.n.iat[0]!=audit["known"]):
            raise ValueError(f"Pairwise metric cohorts differ in {scope}")
    print(f"{year}: verified {n:,} predictions over {len(audit['prediction_sha256'])} months")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("years",nargs="+",type=int)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/问题三计算结果")
    args=parser.parse_args()
    for year in args.years:
        verify(args.output,year)
