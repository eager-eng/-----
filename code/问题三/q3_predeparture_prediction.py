"""Develop, freeze and evaluate the Q3 24-hour retrospective forecasts."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from q3_inputs import (ROOT, CLASSES, INVALID_TIME, airport_metadata, historical, history_tables,
                       load_period, release_registry, weather)
from q3_models import (Forecast, EVENTS, MODELS, LGBMClassifier, calibrate, intercept_slope,
                       reliability, score_event, temperature)
from q3_open_meteo import enrich

FOLDS=[("2024Q1","2024-01","2024-03"),("2024Q2","2024-04","2024-06"),("2024Q3","2024-07","2024-09")]
FREEZE_AT=pd.Timestamp("2024-12-31T00:00:00Z")


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def align(p,y):
    if p.shape!=(len(y),5) or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-10,rtol=0):
        raise ValueError("Invalid five-state probability")


def event_data(y,p,event):
    indices=EVENTS[event]
    return np.isin(y,indices),p[:,indices].sum(axis=1)


def row_metrics(y,p,tag,selected_events):
    rows=[]
    for event in selected_events:
        target,pred=event_data(y,p,event) if p.shape[1]==5 else (y<=2,p[:,1])
        record=score_event(target,pred)
        rows.append({**tag,"event":event,**record})
    return rows


def profile(frame):
    sample=frame.assign(known=frame.y.ge(0),eligible=frame.cutoff.ne(INVALID_TIME),
                        double=frame.depwx_avail.eq(1)&frame.arrwx_avail.eq(1))
    return sample.groupby(["month","origin"],observed=True).agg(
        n=("row","size"),known=("known","sum"),eligible=("eligible","sum"),
        wx_departure=("depwx_avail","sum"),wx_arrival=("arrwx_avail","sum"),
        wx_both=("double","sum")).reset_index()


def eligible(development,releases,candidate_cutoff):
    return development.y.ge(0)&development.cutoff.ne(INVALID_TIME)&development.month.map(
        releases.eligible_sec).astype("int64").le(candidate_cutoff)


def fold_outcomes(development,baseline,releases,info,families,load=True):
    result={}
    audit=[]
    for label,first,last in FOLDS:
        token=development.month.astype(str)
        valid=development.y.ge(0)&development.cutoff.ne(INVALID_TIME)&token.ge(first)&token.le(last)
        cutoff=int(development.loc[valid,"cutoff"].min())
        fit=eligible(development,releases,cutoff)
        fitset=development.loc[fit]
        scoring=development.loc[valid]
        y=scoring.y.to_numpy()
        rows=scoring[["month","date","row","y","cutoff"]].copy()
        predictions={}
        for family in families:
            print(f"{label} Info-{info} {family} fit eligible={len(fitset):,}; score={len(y):,}",flush=True)
            model=None
            if family=="history":
                p=baseline[valid]
            else:
                model=Forecast(family,info,load=load).fit(fitset)
                p=model.predict(scoring)
            if family!="binary_tree":
                align(p,y)
            predictions[family]=p
            audit.append({"fold":label,"info":info,"family":family,"cutoff_utc":str(pd.Timestamp(cutoff,unit="s",tz="UTC")),
                          "latest_label_month":str(fitset.month.astype(str).max()),"n_eligible_fit":len(fitset),
                          "n_fitted":len(fitset) if model is None else model.n_training,"n_score":len(y),
                          "engine":"lagged_history" if model is None else model.engine,
                          "diversion_level":"" if model is None else model.diversion_level,
                          "convergence_warnings":0 if model is None else model.convergence_warnings})
        result[label]=(rows,predictions)
    return result,audit


def dev_scores(folds,info,releases):
    rows=[]
    method_comparison=[]
    keys=list(folds)
    for idx,(key,(frame,predictions)) in enumerate(folds.items()):
        y=frame.y.to_numpy()
        for model,p in predictions.items():
            events=("D60",) if model=="binary_tree" else ("D60","C","L60","V","R1")
            rows.extend(row_metrics(y,p,{"fold":key,"info":info,"model":model,"calibration":"raw"},events))
            if p.shape[1]!=5 or model=="history":
                continue
            old_y,old_p=[],[]
            for previous in keys[:idx]:
                before, probs=folds[previous]
                visible=before.month.map(releases.eligible_sec).astype("int64").le(frame.cutoff.min())
                old_y.append(before.loc[visible,"y"].to_numpy())
                old_p.append(probs[model][visible])
            if not old_y or sum(len(y) for y in old_y)<1000:
                continue
            value=calibrate(np.concatenate(old_y),np.concatenate(old_p))
            calibrated=temperature(p,value)
            rows.extend(row_metrics(y,calibrated,{"fold":key,"info":info,"model":model,
                                                  "calibration":"prior_fold_temperature"},events))
            true=(y<=2).astype(int)
            raw_loss=(p[:,:3].sum(axis=1)-true)**2
            cal_loss=(calibrated[:,:3].sum(axis=1)-true)**2
            method_comparison.append({"fold":key,"info":info,"model":model,"training_n":sum(map(len,old_y)),
                                      "n":len(y),"brier_difference":np.mean(cal_loss-raw_loss),"temperature":value})
    return rows,method_comparison


def develop(args,registry,metadata):
    args.output.mkdir(parents=True,exist_ok=True)
    dev=load_period(args.development,metadata)
    if set(dev.date.dt.year.unique())!={2023,2024}:
        raise ValueError("Development is restricted to 2023–2024")
    if len(dev)!=3247351 or dev.y.value_counts().to_dict()!={4:2466313,3:477525,2:252997,0:42319,1:8196,-1:1}:
        raise ValueError("Development labels do not reconcile to Q1")
    dev["label_release"]=dev.month.map(registry.eligible_sec).astype("int64")
    hist=history_tables(dev,registry)
    dev,baseline=historical(dev,hist,registry)
    if args.weather_source=="open_meteo":
        dev=enrich(dev,buffer_hours=args.weather_buffer)
    else:
        dev=weather(dev,metadata,buffer_hours=args.weather_buffer)
    audit_profile=profile(dev)
    audit_profile.to_csv(args.output/"development_coverage.csv",index=False,encoding="utf-8-sig")
    clocks=dev.groupby(["month","dep_status","arr_status"],observed=True).size().rename("n").reset_index()
    clocks.to_csv(args.output/"development_clock_audit.csv",index=False,encoding="utf-8-sig")
    cover=dev.groupby("month",observed=True).depwx_avail.mean()
    available_B=all(any(cover.get(f"2024-{month:02d}",0)>0 for month in group)
                    for group in (range(1,4),range(4,7),range(7,10))) and any(
                    cover.get(f"2023-{month:02d}",0)>0 for month in range(1,10))
    print("Development weather:",str(cover.to_dict()),"; full B supported:",available_B,flush=True)
    all_rows=[]
    all_cal=[]
    fit_cutoff=int(FREEZE_AT.timestamp())
    freeze_fit=eligible(dev,registry,fit_cutoff)
    if dev.loc[freeze_fit,"month"].astype(str).max()!="2024-09":
        raise ValueError("Year-start model must freeze at September 2024 by release registry")
    reference_signature=[file_hash(Path(__file__)),file_hash(Path(__file__).with_name("q3_models.py")),
                         file_hash(Path(__file__).with_name("q3_inputs.py")),
                         file_hash(ROOT/"data/audit/bts_release_calendar.csv"),str(LGBMClassifier)]
    if args.a_cache and args.a_cache.exists():
        signature,comparisons,audit=joblib.load(args.a_cache)
        if signature[1:]!=reference_signature[1:]:
            raise ValueError("Out-of-fold cache belongs to different code or release dates")
    else:
        comparisons,audit=fold_outcomes(dev,baseline,registry,"A",MODELS)
        if args.a_cache:
            joblib.dump((reference_signature,comparisons,audit),args.a_cache,compress=0)
    rows,cal_rows=dev_scores(comparisons,"A",registry)
    all_rows+=rows; all_cal+=cal_rows
    pooled=[]
    for family in MODELS:
        data=np.concatenate([frame.y.to_numpy() for frame,_ in comparisons.values()])
        forecast=np.concatenate([prediction[family] for _,prediction in comparisons.values()])
        pooled.extend(row_metrics(data,forecast,{"fold":"pooled","info":"A","model":family,"calibration":"raw"},
                                  ("D60",) if family=="binary_tree" else ("D60","C","L60","V","R1")))
    all_rows+=pooled
    table=pd.DataFrame(pooled)
    winner=table.loc[table.event.eq("D60") & table.model.isin(["logistic","joint_tree","chain_tree"])].sort_values(
        ["brier","logloss","ap"],ascending=[True,True,False]).iloc[0].model
    print("Selected 5-state architecture",winner,flush=True)
    # Complete the information-set comparison using the selected architecture when exposure spans the folds.
    tracks={"A":comparisons}
    info_comparison=[]
    primary_info="A"
    if available_B:
        later,rows_audit=fold_outcomes(dev,baseline,registry,"B",[winner])
        audit+=rows_audit
        rows,cal_rows=dev_scores(later,"B",registry)
        all_rows+=rows; all_cal+=cal_rows
        tracks["B"]=later
        core=np.concatenate([frame.index.to_numpy() for frame,_ in comparisons.values()])
        targets=dev.loc[core,"y"].to_numpy()
        both=dev.loc[core,"depwx_avail"].eq(1).to_numpy()&dev.loc[core,"arrwx_avail"].eq(1).to_numpy()
        for info,folds in tracks.items():
            candidate=np.concatenate([prediction[winner] for _,prediction in folds.values()])
            if not np.array_equal(np.concatenate([frame.index.to_numpy() for frame,_ in folds.values()]),core):
                raise ValueError("Development A/B scores are not from identical rows")
            for scope,mask in (("all",np.ones(len(core),dtype=bool)),("both_covered",both)):
                if mask.any():
                    info_comparison+=row_metrics(targets[mask],candidate[mask],{"fold":"pooled","info":info,
                                                   "scope":scope,"model":winner},("D60","C","L60"))
        info_table=pd.DataFrame(info_comparison)
        scores=info_table.loc[info_table.scope.eq("all")&info_table.event.eq("D60")]
        primary_info=scores.sort_values(["brier","logloss","info"]).iloc[0]["info"]
    fit_artifacts={}
    for info, folds in tracks.items():
        predictions=[]
        labels=[]
        for frame,probs in folds.values():
            permitted=frame.month.map(registry.eligible_sec).astype("int64").le(fit_cutoff)
            labels.append(frame.loc[permitted,"y"].to_numpy())
            predictions.append(probs[winner][permitted])
        validation=[r for r in all_cal if r["model"]==winner and r["info"]==info]
        retained=bool(validation and np.average([r["brier_difference"] for r in validation],
                                                 weights=[r["n"] for r in validation])<0)
        y_oof=np.concatenate(labels); p_oof=np.concatenate(predictions)
        value=calibrate(y_oof,p_oof) if retained else 1.0
        if retained:
            all_rows+=row_metrics(y_oof,temperature(p_oof,value),{"fold":"pooled","info":info,"model":winner,
                                         "calibration":"apparent_fitted_temperature"},("D60","C","L60","V","R1"))
        print(f"Freezing {winner} Info-{info}, temperature={value:.5f}",flush=True)
        fit_artifacts[info]=(Forecast(winner,info).fit(dev.loc[freeze_fit]),value)
    outputs={"development_metrics":pd.DataFrame(all_rows),"development_folds":pd.DataFrame(audit),
             "development_calibration":pd.DataFrame(all_cal)}
    for name,value in outputs.items():
        value.to_csv(args.output/(name+".csv"),index=False,encoding="utf-8-sig")
    if info_comparison:
        pd.DataFrame(info_comparison).to_csv(args.output/"development_info_comparison.csv",index=False,encoding="utf-8-sig")
    if args.model_path.is_relative_to(args.output):
        args.model_path.parent.mkdir(exist_ok=True,parents=True)
    frozen_history=history_tables(dev.loc[freeze_fit],registry)
    joblib.dump({"models":fit_artifacts,"history_tables":frozen_history,"history_last":"2024-09",
                 "weather_buffer":args.weather_buffer,"freeze_at":str(FREEZE_AT),
                 "weather_source":args.weather_source,
                 "development_version":winner},args.model_path,compress=3)
    fitted_months=set(dev.loc[freeze_fit,"month"].astype(str))
    manifests={p.parent.name:file_hash(p) for p in sorted(args.development.glob("20??-??/manifest.json"))
               if p.parent.name in fitted_months}
    manifest={"input_months":manifests,"earliest_model_utc":str(FREEZE_AT),"last_fit_label_month":"2024-09",
              "fit_n":int(freeze_fit.sum()),"model":winner,"versions":sorted(fit_artifacts),
              "primary_info":primary_info,
              "engines":{info:model.engine for info,(model,_) in fit_artifacts.items()},
              "convergence_warnings":{info:model.convergence_warnings for info,(model,_) in fit_artifacts.items()},
              "calibration":{info:value for info,(_,value) in fit_artifacts.items()},
              "diversion":{info:model.diversion_level for info,(model,_) in fit_artifacts.items()},
              "weather_status":"coverage spans development folds" if available_B else "weather coverage insufficient",
              "weather_source":args.weather_source,
              "weather_availability_proxy_hours":args.weather_buffer,
              "release_calendar_sha256":file_hash(ROOT/"data/audit/bts_release_calendar.csv"),
              "airport_metadata_sha256":file_hash(ROOT/"data/raw/iem/openflights_airports.dat"),
              "archive_hashes": {p.name:file_hash(p) for p in sorted((ROOT/"data/raw/open_meteo").glob("*_202[34].csv.gz"))}
              if args.weather_source=="open_meteo" else {},
              "model_file":args.model_path.name,"model_sha256":file_hash(args.model_path),
              "scripts":{p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob("q3_*.py"))}}
    (args.output/"frozen_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Frozen manifest written",flush=True)


def block_draws(dates,reps=1000,block=7):
    rng=np.random.default_rng(2026)
    weights=np.zeros((reps,len(dates)),dtype=np.float32)
    month=np.asarray(dates.month)
    seasons=(month%12)//3
    starts=np.flatnonzero(np.r_[True,(seasons[1:]!=seasons[:-1])|(dates.year[1:]!=dates.year[:-1])])
    for start,stop in zip(starts,np.r_[starts[1:],len(dates)]):
        n=stop-start
        k=min(block,n)
        times=rng.integers(start,stop-k+1,size=(reps,(n+k-1)//k,1))+np.arange(k)
        np.add.at(weights,(np.arange(reps)[:,None],times.reshape(reps,-1)[:,:n]),1)
    return weights


def uncertainty(frame,forecasts):
    dates=pd.date_range(frame.date.min(),frame.date.max())
    index=dates.get_indexer(frame.date)
    weights=block_draws(dates)
    y=frame.y.to_numpy()
    rows=[]
    for event in ("D60","C","L60"):
        for first,second in (("A","B0"),("B","B0"),("B","A"),("A","A_raw")):
            if first not in forecasts or second not in forecasts:
                continue
            observed,p=event_data(y,forecasts[first],event)
            _,q=event_data(y,forecasts[second],event)
            for scope, mask in (("all",np.ones(len(y),dtype=bool)),
                                ("both_covered",frame.depwx_avail.eq(1).to_numpy()&frame.arrwx_avail.eq(1).to_numpy())):
                if scope=="both_covered" and first!="B":
                    continue
                day_n=np.bincount(index[mask],minlength=len(dates))
                if day_n.sum()==0:
                    continue
                day_loss=np.bincount(index[mask],weights=((p-observed)**2-(q-observed)**2)[mask],minlength=len(dates))
                totals=weights@day_n
                valid=totals>0
                values=(weights@day_loss)[valid]/totals[valid]
                low,high=np.quantile(values,[.025,.975])
                rows.append({"event":event,"first":first,"second":second,"scope":scope,
                             "n":int(day_n.sum()),"brier_difference":day_loss.sum()/day_n.sum(),
                             "lo":low,"hi":high,"valid_reps":int(valid.sum()),"block":7})
    return pd.DataFrame(rows)


def evaluate(args,registry,metadata):
    manifest=json.loads((args.output/"frozen_manifest.json").read_text(encoding="utf-8"))
    if file_hash(args.model_path)!=manifest["model_sha256"]:
        raise ValueError("Frozen model artifact hash differs")
    package=joblib.load(args.model_path)
    data=load_period(args.test,metadata)
    if data.date.dt.year.min()<2025:
        raise ValueError("Post-freeze evaluation expects 2025 or later")
    data,baseline=historical(data,package["history_tables"],registry,package["history_last"])
    if package.get("weather_source")=="open_meteo":
        data=enrich(data,buffer_hours=package["weather_buffer"])
    else:
        data=weather(data,metadata,buffer_hours=package["weather_buffer"])
    eligible_mask=data.cutoff.ge(int(FREEZE_AT.timestamp()))
    forecast={"B0":baseline[eligible_mask]}
    all_eligible=data.loc[eligible_mask].reset_index(drop=True)
    for name,(model,value) in package["models"].items():
        p=model.predict(all_eligible)
        align(p,all_eligible.y)
        forecast[name+"_raw"]=p
        forecast[name]=temperature(p,value) if value!=1 else p
    year=str(int(data.date.dt.year.min()))
    period=args.output/year
    period.mkdir(parents=True,exist_ok=True)
    profile(data).to_csv(period/"coverage.csv",index=False,encoding="utf-8-sig")
    all_eligible["both_covered"]=all_eligible.depwx_avail.eq(1)&all_eligible.arrwx_avail.eq(1)
    known=all_eligible.y.ge(0).to_numpy()
    scored=all_eligible.loc[known].reset_index(drop=True)
    known_prob={key:value[known] for key,value in forecast.items()}
    panel=manifest.get("primary_info","A")
    results=[]
    joint_scores=[]
    curves=[]
    for name,p in known_prob.items():
        if name.endswith("_raw") and np.array_equal(p,known_prob[name.removesuffix("_raw")]):
            continue
        targets=scored.y.to_numpy()
        target_probability=p[np.arange(len(p)),targets]
        joint_brier=(p*p).sum(axis=1)-2*target_probability+1
        joint_logloss=-np.log(np.clip(target_probability,1e-10,1))
        results.extend(row_metrics(targets,p,{"year":year,"model":name,"scope":"all"},EVENTS))
        joint_scores.append({"model":name,"scope":"all","n":len(p),"brier":joint_brier.mean(),
                             "logloss":joint_logloss.mean()})
        if "B" in known_prob and name in ("A","B","B0"):
            both=scored.both_covered.to_numpy()
            if both.sum()>10:
                results.extend(row_metrics(scored.loc[both,"y"].to_numpy(),p[both],
                                           {"year":year,"model":name,"scope":"both_covered"},EVENTS))
                joint_scores.append({"model":name,"scope":"both_covered","n":int(both.sum()),
                                     "brier":joint_brier[both].mean(),"logloss":joint_logloss[both].mean()})
        for event in ("D60","C","L60"):
            y,phat=event_data(scored.y.to_numpy(),p,event)
            r=reliability(y,phat)
            r["model"]=name
            r["event"]=event
            r["intercept"],r["slope"]=intercept_slope(y,phat)
            curves.append(r)
    results=pd.DataFrame(results)
    for scope,sub in results.loc[results.model.eq("B0")].groupby("scope"):
        matches=results.scope.eq(scope)
        results.loc[matches,"bss"]=1-results.loc[matches,"brier"]/results.loc[matches,"event"].map(sub.set_index("event").brier)
    risks=scored[["date","month","origin","season","tod","carrier","dest","both_covered","d60"]].copy()
    risks["pD60"]=known_prob[panel][:,:3].sum(axis=1)
    by=[]
    for dimensions in (["origin","season"],["origin","tod"],["origin","month"],["both_covered"]):
        groups=risks.groupby(dimensions,observed=True).agg(n=("d60","size"),observed=("d60","mean"),
                                                          predicted=("pD60","mean")).reset_index()
        groups["residual"]=groups.observed-groups.predicted
        groups["slice"]="*".join(dimensions)
        by.append(groups)
    days=risks.groupby("date",observed=True).agg(n=("d60","size"),observed=("d60","mean"),
                                               predicted=("pD60","mean")).reset_index()
    days["residual"]=days.observed-days.predicted
    days["empirical_top5"]=days.observed.ge(days.observed.quantile(.95))
    outputs={"metrics":results,"multiclass_metrics":pd.DataFrame(joint_scores),
             "calibration":pd.concat(curves,ignore_index=True),"failure_groups":pd.concat(by,ignore_index=True),
             "daily":days,"paired_brier_intervals":uncertainty(scored,known_prob)}
    for name,table in outputs.items():
        table.to_csv(period/(name+".csv"),index=False,encoding="utf-8-sig")
    predictions=period/"predictions"
    predictions.mkdir(exist_ok=True)
    for month, positions in all_eligible.groupby("month",observed=True).indices.items():
        cols=["month","row","date","origin","dest","carrier","dep_utc","arr_utc","cutoff","depwx_avail","arrwx_avail"]
        published=all_eligible.iloc[positions][cols].copy()
        published["model_version"]=manifest["model_sha256"][:12]+":"+panel
        for k,event in enumerate(CLASSES):
            published["p_"+event]=forecast[panel][positions,k]
        published["p_D60"]=forecast[panel][positions,:3].sum(axis=1)
        published.to_csv(predictions/(str(month)+".csv.gz"),index=False,float_format="%.9g",
                         compression={"method":"gzip","mtime":0},encoding="utf-8")
    report={"scope":len(data),"eligible":len(all_eligible),"known":len(scored),"unknown":int(data.y.lt(0).sum()),
            "scored_events":int(scored.d60.sum()),"predicted_model":panel,"pairs":len(days),
            "frozen_model_sha256":manifest["model_sha256"],
            "monthly_inputs":{p.parent.name:file_hash(p) for p in sorted(args.test.glob("20??-??/manifest.json"))},
            "prediction_sha256":{p.name:file_hash(p) for p in sorted(predictions.glob("*.csv.gz"))}}
    (period/"evaluation_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(report,flush=True)
    from q3_figures import render_evaluation
    render_evaluation(period,args.figures,year,panel)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=["develop","evaluate"])
    parser.add_argument("--development",type=Path,default=ROOT/"data/analysis/development")
    parser.add_argument("--test",type=Path,default=ROOT/"data/analysis/q3_holdout2025")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/问题三计算结果")
    parser.add_argument("--figures",type=Path,default=ROOT/"figures/问题三计算结果")
    parser.add_argument("--model-path",type=Path)
    parser.add_argument("--weather-buffer",type=int,default=5)
    parser.add_argument("--weather-source",choices=["iem","open_meteo"],default="iem")
    parser.add_argument("--a-cache",type=Path)
    args=parser.parse_args()
    args.model_path=args.model_path or args.output/"year_start_frozen.joblib"
    return args


if __name__=="__main__":
    options=parse_args()
    registry=release_registry()
    metadata=airport_metadata()
    if options.task=="develop":
        develop(options,registry,metadata)
    else:
        evaluate(options,registry,metadata)
