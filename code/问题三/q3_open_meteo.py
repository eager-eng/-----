"""Conservative 48-hour archived GFS temperature fallback when IEM coverage is unavailable."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from q3_inputs import ROOT, INVALID_TIME, WX

URL="https://previous-runs-api.open-meteo.com/v1/forecast"
TEMPERATURE="temperature_2m_previous_day2"


def airport_coords(root=ROOT):
    return pd.read_csv(root/"data/raw/iem/openflights_airports.dat",header=None,
                       keep_default_na=False).loc[:,[4,6,7]].drop_duplicates(4).set_index(4)


def acquire(airport,year,lat,lon,folder,end):
    target=folder/f"{airport}_{year}.csv.gz"
    url=URL+"?"+urlencode({"latitude":lat,"longitude":lon,"hourly":TEMPERATURE,
                           "models":"gfs_seamless","timezone":"GMT","timeformat":"unixtime",
                           "start_date":f"{year}-01-01","end_date":f"{year}-{end}"})
    if target.exists():
        return {"airport":airport,"year":year,"status":"cached","source":url}
    output=subprocess.run(["curl.exe","-fLsS","--compressed","--retry","1","--max-time","55",url],
                          capture_output=True)
    if output.returncode:
        return {"airport":airport,"year":year,"status":output.stderr.decode(errors="replace").strip(),"source":url}
    response=json.loads(output.stdout)
    rows=pd.DataFrame({"target":response["hourly"]["time"],"temperature_c":response["hourly"][TEMPERATURE]})
    count=int(rows.temperature_c.notna().sum())
    if count:
        temp=target.with_suffix(".tmp")
        rows.to_csv(temp,index=False,compression={"method":"gzip","mtime":0})
        temp.replace(target)
    return {"airport":airport,"year":year,"status":"available" if count else "empty","hours":count,"source":url,
            "grid_lat":response["latitude"],"grid_lon":response["longitude"]}


def enrich(frame,buffer_hours=6,root=ROOT):
    """Use forecasts fixed at 48 hours lead with a conservative publication margin."""
    folder=root/"data/raw/open_meteo"
    box=np.full((len(frame),len(WX)),np.nan,dtype=np.float32)
    width=len(WX)//2
    box[:,[width-1,2*width-1]]=0
    cutoff=frame.cutoff.to_numpy()
    years=range(frame.date.dt.year.min()-1,frame.date.dt.year.max()+2)
    for j,(location,clock) in enumerate((("origin","dep_utc"),("dest","arr_utc"))):
        target=frame[clock].to_numpy()
        for code,rows in frame.groupby(location,observed=True).indices.items():
            paths=[folder/f"{code}_{y}.csv.gz" for y in years]
            forecasts=[pd.read_csv(p) for p in paths if p.exists()]
            if not forecasts:
                continue
            table=pd.concat(forecasts,ignore_index=True).drop_duplicates("target").set_index("target")
            valid=rows[(target[rows]!=INVALID_TIME)&(cutoff[rows]!=INVALID_TIME)]
            nearest=((target[valid]+1800)//3600)*3600
            allowed=nearest-48*3600+buffer_hours*3600<=cutoff[valid]
            selected=table.temperature_c.reindex(nearest[allowed]).to_numpy()
            chosen=valid[allowed][np.isfinite(selected)]
            box[chosen,j*width]=selected[np.isfinite(selected)]*1.8+32
            box[chosen,(j+1)*width-2]=48
            box[chosen,(j+1)*width-1]=1
    return frame.join(pd.DataFrame(box,index=frame.index,columns=WX))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports",nargs="+",required=True)
    parser.add_argument("--years",nargs="+",type=int,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--end",default="12-31",help="End month and day for the latest requested year")
    args=parser.parse_args()
    folder=ROOT/"data/raw/open_meteo"
    folder.mkdir(parents=True,exist_ok=True)
    locations=airport_coords()
    tasks=[(code,year,*map(float,locations.loc[code]),folder,args.end if year==max(args.years) else "12-31")
           for code in args.airports if code in locations.index for year in args.years]
    with ThreadPoolExecutor(args.workers) as workers:
        manifest=list(workers.map(lambda task:acquire(*task),tasks))
    output=folder/f"acquisition_{min(args.years)}_{max(args.years)}_{len(args.airports)}.json"
    output.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(pd.DataFrame(manifest).drop(columns="source").to_string(index=False),flush=True)


if __name__=="__main__":
    main()
