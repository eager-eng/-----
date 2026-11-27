"""Build time-audited Q3 features from public flight and forecast snapshots."""

from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"code"))
from bts_dst_audit import AIRPORT_ZONES, utc_candidates

CLASSES = ("C", "V", "L", "M", "T")
INVALID_TIME = np.iinfo(np.int64).min
FIELDS = ["source_file_month","source_row","FlightDate","Origin","Dest","DOT_ID_Reporting_Airline",
          "CRSDepTime_hhmm","CRSDepTime_day_offset","CRSArrTime_hhmm","CRSElapsedTime",
          "Distance","is_selected_origin","outcome_code","severe_60","ArrDelay"]
CAT = ["origin","dest","carrier","season","tod","dow"]
CALENDAR = ["dep_sin","dep_cos","arr_sin","arr_cos","distance","elapsed","holiday","peak"]
LOAD = ["day_departures","hour_departures","carrier_hour_departures"]
HISTKEYS = [["origin","season","tod"],["carrier","season"],["origin","dest"]]
HIST = [f"h{i}_{item}" for i in range(len(HISTKEYS)) for item in ("C","L","D")]
WX_FIELDS = ("tmp","dpt","wsp","vis","cig","poz","pos","p06","q06","t06_2")
WX = [f"{side}_{column}" for side in ("depwx","arrwx")
      for column in (*WX_FIELDS,"age","avail")]
SECONDS_DAY = 86400


def release_registry(root=ROOT):
    dates = pd.read_csv(root/"data/audit/bts_release_calendar.csv", dtype={"month":str})
    # The directory's date is public, but not an authenticated UTC publication instant.
    dates["eligible_utc"] = pd.to_datetime(dates.directory_date,utc=True) + pd.Timedelta(hours=36)
    dates["eligible_sec"] = dates.eligible_utc.dt.as_unit("s").astype("int64")
    return dates.set_index("month").sort_index()


def airport_metadata(root=ROOT):
    metadata = root/"data/raw/iem/openflights_airports.dat"
    airports = {code:("K"+code,zone.key) for code,zone in AIRPORT_ZONES.items()}
    if metadata.exists():
        raw = pd.read_csv(metadata,header=None,keep_default_na=False)
        rows = raw.loc[raw[4].str.fullmatch("[A-Z]{3}") & raw[5].str.fullmatch("[A-Z]{4}") &
                       raw[11].str.contains("/"),[4,5,11]]
        for code,station,zone in rows.itertuples(index=False,name=None):
            try:
                ZoneInfo(zone)
            except KeyError:
                continue
            if code not in airports:
                airports[code] = (station,zone)
    return airports


def clock_minutes(values):
    raw = pd.to_numeric(values,errors="coerce")
    return (raw//100)*60 + raw%100


def seconds(ts):
    return ts.dt.as_unit("s").astype("int64").to_numpy()


def flight_clocks(frame, metadata):
    day = pd.to_datetime(frame.FlightDate,format="%Y-%m-%d")
    wall_minute = clock_minutes(frame.CRSDepTime_hhmm)
    local = day + pd.to_timedelta(wall_minute + 1440*pd.to_numeric(
        frame.CRSDepTime_day_offset,errors="coerce"),unit="m")
    arrive_min = clock_minutes(frame.CRSArrTime_hhmm)
    elapsed = pd.to_numeric(frame.CRSElapsedTime,errors="coerce")
    dep = np.full(len(frame),INVALID_TIME,dtype=np.int64)
    status = np.full(len(frame),"origin_clock_unresolved",dtype=object)
    for airport, rows in frame.groupby("Origin",sort=False).indices.items():
        zone = metadata[airport][1]
        dep[rows] = seconds(local.iloc[rows].dt.tz_localize(
            zone,ambiguous="NaT",nonexistent="NaT").dt.tz_convert("UTC"))
        for pos in rows[dep[rows]==INVALID_TIME]:
            if pd.isna(local.iat[pos]) or pd.isna(elapsed.iat[pos]) or pd.isna(arrive_min.iat[pos]):
                continue
            target = metadata.get(frame.Dest.iat[pos])
            if not target:
                continue
            variants = utc_candidates(local.iat[pos].to_pydatetime(),ZoneInfo(zone))
            matches = []
            for item in variants:
                there=(item+pd.Timedelta(minutes=int(elapsed.iat[pos]))).astimezone(ZoneInfo(target[1]))
                if 60*there.hour+there.minute==arrive_min.iat[pos]:
                    matches.append(item)
            if len(matches)==1:
                dep[pos]=int(matches[0].timestamp())
                status[pos]="origin_fold_resolved"
    valid = dep != INVALID_TIME
    status[valid & (status!="origin_fold_resolved")] = "origin_valid"
    arr = np.full(len(frame),INVALID_TIME,dtype=np.int64)
    valid_elapsed = valid & elapsed.gt(0).to_numpy()
    arr[valid_elapsed] = dep[valid_elapsed] + 60*elapsed.iloc[np.flatnonzero(valid_elapsed)].to_numpy(dtype=np.int64)
    arr_status = np.full(len(frame),"arrival_clock_unresolved",dtype=object)
    for airport,rows in frame.groupby("Dest",sort=False).indices.items():
        place=metadata.get(airport)
        if not place:
            arr[rows]=INVALID_TIME
            arr_status[rows]="destination_zone_unmapped"
            continue
        some=rows[arr[rows]!=INVALID_TIME]
        if not len(some):
            continue
        target=pd.to_datetime(arr[some],unit="s",utc=True).tz_convert(place[1])
        match=target.hour.to_numpy()*60+target.minute.to_numpy()==arrive_min.iloc[some].to_numpy()
        arr_status[some]=np.where(match,"arrival_valid","destination_clock_conflict")
        arr[some[~match]]=INVALID_TIME
    cutoff=np.full(len(frame),INVALID_TIME,dtype=np.int64)
    cutoff[valid]=dep[valid]-SECONDS_DAY
    return day,wall_minute,arrive_min,elapsed,dep,arr,cutoff,status,arr_status


def add_load(frame):
    minute=frame.dep_utc.to_numpy()
    frame["day_departures"]=frame.groupby(["origin","date"],observed=True).origin.transform("size").astype("float32")
    for name,keys in (("hour_departures",["origin"]),("carrier_hour_departures",["origin","carrier"])):
        counts=np.full(len(frame),np.nan,dtype="float32")
        for subset in frame.groupby(keys,observed=True,sort=False).indices.values():
            index=subset[minute[subset]!=INVALID_TIME]
            times=np.sort(minute[index])
            counts[index]=np.searchsorted(times,minute[index]+3600,side="right")-np.searchsorted(
                times,minute[index]-3600,side="left")
        frame[name]=counts
    return frame


def load_period(batch, metadata):
    paths=sorted(batch.glob("20??-??/flights.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No audited monthly files in {batch}")
    frames=[]
    for path in paths:
        raw=pd.read_csv(path,usecols=FIELDS,dtype=str,keep_default_na=False)
        raw=raw.loc[raw.is_selected_origin.eq("True")].reset_index(drop=True)
        day,clock,arr_clock,elapsed,dep,arr,cutoff,dep_status,arr_status=flight_clocks(raw,metadata)
        holiday=USFederalHolidayCalendar().holidays(day.min()-pd.Timedelta(days=2),day.max()+pd.Timedelta(days=2))
        distance=pd.to_numeric(raw.Distance,errors="coerce")
        state=raw.outcome_code.map(dict(zip(CLASSES,range(5)))).fillna(-1).astype("int8")
        confirmed=raw.severe_60.map({"True":1,"False":0}).fillna(-1).astype("int8")
        known=state.ge(0)
        if not ((state[known].le(2).astype(int))==confirmed[known]).all():
            raise ValueError(f"Five-class outcomes disagree with Q1 in {path}")
        frames.append(pd.DataFrame({
            "month":raw.source_file_month,"row":raw.source_row.astype("int32"),"date":day,
            "origin":raw.Origin,"dest":raw.Dest,"carrier":raw.DOT_ID_Reporting_Airline,
            "season": ((day.dt.month%12)//3).astype("int8"),
            "tod":np.select([clock.lt(9*60),clock.lt(14*60),clock.lt(18*60)],[0,1,2],default=3).astype("int8"),
            "dow":day.dt.dayofweek.astype("int8"),
            "dep_sin":np.sin(clock*2*np.pi/1440).astype("float32"),
            "dep_cos":np.cos(clock*2*np.pi/1440).astype("float32"),
            "arr_sin":np.sin(arr_clock*2*np.pi/1440).astype("float32"),
            "arr_cos":np.cos(arr_clock*2*np.pi/1440).astype("float32"),
            "distance":distance.astype("float32"),"elapsed":elapsed.astype("float32"),
            "holiday":day.isin(holiday).astype("int8"),
            "peak":np.logical_or.reduce([day.isin(holiday+pd.Timedelta(days=n)) for n in (-2,-1,0,1,2)]).astype("int8"),
            "dep_utc":dep,"arr_utc":arr,"cutoff":cutoff,"dep_status":dep_status,"arr_status":arr_status,
            "y":state,"d60":confirmed,"delay":pd.to_numeric(raw.ArrDelay,errors="coerce").astype("float32"),
        }))
    result=pd.concat(frames,ignore_index=True)
    for field in ["origin","dest","carrier","month","dep_status","arr_status"]:
        result[field]=result[field].astype("category")
    return add_load(result)


def state_columns(frame):
    return pd.DataFrame(np.column_stack([frame.y.eq(i) for i in range(5)]),index=frame.index,
                        columns=list(CLASSES),dtype=np.int8)


def history_tables(development, releases):
    good=development.loc[development.y.ge(0),["month","y","origin","dest","carrier","season","tod"]]
    records=good.join(state_columns(good))
    total=records.groupby("month",observed=True)[list(CLASSES)].sum()
    tables=[records.groupby(["month",*keys],observed=True)[list(CLASSES)].sum() for keys in HISTKEYS]
    order=sorted(total.index,key=lambda month:releases.loc[month,"eligible_sec"])
    return total,tables,order


def historical(frame, tables, releases, last_month=None):
    total,matrix,order=tables
    # Hold the year-start feature history fixed for the test version.
    order=[month for month in order if last_month is None or month<=last_month]
    dates=releases.loc[order,"eligible_sec"].to_numpy()
    cut=frame.cutoff.to_numpy()
    group=np.searchsorted(dates,cut,side="right")
    history=np.full((len(frame),len(HIST)),np.nan,dtype=np.float32)
    baseline=np.full((len(frame),5),.2,dtype=np.float64)
    for n in np.unique(group):
        if not n:
            continue
        rows=np.flatnonzero(group==n)
        months=order[:n]
        prior=(total.loc[months].sum().to_numpy(dtype=float)+.5)
        prior/=prior.sum()
        for i,(keys,table) in enumerate(zip(HISTKEYS,matrix)):
            exposed=table.loc[table.index.get_level_values("month").isin(months)]
            summed=exposed.groupby(level=keys,observed=True).sum()
            index=pd.MultiIndex.from_frame(frame.iloc[rows][keys])
            counts=summed.reindex(index,fill_value=0).to_numpy(dtype=float)
            p=(counts+150*prior)/(counts.sum(axis=1,keepdims=True)+150)
            history[rows,3*i:3*i+3]=np.c_[p[:,0],p[:,2],p[:,:3].sum(axis=1)]
            if i==0:
                baseline[rows]=p
    for i,field in enumerate(HIST):
        frame[field]=history[:,i]
    frame["history_available"]=(group>0).astype("int8")
    return frame,baseline


def mos_station(station, years, root):
    paths=[root/"data/raw/iem/mos"/f"{station}_{year}.csv" for year in years]
    if station in ("KATL","KORD"):
        paths+=list((root/"data/raw/iem").glob(f"gfs_mos_{station}_*.csv"))
    paths=[p for p in paths if p.exists() and p.stat().st_size>500]
    if not paths:
        return None
    columns=["runtime","ftime",*WX_FIELDS]
    rows=pd.concat([pd.read_csv(path,usecols=lambda key:key in columns) for path in paths],ignore_index=True)
    rows["run"]=seconds(pd.to_datetime(rows.runtime,utc=True,errors="coerce"))
    rows["valid"]=seconds(pd.to_datetime(rows.ftime,utc=True,errors="coerce"))
    rows=rows.loc[rows.run.ne(INVALID_TIME)&rows.valid.ne(INVALID_TIME)].drop_duplicates(["run","valid"])
    for field in WX_FIELDS:
        rows[field]=pd.to_numeric(rows[field],errors="coerce").astype("float32") if field in rows else np.nan
    return rows[["run","valid",*WX_FIELDS]]


def match_mos(target,cutoff,table,buffer_hours):
    """Match an eligible run, preferring the closest valid tick within 90 minutes."""
    result=np.full((len(target),len(WX_FIELDS)+2),np.nan,dtype=np.float32)
    result[:,-1]=0
    if table is None or table.empty:
        return result
    valid_rows=np.flatnonzero((target!=INVALID_TIME)&(cutoff!=INVALID_TIME))
    if not len(valid_rows):
        return result
    # Floor the availability clock to the second. Take the forecast nearest the target.
    available=cutoff[valid_rows]-buffer_hours*3600
    times=target[valid_rows]
    slots_all=((times+5400)//10800)*10800
    chosen=np.full(len(valid_rows),INVALID_TIME,dtype=np.int64)
    grid=table.sort_values(["valid","run"])
    ticks=grid.valid.to_numpy()
    runs=grid.run.to_numpy()
    # Encode the (valid tick, run) ordering in one sortable integer for vectorized as-of lookup.
    stride=10_000_000_000
    lookup=ticks//10800*stride+runs
    positions=np.searchsorted(lookup,slots_all//10800*stride+available,side="right")-1
    closest=np.maximum(positions,0)
    keep=(positions>=0)&(ticks[closest]==slots_all)
    chosen[keep]=runs[closest[keep]]
    result[valid_rows[keep],:len(WX_FIELDS)]=grid.iloc[closest[keep]][list(WX_FIELDS)].to_numpy(dtype=np.float32)
    result[valid_rows[keep],-2]=(times[keep]-chosen[keep])/3600
    result[valid_rows[keep],-1]=1
    # Six-hour products describe a window ending at the next six-hour boundary.
    rain=["p06","q06","t06_2"]
    matched=chosen!=INVALID_TIME
    if np.any(matched):
        endpoints=((times[matched]+21599)//21600)*21600
        index=pd.MultiIndex.from_arrays([chosen[matched],endpoints],names=["run","valid"])
        values=table.set_index(["run","valid"])[rain].reindex(index).to_numpy()
        result[np.ix_(valid_rows[matched],[WX_FIELDS.index(x) for x in rain])]=values
    return result


def weather(frame, metadata, buffer_hours=5, root=ROOT):
    year=frame.date.dt.year
    years=range(int(year.min())-1,int(year.max())+2)
    values=np.full((len(frame),len(WX)),np.nan,dtype=np.float32)
    width=len(WX)//2
    values[:,[width-1,2*width-1]]=0
    cut=frame.cutoff.to_numpy()
    origin=frame.groupby("origin",observed=True).indices
    dest=frame.groupby("dest",observed=True).indices
    for airport in sorted(set(origin)|set(dest)):
        place=metadata.get(airport)
        if not place:
            continue
        table=mos_station(place[0],years,root)
        if airport in origin:
            idx=origin[airport]
            values[idx,:width]=match_mos(frame.dep_utc.to_numpy()[idx],cut[idx],table,buffer_hours)
        if airport in dest:
            idx=dest[airport]
            values[idx,width:]=match_mos(frame.arr_utc.to_numpy()[idx],cut[idx],table,buffer_hours)
    return frame.join(pd.DataFrame(values,index=frame.index,columns=WX))
