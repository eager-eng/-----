from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"code/问题三"))
from q3_inputs import CLASSES, WX_FIELDS, history_tables, historical, match_mos
from q3_models import calibrate, temperature
from q3_open_meteo import enrich


def test_mos_skips_forecast_that_is_too_recent():
    tick=1_704_067_200
    table=pd.DataFrame({"run":[tick-36*3600,tick-30*3600,tick-24*3600],"valid":[tick]*3,
                        **{k:[1.,2.,3.] for k in WX_FIELDS}})
    result=match_mos(np.array([tick]),np.array([tick-24*3600]),table,5)
    assert result[0,0]==2
    assert result[0,-2]==30
    assert result[0,-1]==1


def test_lag_history_disallows_unpublished_label():
    history=pd.DataFrame({"month":["2023-01","2023-02"],"y":[0,4],"origin":["ATL"]*2,
                          "dest":["ORD"]*2,"carrier":["DL"]*2,"season":[0]*2,"tod":[1]*2})
    series=pd.DataFrame({"eligible_sec":[100,200]},index=["2023-01","2023-02"])
    target=history.copy()
    target["cutoff"]=[99,100]
    example,p=historical(target,history_tables(history,series),series)
    assert example.history_available.tolist()==[0,1]
    np.testing.assert_allclose(p[0],.2)
    assert p[1,0]>p[1,4]
    np.testing.assert_allclose(p.sum(axis=1),1)


def test_one_calibration_mapping_preserves_exhaustive_label_space():
    p=np.array([[.10,.02,.10,.18,.60],[.20,.01,.20,.20,.39],
                [.08,.01,.40,.12,.39],[.06,.01,.10,.20,.63]])
    y=np.array([4,0,2,4])
    out=temperature(p,calibrate(y,p))
    assert out.shape==(len(p),len(CLASSES))
    assert (out>=0).all()
    np.testing.assert_allclose(out.sum(axis=1),1)
    np.testing.assert_allclose(out[:,:3].sum(axis=1)+out[:,3:].sum(axis=1),1)


def test_fixed_lead_forecast_checks_each_endpoint_clock(tmp_path):
    midnight=1_704_067_200
    base=tmp_path/"data/raw/open_meteo"
    base.mkdir(parents=True)
    pd.DataFrame({"target":[midnight],"temperature_c":[20]}).to_csv(base/"ATL_2024.csv.gz",index=False)
    pd.DataFrame({"target":[midnight+21*3600],"temperature_c":[20]}).to_csv(base/"SFO_2024.csv.gz",index=False)
    frame=pd.DataFrame({"origin":["ATL"],"dest":["SFO"],"date":pd.to_datetime(["2024-01-01"]),
                        "dep_utc":[midnight],"arr_utc":[midnight+21*3600],"cutoff":[midnight-24*3600]})
    merged=enrich(frame,buffer_hours=6,root=tmp_path).iloc[0]
    assert merged.depwx_avail==1
    assert merged.depwx_tmp==68
    assert merged.arrwx_avail==0
