"""Probability models, coherent calibration and scoring for Q3."""

import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax, logit
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from q3_inputs import CAT, CALENDAR, LOAD, HIST, WX

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

EPS=1e-10
MODELS=("history","logistic","binary_tree","joint_tree","chain_tree")
EVENTS={"D60":(0,1,2),"C":(0,),"L60":(2,),"V":(1,),"M60":(3,),"R1":(4,)}
THREADS=5


class Forecast:
    def __init__(self, family, info, load=True):
        self.family=family
        self.info=info
        numeric=CALENDAR+HIST+["history_available"]
        if load:
            numeric+=LOAD
        if info=="B":
            numeric+=WX
        self.fields=CAT+numeric
        if family=="logistic":
            transform=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                                    StandardScaler())
            self.encoder=ColumnTransformer([
                ("categorical",OneHotEncoder(handle_unknown="ignore",dtype=np.float32),CAT),
                ("numeric",transform,numeric),
            ],sparse_threshold=1)
        else:
            self.encoder=ColumnTransformer([
                ("categorical",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=np.nan,
                                              dtype=np.float32),CAT),
                ("numeric","passthrough",numeric),
            ])
        self.engine="sklearn_logistic" if family=="logistic" else (
            "lightgbm" if LGBMClassifier is not None else "sklearn_hist_gradient_boosting")
        self.convergence_warnings=0
        self.n_training=0
        self.diversion_level=""

    def train_tree(self,x,y,seed=2026,leaves=23):
        if np.unique(y).size<2:
            return None
        if LGBMClassifier is not None:
            model=LGBMClassifier(n_estimators=115,num_leaves=leaves,min_child_samples=110,
                                 learning_rate=.075,reg_lambda=1.5,colsample_bytree=.9,
                                 verbosity=-1,n_jobs=THREADS,random_state=seed)
            model.fit(x,y,categorical_feature=list(range(len(CAT))))
        else:
            # Ordinal levels are trained only on the fit subset. Large categories are numeric ordered features.
            cardinality=[len(items) for items in self.encoder.named_transformers_["categorical"].categories_]
            categorical=[i for i,n in enumerate(cardinality) if n<=255]
            model=HistGradientBoostingClassifier(max_iter=75,max_leaf_nodes=leaves,min_samples_leaf=110,
                                                learning_rate=.10,l2_regularization=1.5,
                                                categorical_features=categorical,early_stopping=False,
                                                random_state=seed)
            model.fit(x,y)
        return model

    def fit(self,frame):
        y=frame.y.to_numpy(dtype=int)
        self.n_training=len(y)
        if len(y)==0 or np.any(y<0) or len(np.unique(y))<5:
            raise ValueError("Training subset needs the five identifiable flight outcomes")
        if self.engine=="sklearn_hist_gradient_boosting" and len(y)>420000:
            # Cap the fallback engine predictably while keeping unweighted, representative labels.
            sample=np.sort(np.random.default_rng(2026).choice(len(y),420000,replace=False))
            frame=frame.iloc[sample]
            y=y[sample]
            self.n_training=len(y)
        with threadpool_limits(limits=THREADS):
            x=self.encoder.fit_transform(frame[self.fields])
            if self.family=="logistic":
                self.model=LogisticRegression(max_iter=450,C=.35,solver="lbfgs",random_state=2026)
                with warnings.catch_warnings(record=True) as log:
                    warnings.simplefilter("always",ConvergenceWarning)
                    self.model.fit(x,y)
                self.convergence_warnings=sum(issubclass(w.category,ConvergenceWarning) for w in log)
            elif self.family=="joint_tree":
                self.model=self.train_tree(x,y)
            elif self.family=="binary_tree":
                self.model=self.train_tree(x,(y<=2).astype(int))
            elif self.family=="chain_tree":
                noncancel=y!=0
                arrived=y>=2
                self.cancel=self.train_tree(x,(y==0).astype(int),leaves=16)
                self.final=self.train_tree(x[arrived],y[arrived]-2)
                self.diversion_prior=(np.sum(y==1)+.5)/(noncancel.sum()+1)
                self.diversion=None
                self.diversion_level="prior"
                # Compare the rare branch with a smoothed historical frequency using a later published slice.
                eligible=np.flatnonzero(noncancel)
                ordered=eligible[np.argsort(frame.date.to_numpy()[eligible],kind="stable")]
                audit_date=pd.Timestamp(frame.date.iloc[ordered[int(len(ordered)*.80)]])
                earlier=ordered[frame.date.to_numpy()[ordered]<audit_date.to_datetime64()]
                later=ordered[frame.date.to_numpy()[ordered]>=audit_date.to_datetime64()]
                if len(later):
                    earlier=earlier[frame.label_release.to_numpy()[earlier]<=frame.cutoff.to_numpy()[later].min()]
                if len(earlier) and np.sum(y[later]==1)>=25:
                    branch_y=(y[earlier]==1).astype(int)
                    event=(y[later]==1).astype(int)
                    prior=(branch_y.sum()+.5)/(len(branch_y)+1)
                    best=np.mean((event-prior)**2)
                    branch=self.train_tree(x[earlier],branch_y,leaves=8)
                    loss=np.mean((branch.predict_proba(x[later])[:,1]-event)**2) if branch is not None else np.inf
                    if loss<best:
                        self.diversion=self.train_tree(x[noncancel],(y[noncancel]==1).astype(int),leaves=8)
                        self.diversion_level="tree"
                    else:
                        branch=LogisticRegression(C=.04,max_iter=120)
                        numeric=np.nan_to_num(x, nan=-1,copy=True)
                        scale=StandardScaler().fit(numeric[earlier])
                        branch.fit(scale.transform(numeric[earlier]),branch_y)
                        loss=np.mean((branch.predict_proba(scale.transform(numeric[later]))[:,1]-event)**2)
                        if loss<best:
                            self.diversion_scale=StandardScaler().fit(numeric[noncancel])
                            self.diversion=LogisticRegression(C=.04,max_iter=120).fit(
                                self.diversion_scale.transform(numeric[noncancel]),(y[noncancel]==1).astype(int))
                            self.diversion_level="logistic"
        return self

    def predict(self,frame):
        x=self.encoder.transform(frame[self.fields])
        with threadpool_limits(limits=THREADS):
            if self.family!="chain_tree":
                return self.model.predict_proba(x)
            c=self.cancel.predict_proba(x)[:,1]
            if self.diversion_level=="prior":
                v=np.full(len(c),self.diversion_prior)
            elif self.diversion_level=="logistic":
                v=self.diversion.predict_proba(self.diversion_scale.transform(np.nan_to_num(x,nan=-1)))[:,1]
            else:
                v=self.diversion.predict_proba(x)[:,1]
            journey=self.final.predict_proba(x)
        return np.column_stack([c,(1-c)*v,((1-c)*(1-v))[:,None]*journey])


def temperature(p,value):
    return softmax(np.log(np.clip(p,EPS,1))/value,axis=1)


def loss_log(y,p):
    return -np.log(np.clip(p[np.arange(len(y)),y],EPS,1)).mean()


def calibrate(y,p):
    return float(minimize_scalar(lambda t:loss_log(y,temperature(p,t)),bounds=(.4,3),
                                 method="bounded",options={"xatol":.0001}).x)


def score_event(y,p):
    y=np.asarray(y,dtype=np.int8)
    p=np.asarray(p)
    order=np.argsort(-p,kind="stable")
    top={}
    for frac in (.1,.2):
        limit=max(1,int(np.ceil(frac*len(y))))
        hit=y[order[:limit]].sum()
        label=f"top{int(frac*100)}"
        top[label+"_precision"]=hit/limit
        top[label+"_recall"]=hit/y.sum() if y.sum() else np.nan
        top[label+"_lift"]=hit/limit/y.mean() if y.sum() else np.nan
    pclip=np.clip(p,EPS,1-EPS)
    if 0<y.mean()<1:
        ap=average_precision_score(y,p)
        auc=roc_auc_score(y,p)
    else:
        ap=auc=np.nan
    bins=np.unique(np.quantile(p,np.linspace(0,1,11)))[1:-1]
    ids=np.searchsorted(bins,p,side="right")
    counts=np.bincount(ids)
    ece=np.abs(np.bincount(ids,weights=p-y,minlength=len(counts))).sum()/len(y)
    return {"n":len(y),"events":int(y.sum()),"observed":y.mean(),"predicted":p.mean(),
            "brier":np.mean((p-y)**2),"logloss":-np.mean(y*np.log(pclip)+(1-y)*np.log1p(-pclip)),
            "ap":ap,"auc":auc,"ece":ece,**top}


def reliability(y,p):
    edges=np.unique(np.quantile(p,np.linspace(0,1,11)))[1:-1]
    partition=np.searchsorted(edges,p,side="right")
    bins=pd.DataFrame({"event":y,"probability":p,"bin":partition}).groupby("bin")
    rows=bins.agg(n=("event","size"),events=("event","sum"),
                  observed=("event","mean"),predicted=("probability","mean")).reset_index()
    return rows


def intercept_slope(y,p):
    predictor=logit(np.clip(p,EPS,1-EPS)).reshape(-1,1)
    with threadpool_limits(limits=THREADS):
        fit=LogisticRegression(C=1e6,max_iter=100).fit(predictor,y)
    return fit.intercept_[0],fit.coef_[0,0]
