"""Render publication figures from audited Q3 outcome summaries."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE,RED,ORANGE,PALE="#4573B4","#D73221","#FCB777","#D2EDF2"


def save(fig,directory,name):
    fig.savefig(directory/(name+".pdf"),bbox_inches="tight",facecolor="white")
    fig.savefig(directory/(name+".png"),dpi=190,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def render_evaluation(period,directory,year,main):
    directory.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(period/"calibration.csv")
    scores=pd.read_csv(period/"metrics.csv")
    contrasts=pd.read_csv(period/"paired_brier_intervals.csv")
    with plt.rc_context({"font.size":9,"pdf.fonttype":42}):
        fig,axes=plt.subplots(1,4,figsize=(12.1,3.1),layout="constrained")
        color={"A":BLUE,"B":RED,"B0":ORANGE}
        for ax,event,title in zip(axes,["D60","C","L60"],["Severe disruption","Cancellation","Arrival ≥60 min late"]):
            panel=data[data.event.eq(event)]
            edge=100*panel[["observed","predicted"]].max().max()*1.05
            ax.plot([0,edge],[0,edge],color=".65",lw=.8)
            for name in panel.model.unique():
                if name not in color:
                    continue
                line=panel[panel.model.eq(name)]
                ax.plot(100*line.predicted,100*line.observed,"o-",lw=1.3,ms=3,color=color[name],
                        label={"A":"Schedule + history","B":"With forecast","B0":"Lagged rates"}[name])
            ax.set(xlim=(0,edge),ylim=(0,edge),xlabel="Predicted (%)",ylabel="Observed (%)",title=title)
        axes[0].legend(loc="upper left",frameon=False,fontsize=7)
        parts=[]
        for name,rows in data[data.event.eq("D60")].groupby("model"):
            if name not in color:
                continue
            mean=rows.events.sum()/rows.n.sum()
            rel=((rows.predicted-rows.observed)**2*rows.n).sum()/rows.n.sum()
            res=((rows.observed-mean)**2*rows.n).sum()/rows.n.sum()
            unc=mean*(1-mean)
            brier=scores.loc[scores.model.eq(name)&scores.event.eq("D60")&scores.scope.eq("all"),"brier"].iat[0]
            parts.append(dict(model=name,reliability=rel,resolution=res,uncertainty=unc,
                              within_bin_residual=brier-rel+res-unc))
        decomposition=pd.DataFrame(parts)
        decomposition.to_csv(period/"brier_decomposition.csv",index=False,encoding="utf-8-sig")
        ax=axes[3]
        for index,record in decomposition.iterrows():
            ax.bar([index-.18,index+.18],1000*record[["reliability","resolution"]].to_numpy(dtype=float),
                   width=.32,color=color[record.model],edgecolor=".4",lw=.4)
        ax.set_xticks(range(len(decomposition)),decomposition.model)
        ax.set(title="Binned Brier components",ylabel="Component × 1,000")
        ax.text(.02,.98,"Left: reliability\nRight: resolution",transform=ax.transAxes,va="top",fontsize=7)
        save(fig,directory,f"q3_calibration_{year}")
        daily=pd.read_csv(period/"daily.csv")
        groups=pd.read_csv(period/"failure_groups.csv")
        matrix=groups[groups.slice.eq("origin*season")].pivot(index="origin",columns="season",values="residual")
        fig,(ax,bx)=plt.subplots(1,2,figsize=(8.6,3.35),layout="constrained")
        limit=100*matrix.abs().max().max()
        pic=ax.imshow(100*matrix,cmap=matplotlib.colors.LinearSegmentedColormap.from_list("q3",[BLUE,"white",RED]),
                      vmin=-limit,vmax=limit,aspect="auto")
        ax.set_yticks(range(len(matrix)),matrix.index)
        ax.set_xticks(range(len(matrix.columns)),[("Winter","Spring","Summer","Fall")[int(v)] for v in matrix.columns])
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                ax.text(j,i,f"{matrix.iloc[i,j]*100:+.1f}",ha="center",va="center",fontsize=8)
        fig.colorbar(pic,ax=ax,label="Observed − predicted (pp)")
        tail=daily.empirical_top5
        bx.scatter(100*daily.loc[~tail,"predicted"],100*daily.loc[~tail,"observed"],s=9,c=BLUE,alpha=.5)
        bx.scatter(100*daily.loc[tail,"predicted"],100*daily.loc[tail,"observed"],s=14,c=RED,label="Highest observed 5%")
        edge=100*daily[["observed","predicted"]].max().max()
        bx.plot([0,edge],[0,edge],c=".6",lw=.8)
        bx.set(xlabel="Daily predicted disruption (%)",ylabel="Daily observed disruption (%)")
        bx.legend(frameon=False,fontsize=8)
        fig.suptitle(f"Frozen Info-{main} · {year}",fontsize=10)
        save(fig,directory,f"q3_failure_diagnostics_{year}")
        weather_contrasts=contrasts[contrasts["first"].eq("B")&contrasts["second"].eq("A")]
        if not weather_contrasts.empty:
            fig,(ax,bx)=plt.subplots(1,2,figsize=(8.1,3.2),layout="constrained")
            events=["D60","C","L60"]
            for scope,delta,tint,label in (("all",-.12,BLUE,"All flights"),
                                           ("both_covered",.12,RED,"Both ends covered")):
                paired=weather_contrasts[weather_contrasts.scope.eq(scope)].set_index("event").loc[events]
                y=[i+delta for i in range(len(events))]
                ax.hlines(y,1000*paired.lo,1000*paired.hi,color=tint,lw=1)
                ax.scatter(1000*paired.brier_difference,y,c=tint,s=17,label=label)
                wide=scores[scores.scope.eq(scope)&scores.model.isin(["A","B"])].pivot(index="event",columns="model",values="ap")
                bx.scatter(100*(wide.loc[events,"B"]-wide.loc[events,"A"]),y,c=tint,s=17)
            for axis,label in ((ax,"Brier difference × 1,000"),(bx,"Average precision difference (pp)")):
                axis.axvline(0,color=".55",lw=.8)
                axis.set_yticks(range(len(events)),events)
                axis.invert_yaxis()
                axis.set_xlabel(label)
            ax.legend(frameon=False,fontsize=8,loc="best")
            save(fig,directory,f"q3_information_gain_{year}")
