"""Render project-page scientific figures from frozen local source snapshots."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "project-page"
ASSETS = PAGE / "dist/assets"
COLORS = {"gru1":"#1f77b4", "gawf1":"#ff7f0e", "lstm2":"#1f77b4", "gawf2":"#ff7f0e"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.labelsize":12,"axes.titlesize":13})


def finish(fig, name):
    fig.savefig(ASSETS / (name+".png"), dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(ASSETS / (name+".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def clean(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(False)


def validation():
    source=PAGE/"sources/embclip_validation.json"
    d=json.loads(source.read_text())
    fig, axes=plt.subplots(1,2,figsize=(9.2,4.1))
    stats={}
    for ax,metric,title,ylabel in zip(axes,("common_sr","common_spl"),("Validation success","Validation path efficiency"),("SR (%)","SPL (%)")):
        for x,model in enumerate(("gru1","gawf1")):
            rows=[r for r in d["runs"] if r["model"]==model]
            vals=np.array([r[metric]*100 for r in rows])
            stats.setdefault(model,{})[metric]={"mean":float(vals.mean()),"seeds":[r["seed"] for r in rows],"values":vals.tolist()}
            sem=vals.std(ddof=1)/np.sqrt(len(vals))
            ax.bar(x,vals.mean(),width=.72,color=COLORS[model],yerr=sem,capsize=2.5,
                   error_kw={"elinewidth":1,"ecolor":"#333333"},zorder=2)
            jitter=np.random.default_rng(0).uniform(-.1584,.1584,len(vals))
            ax.scatter(x+jitter,vals,s=12,color="#333333",alpha=.58,zorder=3,linewidths=0)
        ax.set_xticks([0,1],["EmbCLIP","EmbCLIP-GaWF"],rotation=40,ha="right")
        ax.set_ylabel(ylabel);ax.set_title(title,pad=12);ax.set_xlim(-.55,1.55)
        ax.set_ylim(0,max(r[metric]*100 for r in d["runs"])+5);clean(ax)
    fig.suptitle("EmbCLIP · RoboTHOR · light 5M",fontsize=14,y=1.02)
    fig.tight_layout(w_pad=2.5)
    finish(fig,"embclip-validation")
    (PAGE/"sources/display_statistics.json").write_text(json.dumps(stats,indent=2)+"\n")


def training():
    d=json.loads((PAGE/"sources/embclip_training.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.9))
    start=max(r["points"][0]["step"] for r in d["runs"])
    end=min(r["points"][-1]["step"] for r in d["runs"])
    grid=np.arange(start,end+1,10240)
    for ax,metric,ylabel in zip(axes,("success","spl"),("Training SR (%)","Training SPL (%)")):
        for model in ("gru1","gawf1"):
            arr=[]
            for row in d["runs"]:
                if row["model"]!=model:continue
                points=row["points"]
                values=np.interp(grid,[p["step"] for p in points],[p[metric] for p in points])*100
                arr.append(np.convolve(np.pad(values,(5,5),mode="edge"),np.ones(11)/11,mode="valid"))
            arr=np.array(arr);mean=arr.mean(0);std=arr.std(0)
            ax.plot(grid/1e6,mean,color=COLORS[model],lw=2,label={"gru1":"EmbCLIP","gawf1":"EmbCLIP-GaWF"}[model])
            ax.fill_between(grid/1e6,mean-std,mean+std,color=COLORS[model],alpha=.14,linewidth=0)
        ax.set_xlabel("Global training frames (millions)");ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0);ax.set_xlim(0,5.01);clean(ax);ax.legend(frameon=False,fontsize=10)
    fig.suptitle("RoboTHOR training · mean ± population SD",fontsize=14)
    fig.tight_layout(w_pad=2)
    finish(fig,"embclip-training")


def pointnav():
    p=ROOT/"results/pointnav-rgb-seed0-validation-20261023/validation_scores.json"
    d=json.loads(p.read_text());shutil.copy2(p,PAGE/"sources/pointnav_validation.json")
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.1))
    for ax,metric,title,ylabel in zip(axes,("success","spl"),("Validation success","Validation path efficiency"),("SR (%)","SPL (%)")):
        for x,model in enumerate(("lstm2","gawf2")):
            v=d["models"][model]["summary"]["metrics"][metric]*100
            ax.bar(x,v,width=.72,color=COLORS[model]);ax.scatter([x],[v],s=12,color="#333333",zorder=3)
        ax.set_xticks([0,1],["DD-PPO","DD-PPO-GaWF"],rotation=40,ha="right")
        ax.set_ylabel(ylabel);ax.set_title(title,pad=12);ax.set_xlim(-.55,1.55);ax.set_ylim(0,.5 if metric=="success" else .04);clean(ax)
    fig.suptitle("RGB-only PointNav · Gibson v2 · light 1M",fontsize=14,y=1.02)
    fig.tight_layout(w_pad=2.5);finish(fig,"pointnav-validation")
    source=ROOT/"results/pointnav-rgb-seed0-validation-20261023/sources/training_scalars.json"
    curves=json.loads(source.read_text());shutil.copy2(source,PAGE/"sources/pointnav_training.json")
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.9))
    for ax,metric,ylabel in zip(axes,("success","spl"),("Training SR (%)","Training SPL (%)")):
        for row,color in zip(curves["runs"],("#1f77b4","#ff7f0e")):
            points=row["series"][metric]["points"]
            ax.plot([p["step"]/1e6 for p in points],[p["value"]*100 for p in points],color=color,lw=1.8,label={"LSTM2":"DD-PPO","GaWF2":"DD-PPO-GaWF"}.get(row["label"],row["label"]))
        ax.set_xlabel("Global training frames (millions)");ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);clean(ax);ax.legend(frameon=False)
    fig.suptitle("RGB-only PointNav training · completed training episodes",fontsize=14)
    fig.tight_layout(w_pad=2);finish(fig,"pointnav-training")


if __name__=="__main__":
    ASSETS.mkdir(parents=True,exist_ok=True)
    validation();training();pointnav()
