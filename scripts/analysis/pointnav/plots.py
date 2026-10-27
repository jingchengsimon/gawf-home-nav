"""Plot current PointNav training and validation from structured numeric snapshots."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COLORS = {"lstm2": "#1f77b4", "gawf2": "#ff7f0e"}

from scripts.analysis.plotting import finish, clean, plotting





def validation(source, output):
    _, plt = plotting()
    d=json.loads(Path(source).read_text())
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.1))
    for ax,metric,title,ylabel in zip(axes,("success","spl"),("Validation success","Validation path efficiency"),("SR (%)","SPL (%)")):
        for x,model in enumerate(("lstm2","gawf2")):
            v=d["models"][model]["summary"]["metrics"][metric]*100
            ax.bar(x,v,width=.72,color=COLORS[model]);ax.scatter([x],[v],s=12,color="#333333",zorder=3)
        ax.set_xticks([0,1],["DD-PPO","DD-PPO-GaWF"],rotation=40,ha="right")
        ax.set_ylabel(ylabel);ax.set_title(title,pad=12);ax.set_xlim(-.55,1.55);ax.set_ylim(0,.5 if metric=="success" else .04);clean(ax)
    fig.suptitle("RGB-only PointNav · Gibson v2 · light 1M",fontsize=14,y=1.02)
    fig.tight_layout(w_pad=2.5);finish(fig,output)


def training(source, output):
    _, plt = plotting()
    curves=json.loads(Path(source).read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.9))
    for ax,metric,ylabel in zip(axes,("success","spl"),("Training SR (%)","Training SPL (%)")):
        for row,color in zip(curves["runs"],("#1f77b4","#ff7f0e")):
            points=row["series"][metric]["points"]
            ax.plot([p["step"]/1e6 for p in points],[p["value"]*100 for p in points],color=color,lw=1.8,label={"LSTM2":"DD-PPO","GaWF2":"DD-PPO-GaWF"}.get(row["label"],row["label"]))
        ax.set_xlabel("Global training frames (millions)");ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);clean(ax);ax.legend(frameon=False)
    fig.suptitle("RGB-only PointNav training · completed training episodes",fontsize=14)
    fig.tight_layout(w_pad=2);finish(fig,output)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validation", "training"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.source is None or args.output is None:
        parser.error("--source and --output required")
    result = {"validation": validation, "training": training}[args.operation](args.source, args.output)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
