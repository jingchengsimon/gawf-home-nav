"""Headless saving and common axes formatting for analysis figures."""
from pathlib import Path


def finish(fig, output):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in ("png", "svg"):
            fig.savefig(output.with_suffix("." + suffix), dpi=200, facecolor="white", bbox_inches="tight")
    finally:
        plt.close(fig)

def clean(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(False)



def plotting():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 12, "axes.titlesize": 13})
    return np, plt
