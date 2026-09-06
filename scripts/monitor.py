"""Turn a release's telemetry into something you can look at."""
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flykenstein.world.rules import WorldRules


def plot(run_dir: Path) -> Path:
    df = pd.read_csv(run_dir / "telemetry.csv")
    rules = WorldRules.load(run_dir / "rules.json")

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    # --- the path through the world ---
    ax = fig.add_subplot(gs[:2, 0])
    circ = plt.Circle((0, 0), rules.arena_radius_mm, fill=False, ls="--", color="0.7")
    ax.add_patch(circ)
    for x, y in rules.food_sites:
        ax.add_patch(plt.Circle((x, y), rules.site_radius_mm, color="#2ecc71", alpha=.35))
        ax.plot(x, y, "o", color="#27ae60", ms=7)
    for x, y in rules.hazard_sites:
        ax.add_patch(plt.Circle((x, y), rules.site_radius_mm, color="#e74c3c", alpha=.35))
        ax.plot(x, y, "X", color="#c0392b", ms=9)
    pts = ax.scatter(df.x, df.y, c=df.t_s, cmap="viridis", s=6)
    ax.plot(df.x.iloc[0], df.y.iloc[0], "k^", ms=9, label="release")
    ax.plot(df.x.iloc[-1], df.y.iloc[-1], "ks", ms=9, label="end")
    fig.colorbar(pts, ax=ax, label="time (s)", shrink=.8)
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("path"); ax.legend(loc="upper right", fontsize=8)

    def panel(slot, cols, title, ylabel, colors=None):
        a = fig.add_subplot(gs[slot])
        for i, c in enumerate(cols):
            if c in df:
                a.plot(df.t_s, df[c], lw=1.2, label=c,
                       color=None if colors is None else colors[i])
        for _, seg in df.groupby((df.on_food != df.on_food.shift()).cumsum()):
            if seg.on_food.iloc[0]:
                a.axvspan(seg.t_s.iloc[0], seg.t_s.iloc[-1], color="#2ecc71", alpha=.15)
        for _, seg in df.groupby((df.on_hazard != df.on_hazard.shift()).cumsum()):
            if seg.on_hazard.iloc[0]:
                a.axvspan(seg.t_s.iloc[0], seg.t_s.iloc[-1], color="#e74c3c", alpha=.15)
        a.set_title(title, fontsize=10); a.set_ylabel(ylabel); a.set_xlabel("t (s)")
        a.legend(fontsize=7, ncol=2)
        return a

    panel((0, 1), ["spikes"], "whole-brain spikes per 5 ms tick", "spikes")
    panel((0, 2), ["in_odor_good_l", "in_odor_good_r", "in_sugar", "in_touch"],
          "sensory drive reaching the connectome", "0-1")
    panel((1, 1), ["dn_left", "dn_right", "dn_stop", "dn_back"],
          "descending neurons (DNa01/02, DNp09, MDN)", "Hz")
    panel((1, 2), ["mn9"], "MN9 - proboscis extension", "Hz")
    panel((2, 0), ["drive_l", "drive_r"], "walking drive sent to the body", "amplitude")
    panel((2, 1), ["kc", "mbon", "cx"], "mushroom body and central complex", "spikes/tick")
    panel((2, 2), ["energy"], "energy under our rules", "0-1")

    fig.suptitle(f"Flykenstein release - {run_dir.name}", fontsize=13)
    out = run_dir / "monitor.png"
    fig.savefig(out, dpi=130)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    a = ap.parse_args()
    d = Path(a.run_dir) if a.run_dir else sorted(Path("runs").iterdir())[-1]
    print(plot(d))
