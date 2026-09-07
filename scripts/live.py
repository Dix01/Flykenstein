"""Release the same fly repeatedly. It keeps what it learned.

Each release loads the mushroom body weights and the drive state the previous
one left behind, lives, and writes them back. The synapses are the continuity:
nothing else carries over.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flykenstein.world.loop import Release
from flykenstein.world.rules import WorldRules
from flykenstein.world.state import Life


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--releases", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--life", type=str, default="life")
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="start from a blank brain")
    a = ap.parse_args()

    root = Path(a.out) / time.strftime("life-%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    if a.fresh:
        import shutil
        shutil.rmtree(a.life, ignore_errors=True)

    history = []
    for k in range(a.releases):
        rules = WorldRules()
        rules.duration_s = a.seconds
        rules.seed = k
        rules.life_path = a.life
        out = root / f"release-{k+1:02d}"
        out.mkdir(exist_ok=True)
        rules.save(out / "rules.json")

        rel = Release(rules=rules, vision=a.vision, render=not a.no_video)
        before = rel.mb.depression()
        print(f"\n=== release {k+1}/{a.releases}   "
              f"carried memory: {before*100:.2f}% depressed"
              f"{'  (fresh brain)' if not rel.carried else ''}")
        df = rel.run(verbose=False)
        after = rel.mb.depression()
        df.to_csv(out / "telemetry.csv", index=False)
        if not a.no_video:
            rel.save_video(out / "release.mp4")

        dist = float(np.hypot(np.diff(df.x), np.diff(df.y)).sum())
        on_food = int(df.on_food.sum())
        print(f"    ended {rel.ended} at {df.t_s.iloc[-1]:.1f}s | {dist:.0f} mm | "
              f"{on_food} ticks on food | energy {rel.drives.energy:.2f} | "
              f"memory {before*100:.2f}% -> {after*100:.2f}%")
        history.append({"release": k + 1, "before": before, "after": after,
                        "on_food": on_food, "dist": dist,
                        "energy": rel.drives.energy, "ended": rel.ended})

    life = Life(a.life)
    life.load()
    print("\n" + "=" * 66)
    print("life so far:", life.summary())
    print("\nmemory across releases (fraction of KC->MBON weight lost):")
    for h in history:
        bar = "#" * int(h["after"] * 400)
        print(f"  release {h['release']}: {h['before']*100:5.2f}% -> "
              f"{h['after']*100:5.2f}%  {bar}")
    import pandas as pd
    pd.DataFrame(history).to_csv(root / "history.csv", index=False)
    print(f"\nwritten: {root}")


if __name__ == "__main__":
    main()
