"""Release the fly into the world and monitor it."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flykenstein.world.loop import Release
from flykenstein.world.rules import WorldRules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", type=str, default=None, help="path to a rules JSON")
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    rules = WorldRules.load(a.rules) if a.rules else WorldRules()
    rules.duration_s = a.seconds
    rules.seed = a.seed

    out = Path(a.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rules.save(out / "rules.json")

    print(f"world  : {len(rules.food_sites)} food, {len(rules.hazard_sites)} hazard, "
          f"r={rules.arena_radius_mm}mm")
    rel = Release(rules=rules, vision=a.vision, render=not a.no_video)
    print(f"brain  : {rel.brain.n:,} neurons, {rel.brain.W.nnz:,} edges")
    print(f"body   : {len(rel.fly.actuated_joints)} actuated joints")
    print(f"release: {rules.duration_s}s at {rules.control_ms}ms control tick\n")

    df = rel.run()
    df.to_csv(out / "telemetry.csv", index=False)

    dist = float(np.hypot(np.diff(df.x), np.diff(df.y)).sum())
    print(f"\nended  : {rel.ended} after {df.t_s.iloc[-1]:.2f}s "
          f"({rel.wall_s:.0f}s wall, {rel.wall_s/df.t_s.iloc[-1]:.0f}x slower than life)")
    print(f"path   : {dist:.2f} mm travelled, final ({df.x.iloc[-1]:.2f}, {df.y.iloc[-1]:.2f})")
    print(f"energy : {df.energy.iloc[-1]:.3f}   feeds={rel.feeds}  hazard={rel.hazard_hits}")
    print(f"brain  : {int(df.spikes.sum()):,} spikes, {int(df.active.max())} distinct units active")
    print(f"        KC {df.kc.sum():.0f} | MBON {df.mbon.sum():.0f} | CX {df.cx.sum():.0f}")

    if not a.no_video and rel.save_video(out / "release.mp4"):
        print(f"video  : {out/'release.mp4'}")
    print(f"logged : {out/'telemetry.csv'}")


if __name__ == "__main__":
    main()
