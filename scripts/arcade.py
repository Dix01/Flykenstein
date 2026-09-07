"""Run a release as a watchable thing: live HUD while it computes, real-time
video and a scored telemetry track to play back afterwards.

The video is rendered at play_speed 1.0, so playback runs at the speed the fly
actually lived it. The computation behind it is about a hundred times slower
than that, which is why this writes a file instead of opening a window.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flykenstein.world.loop import Release
from flykenstein.world.rules import WorldRules

BAR = "█"


def bar(v, lo, hi, width=12):
    f = 0.0 if hi <= lo else np.clip((v - lo) / (hi - lo), 0, 1)
    n = int(round(f * width))
    return BAR * n + "·" * (width - n)


def minimap(x, y, rules, w=31, h=13):
    R = rules.arena_radius_mm
    grid = [[" "] * w for _ in range(h)]

    def cell(px, py):
        cx = int(round((px / R * 0.95 + 1) / 2 * (w - 1)))
        cy = int(round((1 - (py / R * 0.95 + 1) / 2) * (h - 1)))
        return np.clip(cx, 0, w - 1), np.clip(cy, 0, h - 1)

    for fx, fy in rules.food_sites:
        cx, cy = cell(fx, fy); grid[cy][cx] = "o"
    for hx, hy in rules.hazard_sites:
        cx, cy = cell(hx, hy); grid[cy][cx] = "x"
    cx, cy = cell(x, y); grid[cy][cx] = "@"
    return ["".join(r) for r in grid]


ACHIEVEMENTS = [
    ("first_taste", "First Taste", "extended the proboscis on food"),
    ("forager", "Forager", "fed at two different sites"),
    ("it_remembers", "It Remembers", "formed more than 1% memory"),
    ("survivor", "Survivor", "ended with more energy than it started"),
    ("burned", "Burned", "walked into a hazard"),
    ("wanderer", "Wanderer", "covered more than 100 mm"),
]


def score_row(prev, row, rules):
    s = 0.0
    if row["on_food"] and row["mn9"] > 5.0:
        s += 100.0
    if row["on_hazard"]:
        s -= 200.0
    s += 50.0 * (rules.control_ms / 1000.0)          # staying alive
    s += 4000.0 * max(0.0, row["depressed"] - prev)  # learning something
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/arcade")
    ap.add_argument("--life", type=str, default="life")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-food", action="store_true",
                    help="control world: no food anywhere, memory should stay flat")
    ap.add_argument("--hud-hz", type=float, default=4.0)
    a = ap.parse_args()

    rules = WorldRules()
    rules.duration_s = a.seconds
    rules.seed = a.seed
    rules.life_path = a.life
    if a.no_food:
        rules.food_sites = ()
    if a.fresh:
        import shutil
        shutil.rmtree(a.life, ignore_errors=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rules.save(out / "rules.json")

    rel = Release(rules=rules, vision=False, render=not a.no_food)
    if rel.camera is not None:
        rel.camera.play_speed = 1.0     # play back at the speed it was lived

    print(f"\n  FLYKENSTEIN  ·  {rel.brain.n:,} neurons · {rel.brain.W.nnz:,} synapses "
          f"· {len(rel.mb.ptr):,} of them able to change")
    print(f"  world: {len(rules.food_sites)} food · {len(rules.hazard_sites)} hazard "
          f"· carried memory {rel.mb.depression()*100:.2f}%\n")

    # Stream a HUD while it computes, by stepping the release one tick at a time.
    score = 0.0
    unlocked = {}
    sites_fed = set()
    every = max(1, int(round((1000.0 / rules.control_ms) / max(a.hud_hz, 1e-6))))
    t0 = time.time()
    prev_dep = rel.mb.depression()

    for tick, row in enumerate(rel.iter_run()):
        score += score_row(prev_dep, row, rules)
        prev_dep = row["depressed"]

        if row["on_food"] and row["mn9"] > 5.0:
            sites_fed.add((round(row["x"] / 6), round(row["y"] / 6)))
            unlocked.setdefault("first_taste", row["t_s"])
        if len(sites_fed) >= 2:
            unlocked.setdefault("forager", row["t_s"])
        if row["depressed"] > 0.01:
            unlocked.setdefault("it_remembers", row["t_s"])
        if row["on_hazard"]:
            unlocked.setdefault("burned", row["t_s"])

        if tick % every == 0:
            mm = minimap(row["x"], row["y"], rules)
            print(f"  t {row['t_s']:5.2f}s   score {int(score):7d}   "
                  f"E [{bar(row['energy'],0,1.2)}] {row['energy']:.2f}   "
                  f"hunger {row['hunger']:.2f}")
            print(f"  {mm[len(mm)//2]}   memory [{bar(row['depressed'],0,0.06)}] "
                  f"{row['depressed']*100:5.2f}%   KC {int(row['kc_active']):3d}   "
                  f"brain {int(row['spikes']):5d} spikes"
                  + ("   ★ FEEDING" if row["on_food"] and row["mn9"] > 5 else "")
                  + ("   ☠ HAZARD" if row["on_hazard"] else ""))

    df = rel.telemetry.frame()
    if not df.empty:
        if df.energy.iloc[-1] > rules.energy_start:
            unlocked.setdefault("survivor", df.t_s.iloc[-1])
        dist = float(np.hypot(np.diff(df.x), np.diff(df.y)).sum())
        if dist > 100:
            unlocked.setdefault("wanderer", df.t_s.iloc[-1])
    else:
        dist = 0.0

    df.to_csv(out / "telemetry.csv", index=False)
    if rel.camera is not None:
        rel.save_video(out / "release.mp4")

    meta = {"score": int(score), "seed": a.seed, "ended": rel.ended, "distance_mm": dist,
            "seconds": float(df.t_s.iloc[-1]) if not df.empty else 0.0,
            "wall_s": time.time() - t0,
            "memory_start": float(prev_dep) if df.empty else float(df.depressed.iloc[0]),
            "memory_end": float(prev_dep), "feeds": rel.feeds,
            "hazard_ticks": rel.hazard_hits, "no_food": a.no_food,
            "food_sites": [list(p) for p in rules.food_sites],
            "hazard_sites": [list(p) for p in rules.hazard_sites],
            "arena_radius_mm": rules.arena_radius_mm,
            "neurons": int(rel.brain.n), "synapses": int(rel.brain.W.nnz),
            "plastic": int(len(rel.mb.ptr)),
            "achievements": [{"id": k, "name": n, "how": h,
                              "at": unlocked.get(k)} for k, n, h in ACHIEVEMENTS]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  ── run over: {rel.ended} ──")
    print(f"  SCORE {int(score):,}   {dist:.0f} mm   memory {meta['memory_end']*100:.2f}%")
    for k, n, h in ACHIEVEMENTS:
        mark = f"unlocked at {unlocked[k]:.2f}s" if k in unlocked else "locked"
        print(f"    {'★' if k in unlocked else '·'} {n:14s} {mark}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
