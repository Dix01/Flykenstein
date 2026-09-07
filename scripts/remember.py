"""Ask the fly what it remembers.

The answer is a set of weakened synapses, so that is what this prints: which
mushroom body compartments have lost drive, and by how much. The episode log
alongside it is our record of what happened, not the animal's.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flykenstein.brain.lif import Brain
from flykenstein.brain.plasticity import MushroomBody
from flykenstein.brain.ports import Ports
from flykenstein.world.state import Life


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--life", type=str, default="life")
    a = ap.parse_args()

    brain = Brain()
    ports = Ports(brain)
    mb = MushroomBody(brain, ports)
    life = Life(a.life)
    if not life.load(mb):
        print(f"no life at {a.life} - nothing has happened to it yet")
        return

    print(f"life: {life.summary()}\n")
    total = mb.depression()
    print(f"mushroom body: {total*100:.2f}% of KC->MBON weight lost to experience")

    comp = mb.per_compartment()
    order = np.argsort(comp)[::-1]
    names = brain.meta["primary_type"].iloc[mb.mbon].astype(str).to_numpy()
    print("\nmost changed compartments:")
    for i in order[:12]:
        if comp[i] <= 0:
            break
        print(f"  {names[i]:14s} {comp[i]*100:5.1f}%  {'#' * int(comp[i]*60)}")
    touched = int((comp > 0.01).sum())
    print(f"\n{touched} of {len(comp)} compartments carry a trace above 1%.")

    w = brain.W.data[mb.ptr]
    lost = (mb.w0 - w) / np.maximum(mb.w0, 1e-9)
    print(f"{int((lost > 0.1).sum()):,} of {len(lost):,} individual synapses "
          f"are more than 10% weaker than they started.")

    if life.episodes:
        print("\nepisode log (our record, not its memory):")
        for e in life.episodes[-12:]:
            print(f"  release {e['release']}  t={e.get('t',0):6.2f}s  {e['kind']:8s} "
                  f"at ({e.get('x',0):6.2f}, {e.get('y',0):6.2f})")


if __name__ == "__main__":
    main()
