"""Positive control: reproduce the sugar -> proboscis-extension result.

Shiu et al. 2024 activate the 21 right-hemisphere sugar-sensing gustatory
receptor neurons with 150 Hz Poisson input and read out MN9, the motor neuron
that drives proboscis extension. If this circuit lights up, the connectome is
loaded correctly and the integrator is doing real work.
"""
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flykenstein.brain.lif import Brain

SUGAR_R = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265


def rates(brain, spikes, ms):
    counts = np.zeros(brain.n)
    for s in spikes:
        counts[s] += 1
    return counts / (ms / 1000.0)


def main(n_trials=3, ms=1000.0, rate_hz=150.0):
    brain = Brain(seed=0)
    sugar = brain.index_of.reindex(SUGAR_R).dropna().astype(int).to_numpy()
    mn9 = int(brain.index_of[MN9])
    print(f"sugar GRNs found : {len(sugar)}/{len(SUGAR_R)}    MN9 index: {mn9}")

    for label, drive in (("sugar ON ", rate_hz), ("sugar OFF", 0.0)):
        all_rates = []
        for trial in range(n_trials):
            brain.reset()
            brain.rng = np.random.default_rng(trial)
            poi = np.zeros(brain.n, dtype=np.float32)
            poi[sugar] = drive
            brain.exempt[sugar] = True          # Poisson targets skip refractory
            t0 = time.time()
            spikes = brain.run(ms, poisson=poi)
            r = rates(brain, spikes, ms)
            all_rates.append(r)
            n_active = int((r > 0).sum())
            print(f"  {label} trial {trial}: MN9 {r[mn9]:6.1f} Hz | "
                  f"{n_active:6d} neurons active | {time.time()-t0:5.1f}s")
        r = np.mean(all_rates, axis=0)
        top = np.argsort(r)[::-1][:8]
        names = brain.meta.loc[top, "primary_type"].fillna("?").tolist()
        print(f"  {label} mean MN9 = {r[mn9]:.1f} Hz;  top units: "
              + ", ".join(f"{n}:{r[i]:.0f}Hz" for n, i in zip(names, top)))


if __name__ == "__main__":
    main()
