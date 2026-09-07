"""Does the mushroom body actually learn? A paired/unpaired conditioning assay.

The standard fly experiment: present an odour, pair it with reward, then present
it again and see whether the mushroom body output has changed. A second odour is
never paired and serves as the control - if it changes too, nothing specific was
learned, only global drift.

Reward is delivered by driving the PAM dopaminergic neurons, which is what
sugar does in a real fly; punishment by driving PPL1.
"""

import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flykenstein.brain.lif import Brain, LIFParams
from flykenstein.brain.ports import Ports
from flykenstein.brain.plasticity import MushroomBody, PlasticityParams


def present(brain, ports, mb, stim, ms, learn=False, dan_hz=0.0, dan_set=None,
            plast_ms=25.0):
    """Run a stimulus, optionally with dopamine and plasticity on."""
    rates = ports.drive(stim)
    if dan_hz and dan_set is not None:
        rates[dan_set] = dan_hz
    steps = int(round(ms / brain.p.dt))
    chunk = int(round(plast_ms / brain.p.dt))
    total = np.zeros(brain.n, dtype=np.float32)
    counts = np.zeros(brain.n, dtype=np.float32)
    for i in range(steps):
        counts[brain.step(poisson=rates)] += 1
        if (i + 1) % chunk == 0:
            total += counts
            if learn:
                mb.update(counts, plast_ms)
            counts[:] = 0
    total += counts
    return total


def mbon_rate(counts, ports, ms):
    idx = ports.readout["mbon"].idx
    return float(counts[idx].sum() / len(idx) / (ms / 1000.0))


def kc_rate(counts, ports, ms):
    idx = ports.readout["kenyon"].idx
    return float(counts[idx].sum() / len(idx) / (ms / 1000.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-ms", type=float, default=800.0)
    ap.add_argument("--train-ms", type=float, default=3000.0)
    ap.add_argument("--eta", type=float, default=4.0)
    ap.add_argument("--reward", choices=["PAM", "PPL1"], default="PAM")
    ap.add_argument("--unpaired", action="store_true",
                    help="deliver dopamine without the odour (temporal control)")
    a = ap.parse_args()

    brain = Brain(params=LIFParams(), seed=0)
    ports = Ports(brain)
    brain.exempt[ports.exempt_sensory()] = True
    mb = MushroomBody(brain, ports, PlasticityParams(eta=a.eta))

    pt = brain.meta["primary_type"].astype(str)
    dan_set = brain.meta.index[pt.str.startswith(a.reward)].to_numpy(np.int32)
    print(f"{a.reward} dopaminergic neurons: {len(dan_set)}")
    print(f"plastic KC->MBON synapses: {len(mb.ptr):,} in {int(mb.has_da.sum())} compartments\n")

    CS_PLUS = {"odor_good_l": 1.0, "odor_good_r": 1.0}    # the paired odour
    CS_MINUS = {"odor_bad_l": 1.0, "odor_bad_r": 1.0}     # never paired

    kc_slot = {v: i for i, v in enumerate(mb.kc)}

    def test(label):
        out = {}
        for name, stim in (("CS+", CS_PLUS), ("CS-", CS_MINUS)):
            brain.reset(); brain.rng = np.random.default_rng(7)
            c = present(brain, ports, mb, stim, a.test_ms)
            active = np.flatnonzero(c[mb.kc] > 0)
            out[name] = {"mbon": mbon_rate(c, ports, a.test_ms),
                         "kc": kc_rate(c, ports, a.test_ms),
                         "active": active,
                         "kc_drive": float(mb.kc_drive(c).sum())}
        print(f"{label:9s} KC->MBON drive  CS+ {out['CS+']['kc_drive']:9.0f}   "
              f"CS- {out['CS-']['kc_drive']:9.0f}     "
              f"(MBON pop. {out['CS+']['mbon']:.2f} / {out['CS-']['mbon']:.2f} Hz)")
        return out

    before = test("before")
    plus_kcs = set(before["CS+"]["active"].tolist())
    minus_kcs = set(before["CS-"]["active"].tolist())
    only_plus = np.array(sorted(plus_kcs - minus_kcs), dtype=int)
    only_minus = np.array(sorted(minus_kcs - plus_kcs), dtype=int)

    brain.reset(); brain.rng = np.random.default_rng(11)
    train_stim = {} if a.unpaired else CS_PLUS
    kind = "dopamine alone (unpaired)" if a.unpaired else "CS+ paired with dopamine"
    print(f"\ntraining : {kind}, {a.train_ms:.0f} ms")
    present(brain, ports, mb, train_stim, a.train_ms, learn=True,
            dan_hz=150.0, dan_set=dan_set)
    print(f"           total KC->MBON weight lost: {mb.depression()*100:.1f}%")
    comp = mb.per_compartment()
    top = np.argsort(comp)[::-1][:5]
    names = brain.meta["primary_type"].iloc[mb.mbon[top]].astype(str).tolist()
    print("           most depressed compartments: "
          + ", ".join(f"{n} {comp[i]*100:.0f}%" for n, i in zip(names, top)))

    # The direct test of associative specificity: are the synapses of the
    # Kenyon cells that code the paired odour more depressed than those of the
    # Kenyon cells that code the odour never paired with anything?
    w = brain.W.data[mb.ptr]
    lost = (mb.w0 - w) / np.maximum(mb.w0, 1e-9)
    def lost_on(kcs):
        if not len(kcs):
            return float("nan")
        sel = np.isin(mb.pre_slot, kcs)
        return float(lost[sel].mean() * 100) if sel.any() else float("nan")
    lp, lm = lost_on(only_plus), lost_on(only_minus)
    print(f"           depression on CS+-only KC synapses : {lp:.1f}%  ({len(only_plus)} KCs)")
    print(f"           depression on CS--only KC synapses : {lm:.1f}%  ({len(only_minus)} KCs)")
    print(f"           associative specificity            : {lp - lm:+.1f} points\n")

    after = test("after")

    def pct(k, key):
        b0, a0 = before[k][key], after[k][key]
        return 100 * (a0 - b0) / max(abs(b0), 1e-9)
    dplus, dminus = pct("CS+", "kc_drive"), pct("CS-", "kc_drive")
    print(f"\nchange in KC->MBON drive:  CS+ {dplus:+.1f}%     CS- {dminus:+.1f}%")
    print(f"learning specific to the paired odour: {dplus - dminus:+.1f} percentage points")
    print(f"change in MBON population rate: "
          f"CS+ {pct('CS+','mbon'):+.1f}%  CS- {pct('CS-','mbon'):+.1f}%  "
          f"(diluted by non-mushroom-body input)")


if __name__ == "__main__":
    main()
