"""Learning where the fly actually does it: the KC -> MBON synapse.

The mushroom body is the fly's associative memory. Kenyon cells carry a sparse,
high-dimensional code for the current odour; MBONs read that code out and bias
approach or avoidance; dopaminergic neurons signal that something good or bad
just happened. The established rule (Hige et al. 2015, Cohn et al. 2015, Aso &
Rubin 2016) is *dopamine-gated depression*: when a Kenyon cell is active and
dopamine is present in the same compartment, that Kenyon cell's synapse onto
that compartment's MBON gets weaker, and stays weaker.

That is the whole trick. The odour that was present when the reward arrived
stops driving the MBON that suppresses approach, so next time the same odour
arrives, the animal approaches. The memory is a set of weakened synapses.

Two things here are taken from the connectome rather than invented:

* which synapses can learn - the 21,438 real KC->MBON edges;
* which dopaminergic neurons govern which of them - the compartment structure,
  read off the dopaminergic DAN->MBON innervation in the modulatory matrix.

The learning rate, the trace time constants and the forgetting rate are ours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class PlasticityParams:
    eta: float = 0.9              # depression rate (per second at full drive)
    tau_elig_ms: float = 1200.0   # how long a Kenyon cell stays eligible
    tau_da_ms: float = 400.0      # dopamine decay in a compartment
    tau_forget_s: float = 180.0   # drift of weights back to baseline
    kc_ref_hz: float = 20.0       # KC rate treated as fully active
    da_ref_hz: float = 20.0       # DAN rate treated as full dopamine
    floor: float = 0.15           # a synapse never falls below this of baseline


class MushroomBody:
    """Plastic KC->MBON synapses under dopaminergic control."""

    def __init__(self, brain, ports, params: PlasticityParams | None = None):
        self.brain = brain
        self.p = params or PlasticityParams()
        self.kc = ports.readout["kenyon"].idx
        self.mbon = ports.readout["mbon"].idx
        self.dan = ports.readout["dan"].idx

        n = brain.n
        W = brain.W
        is_mbon = np.zeros(n, dtype=bool)
        is_mbon[self.mbon] = True
        mbon_slot = np.full(n, -1, dtype=np.int32)
        mbon_slot[self.mbon] = np.arange(len(self.mbon))
        kc_slot = np.full(n, -1, dtype=np.int32)
        kc_slot[self.kc] = np.arange(len(self.kc))

        ptr, pre, post = [], [], []
        for i in self.kc:
            s, e = W.indptr[i], W.indptr[i + 1]
            cols = W.indices[s:e]
            sel = np.flatnonzero(is_mbon[cols])
            if sel.size:
                ptr.append(s + sel)
                pre.append(np.full(sel.size, i, dtype=np.int32))
                post.append(cols[sel])
        self.ptr = np.concatenate(ptr) if ptr else np.zeros(0, np.int64)
        self.pre = np.concatenate(pre) if pre else np.zeros(0, np.int32)
        self.post = np.concatenate(post) if post else np.zeros(0, np.int32)
        self.pre_slot = kc_slot[self.pre]
        self.post_slot = mbon_slot[self.post]
        self.w0 = W.data[self.ptr].copy()

        # Compartments: which dopaminergic neurons innervate which MBON.
        C = brain.M[self.dan][:, self.mbon].toarray()
        colsum = C.sum(axis=0, keepdims=True)
        self.C = np.divide(C, colsum, out=np.zeros_like(C), where=colsum > 0)
        self.has_da = (colsum.ravel() > 0)

        self.reset_state()

    # ---- state ---------------------------------------------------------
    def reset_state(self) -> None:
        self.elig = np.zeros(len(self.kc), dtype=np.float32)
        self.da = np.zeros(len(self.mbon), dtype=np.float32)

    def reset_weights(self) -> None:
        self.brain.W.data[self.ptr] = self.w0
        self.reset_state()

    # ---- the rule -------------------------------------------------------
    def update(self, counts: np.ndarray, dt_ms: float) -> dict:
        """Apply one plasticity step from a window of spike counts."""
        p = self.p
        dt_s = dt_ms / 1000.0

        kc_hz = counts[self.kc] / dt_s
        dan_hz = counts[self.dan] / dt_s

        self.elig *= np.exp(-dt_ms / p.tau_elig_ms)
        self.elig += np.clip(kc_hz / p.kc_ref_hz, 0, 1)
        np.clip(self.elig, 0, 1, out=self.elig)

        self.da *= np.exp(-dt_ms / p.tau_da_ms)
        self.da += self.C.T @ np.clip(dan_hz / p.da_ref_hz, 0, 1)
        np.clip(self.da, 0, 1, out=self.da)

        w = self.brain.W.data[self.ptr]
        drive = self.elig[self.pre_slot] * self.da[self.post_slot]
        w -= p.eta * dt_s * drive * self.w0                    # dopamine-gated depression
        w += (self.w0 - w) * (dt_s / p.tau_forget_s)           # slow forgetting
        np.clip(w, p.floor * self.w0, self.w0, out=w)
        self.brain.W.data[self.ptr] = w

        return {"da_max": float(self.da.max()),
                "elig_mean": float(self.elig.mean()),
                "depressed": float(self.depression())}

    # ---- reading the memory out -----------------------------------------
    def depression(self) -> float:
        """Fraction of total KC->MBON weight currently lost to learning."""
        w = self.brain.W.data[self.ptr]
        return float(1.0 - w.sum() / self.w0.sum())

    def per_compartment(self) -> np.ndarray:
        """Depression per MBON - the memory, one number per compartment."""
        w = self.brain.W.data[self.ptr]
        lost = np.bincount(self.post_slot, weights=(self.w0 - w), minlength=len(self.mbon))
        base = np.bincount(self.post_slot, weights=self.w0, minlength=len(self.mbon))
        return np.divide(lost, base, out=np.zeros_like(lost), where=base > 0)

    # ---- persistence: this is what makes it a life and not a run ---------
    def save(self, path) -> None:
        np.savez_compressed(path, w=self.brain.W.data[self.ptr], w0=self.w0,
                            elig=self.elig, da=self.da)

    def load(self, path) -> None:
        z = np.load(path)
        if z["w"].shape != self.ptr.shape:
            raise ValueError("saved memory does not match this connectome build")
        self.brain.W.data[self.ptr] = z["w"]
        self.elig = z["elig"]
        self.da = z["da"]
