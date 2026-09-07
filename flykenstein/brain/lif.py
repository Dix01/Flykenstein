"""Leaky integrate-and-fire engine over the FlyWire connectome.

Dynamics and constants follow Shiu et al. 2024 (Nature 634:210), which in turn
takes its membrane parameters from Kakaria & de Bivort 2017, its synaptic time
constant from Juergensen et al. 2021, its refractory period from Lazar et al.
2021 and its synaptic delay from Paul et al. 2015:

    dv/dt = (v_0 - v + g) / t_mbr        (unless refractory)
    dg/dt = -g / tau
    spike when v > v_th  ->  v = v_rst, g = 0, refractory for t_rfc
    presynaptic spike    ->  g_post += w_syn * signed_synapse_count, after t_dly

Everything is in millivolts and milliseconds. The implementation is a plain
NumPy/SciPy loop: at 139k neurons and 2.7M edges the per-step cost is dominated
by gathering the rows of the CSR matrix belonging to neurons that actually
spiked, which is cheap because fly firing rates are low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LIFParams:
    v_0: float = -52.0        # resting potential (mV)
    v_rst: float = -52.0      # reset potential (mV)
    v_th: float = -45.0       # spike threshold (mV)
    t_mbr: float = 20.0       # membrane time constant (ms)
    tau: float = 5.0          # synaptic time constant (ms)
    t_rfc: float = 2.2        # refractory period (ms)
    t_dly: float = 1.8        # synaptic delay (ms)
    w_syn: float = 0.275      # mV per synapse
    dt: float = 0.1           # integration step (ms)

    # Neuromodulator channel (not part of the published fast model).
    mod_tau: float = 500.0    # ms; monoamine effects outlast synaptic ones
    mod_gain: float = 0.0     # mV per unit modulator; 0 keeps the baseline model
    f_poi: float = 250.0      # scaling of Poisson input synapses (Shiu et al.)


class Brain:
    """A connectome-constrained spiking network."""

    def __init__(self, npz: Path | str | None = None,
                 meta: Path | str | None = None,
                 params: LIFParams | None = None,
                 seed: int | None = None):
        npz = Path(npz or ROOT / "data" / "brain_783.npz")
        meta = Path(meta or ROOT / "data" / "neurons_783.parquet")
        z = np.load(npz, allow_pickle=False)
        n = int(z["shape"][0])

        self.n = n
        self.W = sparse.csr_matrix(
            (z["fast_data"], z["fast_indices"], z["fast_indptr"]), shape=(n, n))
        self.M = sparse.csr_matrix(
            (z["mod_data"], z["mod_indices"], z["mod_indptr"]), shape=(n, n))
        # Monoamine edges sit in the fast matrix as explicit zeros; drop them so
        # nnz means "edge that can carry current" and so synapse indices are stable.
        self.W.eliminate_zeros()
        self.root_id = z["root_id"]
        self.mod_kind = z["mod_kind"]
        self.meta = pd.read_parquet(meta)
        self.index_of = pd.Series(np.arange(n, dtype=np.int32), index=self.root_id)

        self.p = params or LIFParams()
        self.rng = np.random.default_rng(seed)

        self.f_poi = self.p.f_poi
        self._delay_steps = max(1, int(round(self.p.t_dly / self.p.dt)))
        self.reset()

    # ---- state ---------------------------------------------------------
    def reset(self) -> None:
        p = self.p
        self.v = np.full(self.n, p.v_0, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.mod = np.zeros(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.float32)   # ms remaining
        self.exempt = np.zeros(self.n, dtype=bool)         # no refractory (driven inputs)
        self._ring = np.zeros((self._delay_steps, self.n), dtype=np.float32)
        self._ring_i = 0
        self.t = 0.0
        self.step_count = 0

    # ---- helpers -------------------------------------------------------
    def ids(self, **query) -> np.ndarray:
        """Indices of neurons matching metadata equality/`in` constraints."""
        m = pd.Series(True, index=self.meta.index)
        for col, want in query.items():
            series = self.meta[col]
            if isinstance(want, (list, tuple, set, np.ndarray)):
                m &= series.isin(list(want))
            else:
                m &= series == want
        return self.meta.index[m].to_numpy(dtype=np.int32)

    def _gather(self, spiked: np.ndarray, mat: sparse.csr_matrix) -> np.ndarray:
        """Sum the CSR rows of `spiked` into a dense postsynaptic vector."""
        out = np.zeros(self.n, dtype=np.float32)
        if spiked.size == 0:
            return out
        indptr, indices, data = mat.indptr, mat.indices, mat.data
        starts = indptr[spiked]
        counts = indptr[spiked + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return out
        # Expand each row's [start, start+count) slice without a Python loop.
        cum = np.concatenate(([0], np.cumsum(counts)[:-1]))
        offsets = np.repeat(starts - cum, counts) + np.arange(total)
        return np.bincount(indices[offsets], weights=data[offsets],
                           minlength=self.n).astype(np.float32)

    # ---- integration ---------------------------------------------------
    def step(self, inject: np.ndarray | None = None,
             poisson: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt. Returns indices of neurons that spiked this step.

        inject  : per-neuron voltage added this step (mV), e.g. sensory drive
        poisson : per-neuron Poisson rate in Hz, delivered as w_syn*f_poi kicks
        """
        p, dt = self.p, self.p.dt

        # Synaptic input scheduled t_dly ago.
        arrived = self._ring[self._ring_i].copy()
        self._ring[self._ring_i] = 0.0

        self.g += arrived
        if p.mod_gain:
            self.g += p.mod_gain * self.mod
        if inject is not None:
            self.g += inject
        poisson_kick = None
        if poisson is not None:
            k = self.rng.poisson(np.maximum(poisson, 0.0) * dt * 1e-3)
            # Shiu et al. drive v directly (target_var='v') with w_syn * f_poi.
            poisson_kick = (k * p.w_syn * self.f_poi).astype(np.float32)

        active = self.refrac <= 0.0
        if poisson_kick is not None:
            self.v += poisson_kick
        self.v[active] += dt * (p.v_0 - self.v[active] + self.g[active]) / p.t_mbr
        self.g[active] -= dt * self.g[active] / p.tau
        self.mod -= dt * self.mod / p.mod_tau
        self.refrac[~active] -= dt

        spiked = np.flatnonzero((self.v > p.v_th) & active).astype(np.int32)
        if spiked.size:
            self.v[spiked] = p.v_rst
            self.g[spiked] = 0.0
            rf = np.where(self.exempt[spiked], 0.0, p.t_rfc)
            self.refrac[spiked] = rf
            # Schedule fast synaptic input for arrival after t_dly.
            # Slot self._ring_i is read again after a full lap, i.e. t_dly from now.
            tgt = self._ring_i
            self._ring[tgt] += self._gather(spiked, self.W) * p.w_syn
            if self.M.nnz:
                self.mod += self._gather(spiked, self.M) * p.w_syn

        self._ring_i = (self._ring_i + 1) % self._delay_steps
        self.t += dt
        self.step_count += 1
        return spiked

    def run(self, ms: float, **kw) -> list[np.ndarray]:
        return [self.step(**kw) for _ in range(int(round(ms / self.p.dt)))]

    def silence(self, idx: np.ndarray) -> None:
        """Zero all outgoing weights of `idx` (the standard ablation control)."""
        for i in np.atleast_1d(idx):
            s, e = self.W.indptr[i], self.W.indptr[i + 1]
            self.W.data[s:e] = 0.0
