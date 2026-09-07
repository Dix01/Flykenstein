"""A heading signal for the central complex, built from real anatomy.

The fly's compass lives in the ellipsoid body. EPG neurons tile it as a ring,
and a bump of activity travels around that ring as the animal turns, so the
bump's angular position *is* the animal's heading. In a real fly the bump is
anchored by visual landmarks and updated by self-motion.

There is no visual landmark system here, so the world hands the bump its angle
directly. What is not invented is the map: each EPG neuron's preferred heading
is taken from where it actually sits around the centroid of the ellipsoid body
in the FlyWire volume. The ring is anatomical; only the anchoring is ours.
"""

from __future__ import annotations

import numpy as np


class Compass:
    def __init__(self, brain, ports=None, kappa: float = 2.5):
        m = brain.meta
        pt = m["primary_type"].astype(str)
        self.idx = m.index[pt == "EPG"].to_numpy(dtype=np.int32)
        xy = m.loc[self.idx, ["x", "y"]].to_numpy(dtype=float)
        centre = xy.mean(axis=0)
        d = xy - centre
        self.preferred = np.arctan2(d[:, 1], d[:, 0])   # radians, one per EPG
        self.kappa = kappa                              # bump width (von Mises)

    def drive(self, heading_rad: float, level: float = 1.0) -> np.ndarray:
        """Von Mises bump over the EPG ring, peaked at `heading_rad`."""
        w = np.exp(self.kappa * (np.cos(self.preferred - heading_rad) - 1.0))
        return level * w.astype(np.float32)

    def read(self, counts: np.ndarray, dt_s: float) -> tuple[float, float]:
        """Decode the bump back out: (angle, vector strength)."""
        r = counts[self.idx] / max(dt_s, 1e-9)
        if r.sum() <= 0:
            return float("nan"), 0.0
        v = (r * np.exp(1j * self.preferred)).sum() / r.sum()
        return float(np.angle(v)), float(np.abs(v))
