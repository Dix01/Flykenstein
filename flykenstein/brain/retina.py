"""Map NeuroMechFly's ommatidia onto real R1-6 photoreceptors, retinotopically.

flygym renders each compound eye as a set of ommatidia with known positions.
FlyWire's column assignment gives every optic-lobe neuron a hexagonal column
coordinate (p, q). Matching one to the other by nearest neighbour in a shared
2-D layout means a bright patch in the upper-left of the visual field drives the
photoreceptors of the columns that actually look up and to the left, instead of
the whole eye lighting up together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


class Retina:
    def __init__(self, brain, ommatidia_xy: dict[str, np.ndarray] | None = None):
        m = brain.meta
        pt = m["primary_type"].astype(str)
        self.eyes = {}
        for eye, sd in (("left", "left"), ("right", "right")):
            idx = m.index[pt.str.startswith("R1-6") & (m["side"] == sd)].to_numpy(np.int32)
            xy = m.loc[idx, ["x", "y"]].to_numpy(dtype=float) if len(idx) else np.zeros((0, 2))
            if len(xy):
                xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-9)
            self.eyes[eye] = {"idx": idx, "xy": xy, "map": None}
        self.ommatidia_xy = ommatidia_xy

    def fit(self, ommatidia_xy: dict[str, np.ndarray]) -> None:
        """Assign every photoreceptor the nearest ommatidium in normalised space."""
        for eye, spec in self.eyes.items():
            om = np.asarray(ommatidia_xy[eye], dtype=float)
            if not len(om) or not len(spec["xy"]):
                spec["map"] = np.zeros(len(spec["idx"]), dtype=np.int32)
                continue
            omn = (om - om.mean(0)) / (om.std(0) + 1e-9)
            d = ((spec["xy"][:, None, :] - omn[None, :, :]) ** 2).sum(-1)
            spec["map"] = np.argmin(d, axis=1).astype(np.int32)

    def drive(self, eye: str, intensities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-photoreceptor drive for one eye, given per-ommatidium intensity."""
        spec = self.eyes[eye]
        if spec["map"] is None or not len(spec["idx"]):
            return spec["idx"], np.zeros(len(spec["idx"]), dtype=np.float32)
        v = np.asarray(intensities, dtype=np.float32).ravel()
        return spec["idx"], v[np.clip(spec["map"], 0, len(v) - 1)]
