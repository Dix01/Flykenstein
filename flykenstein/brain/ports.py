"""Where the world touches the brain, and where the brain touches the world.

Sensory ports are sets of real FlyWire neurons that a world stimulus is allowed
to drive. Motor ports are sets of descending neurons whose firing rate the body
is allowed to read. Every set is selected from published annotations - cell
types, super-classes and community labels - not invented.

The honest caveat: the *sensory* side is grounded (these really are the sugar
GRNs, these really are the ORNs of a given glomerulus). The *motor* side is
only half grounded. The descending neurons are real and their behavioural roles
are established in the literature, but the mapping from their firing rates onto
a two-dimensional walking drive is a hand-written convention, because the
brain-to-muscle wiring lives in the ventral nerve cord, which is a different
dataset. Eon's embodied fly makes the same compromise and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The 21 right-hemisphere sugar-sensing GRNs stimulated in Shiu et al. 2024.
SUGAR_GRN_R = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265  # proboscis extension motor neuron

# Olfactory receptor neuron classes with established valence.
ORN_ATTRACTIVE = ["ORN_DM1", "ORN_DM4", "ORN_DM2", "ORN_VM2"]   # vinegar / food esters
ORN_AVERSIVE = ["ORN_DA2", "ORN_DL5", "ORN_V"]                  # geosmin, aversive, CO2


@dataclass
class Port:
    name: str
    idx: np.ndarray
    note: str = ""

    def __len__(self) -> int:
        return len(self.idx)

    def __repr__(self) -> str:
        return f"<Port {self.name}: {len(self.idx)} neurons>"


class Ports:
    """Named neuron sets for driving and reading a `Brain`."""

    def __init__(self, brain):
        self.brain = brain
        m = brain.meta
        ptype = m["primary_type"].astype("string").fillna("")

        def by_ids(ids):
            s = brain.index_of.reindex(ids).dropna()
            return s.astype(int).to_numpy()

        def by_type(types):
            return m.index[ptype.isin(types)].to_numpy(dtype=np.int32)

        def by(**q):
            return brain.ids(**q)

        side = m["side"].astype("string").fillna("")

        # ---- sensory -------------------------------------------------
        self.sensory: dict[str, Port] = {}
        self._add("sugar", by_ids(SUGAR_GRN_R),
                  "labellar sugar GRNs; the Shiu et al. stimulus set")
        self._add("bitter", by_type(["LB1a", "LB1b", "LB1c", "LB1e", "LB2a-b"]),
                  "labellar bristle GRNs of bitter-associated classes")
        for label, types in (("good", ORN_ATTRACTIVE), ("bad", ORN_AVERSIVE)):
            idx = by_type(types)
            sd = side.iloc[idx].to_numpy()
            self._add(f"odor_{label}", idx, f"ORNs of {label} glomeruli, both antennae")
            self._add(f"odor_{label}_l", idx[sd == "left"], f"ORNs of {label} glomeruli, left")
            self._add(f"odor_{label}_r", idx[sd == "right"], f"ORNs of {label} glomeruli, right")
        self._add("touch", by(**{"class": "mechanosensory"}), "all mechanosensory neurons")
        self._add("wind", by_type(["JO-B", "JO-E", "JO-F"]),
                  "Johnston's organ; antennal air current and sound")
        self._add("vision_l", m.index[ptype.str.startswith("R1-6") & (side == "left")].to_numpy(np.int32),
                  "left R1-6 photoreceptors")
        self._add("vision_r", m.index[ptype.str.startswith("R1-6") & (side == "right")].to_numpy(np.int32),
                  "right R1-6 photoreceptors")

        # ---- motor / descending --------------------------------------
        self.motor: dict[str, Port] = {}
        steer = by_type(["DNa02", "DNa01"])
        steer_side = side.iloc[steer].to_numpy()
        self._add_motor("turn_left", steer[steer_side == "left"],
                        "DNa01/DNa02, left; steering")
        self._add_motor("turn_right", steer[steer_side == "right"],
                        "DNa01/DNa02, right; steering")
        self._add_motor("stop", by_type(["DNp09"]), "DNp09; walking arrest / freeze")
        self._add_motor("backward", by_type(["MDN"]), "moonwalker DNs; backward walking")
        self._add_motor("groom", by_type(["DNg12", "DNg12_a", "DNg12_b"]), "DNg12; grooming")
        self._add_motor("all_dn", by(super_class="descending"), "every descending neuron")
        self._add_motor("proboscis", by_ids([MN9]), "MN9; proboscis extension")

        # ---- internal readouts ---------------------------------------
        self.readout: dict[str, Port] = {}
        for nm, q, note in [
            ("kenyon", {"class": "Kenyon_Cell"}, "mushroom body; associative memory"),
            ("mbon", {"class": "MBON"}, "mushroom body output; learned valence"),
            ("dan", {"class": "DAN"}, "dopaminergic; teaching signal"),
            ("central_complex", {"class": "CX"}, "central complex; heading and navigation"),
        ]:
            self.readout[nm] = Port(nm, by(**q), note)

    def _add(self, name, idx, note):
        self.sensory[name] = Port(name, np.asarray(idx, dtype=np.int32), note)

    def _add_motor(self, name, idx, note):
        self.motor[name] = Port(name, np.asarray(idx, dtype=np.int32), note)

    # ---- driving and reading ------------------------------------------
    def drive(self, stimuli: dict[str, float], base_hz: float = 150.0) -> np.ndarray:
        """Build a Poisson rate vector from {port_name: intensity in 0..1}."""
        rates = np.zeros(self.brain.n, dtype=np.float32)
        for name, level in stimuli.items():
            if level <= 0:
                continue
            port = self.sensory[name]
            rates[port.idx] = np.maximum(rates[port.idx], float(level) * base_hz)
        return rates

    def exempt_sensory(self) -> np.ndarray:
        """Indices of all sensory-port neurons (driven units skip refractoriness)."""
        return np.unique(np.concatenate([p.idx for p in self.sensory.values() if len(p)]))

    def summary(self) -> pd.DataFrame:
        rows = []
        for kind, group in (("sensory", self.sensory), ("motor", self.motor),
                            ("readout", self.readout)):
            for p in group.values():
                rows.append({"kind": kind, "port": p.name, "n": len(p), "note": p.note})
        return pd.DataFrame(rows)
