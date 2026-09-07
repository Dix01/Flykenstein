"""The laws of this world.

Everything here is ours, not the fly's. The connectome is fixed biology; this
file is the part we get to invent. Changing a number here changes what kind of
world the animal was released into, which is the whole point of the exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass
class WorldRules:
    # --- geometry -------------------------------------------------------
    arena_radius_mm: float = 60.0        # beyond this the run ends
    food_sites: tuple = ((12.0, 6.0), (-9.0, -14.0), (20.0, -18.0))
    hazard_sites: tuple = ((-14.0, 10.0), (6.0, -22.0))
    site_radius_mm: float = 2.0          # contact radius for taste
    odor_peak: float = 1.0               # peak intensity at a source

    # --- physiology we impose -------------------------------------------
    energy_start: float = 1.0
    energy_drain_per_s: float = 0.02     # cost of simply existing
    energy_per_feed_s: float = 0.35      # gain while proboscis is extended on food
    hazard_cost_per_s: float = 0.25      # cost of sitting in a hazard
    starve_threshold: float = 0.0        # run ends here

    # --- sensory gains (world units -> Poisson Hz on a sensory port) -----
    base_hz: float = 150.0               # Shiu et al. stimulation rate
    odor_gain: float = 1.0
    odor_ref: float = 0.15               # sensor reading treated as full-scale odor
    taste_gain: float = 1.0
    touch_gain: float = 0.4
    vision_gain: float = 0.6
    light_level: float = 1.0             # 0 = darkness

    # --- motor decoding (the hand-written half; see ports.py) ------------
    dn_window_ms: float = 60.0           # sliding window for DN firing rates
    dn_rate_ref_hz: float = 20.0         # DN rate mapped to full drive
    drive_base: float = 1.0              # forward CPG amplitude with no DN input
    drive_turn_gain: float = 0.8
    drive_min: float = -0.5
    drive_max: float = 1.5

    # --- internal machinery ----------------------------------------------
    mod_gain: float = 0.6                # monoamine gain on synaptic drive
    learning: bool = True                # KC->MBON plasticity on
    plasticity_eta: float = 4.0
    compass_hz: float = 80.0             # drive on the EPG heading bump
    life_path: str = "life"              # where memory and drives persist
    carry_memory: bool = True            # load that memory before releasing

    # --- feedback the world writes back into the brain -------------------
    # Our rule, not the fly's: eating stimulates the dopaminergic population,
    # so the world can teach. Set to 0 for a purely feedforward release.
    reward_dan_hz: float = 150.0         # PAM dopaminergic neurons, on food
    punish_dan_hz: float = 150.0         # PPL1 dopaminergic neurons, on hazard

    # --- run control -----------------------------------------------------
    duration_s: float = 5.0
    control_ms: float = 5.0              # sensory/motor exchange interval
    seed: int = 0

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)

    @classmethod
    def load(cls, path) -> "WorldRules":
        with open(path) as f:
            d = json.load(f)
        for k in ("food_sites", "hazard_sites"):
            d[k] = tuple(tuple(x) for x in d[k])
        return cls(**d)
