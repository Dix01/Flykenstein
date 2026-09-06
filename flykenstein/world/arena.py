"""The arena the fly is released into: flat ground, food, hazards, odor plumes."""

from __future__ import annotations

import numpy as np
from flygym.arena import OdorArena

from .rules import WorldRules


class FlykensteinArena(OdorArena):
    """An OdorArena whose sources are our food and hazard sites.

    Odor channel 0 is the attractive one (food), channel 1 the aversive one
    (hazard). NeuroMechFly's antennal sensors report both channels; the loop
    routes them to the attractive and aversive ORN ports respectively.
    """

    def __init__(self, rules: WorldRules, **kwargs):
        food = np.array(rules.food_sites, dtype=float)
        hazard = np.array(rules.hazard_sites, dtype=float)
        sources, peaks, colors = [], [], []

        for x, y in food:
            sources.append([x, y, 1.5])
            peaks.append([rules.odor_peak, 0.0])
            colors.append((0.2, 0.9, 0.3, 1.0))
        for x, y in hazard:
            sources.append([x, y, 1.5])
            peaks.append([0.0, rules.odor_peak])
            colors.append((0.9, 0.2, 0.2, 1.0))

        super().__init__(
            odor_source=np.array(sources),
            peak_odor_intensity=np.array(peaks),
            diffuse_func=lambda x: x**-2,
            marker_colors=colors,
            marker_size=0.4,
            **kwargs,
        )
        self.rules = rules
        self.food_xy = food
        self.hazard_xy = hazard

    # ---- world queries the loop needs ----------------------------------
    def _nearest(self, xy: np.ndarray, sites: np.ndarray) -> tuple[int, float]:
        if len(sites) == 0:
            return -1, np.inf
        d = np.linalg.norm(sites - np.asarray(xy)[:2], axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def on_food(self, xy) -> tuple[bool, float]:
        _, d = self._nearest(xy, self.food_xy)
        return d <= self.rules.site_radius_mm, d

    def on_hazard(self, xy) -> tuple[bool, float]:
        _, d = self._nearest(xy, self.hazard_xy)
        return d <= self.rules.site_radius_mm, d

    def out_of_bounds(self, xy) -> bool:
        return float(np.linalg.norm(np.asarray(xy)[:2])) > self.rules.arena_radius_mm
