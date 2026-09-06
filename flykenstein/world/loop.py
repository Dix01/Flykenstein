"""The closed loop: world -> senses -> connectome -> descending neurons -> body.

One control tick is: read the body's sensors, convert them to Poisson drive on
real sensory neurons, integrate the whole connectome for `control_ms`, measure
the firing rate of the steering/stop/backward descending neurons, turn that
into a walking drive, and step the physics for the same `control_ms`.

Nothing here trains anything. The behaviour, such as it is, comes out of the
wiring plus the rules in rules.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..brain.lif import Brain, LIFParams
from ..brain.ports import Ports
from .arena import FlykensteinArena
from .rules import WorldRules


@dataclass
class Telemetry:
    rows: list = field(default_factory=list)

    def log(self, **kw):
        self.rows.append(kw)

    def frame(self):
        import pandas as pd
        return pd.DataFrame(self.rows)


class Release:
    """A fly, a brain, a world, and a run."""

    def __init__(self, rules: WorldRules | None = None, vision: bool = False,
                 render: bool = True, brain: Brain | None = None):
        from flygym import Fly, Camera
        from flygym.examples.locomotion import HybridTurningController

        self.rules = rules or WorldRules()
        self.brain = brain or Brain(params=LIFParams(), seed=self.rules.seed)
        self.ports = Ports(self.brain)
        self.brain.exempt[self.ports.exempt_sensory()] = True

        self.arena = FlykensteinArena(self.rules)
        self.fly = Fly(
            enable_olfaction=True,
            enable_vision=vision,
            enable_adhesion=True,
            spawn_pos=(0.0, 0.0, 0.25),
            contact_sensor_placements=[
                f"{leg}{seg}" for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
                for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
            ],
        )
        cams = []
        if render:
            cams = [Camera(attachment_point=self.fly.model.worldbody,
                           camera_name="camera_top",
                           targeted_fly_names=[self.fly.name],
                           play_speed=0.5)]
        self.sim = HybridTurningController(fly=self.fly, cameras=cams, arena=self.arena,
                                           timestep=1e-4, seed=self.rules.seed)
        self.camera = cams[0] if cams else None
        self.vision = vision

        self.telemetry = Telemetry()
        self._dn_ema: dict[str, float] = {}
        self.energy = self.rules.energy_start
        self.feeds = 0
        self.hazard_hits = 0
        self.ended = ""

    # ---- encoding: world -> sensory neurons -----------------------------
    def encode(self, obs) -> tuple[dict, dict]:
        r = self.rules
        xy = obs["fly"][0][:2]
        stim: dict[str, float] = {}

        # Sensor order is [L palp, R palp, L antenna, R antenna]; keeping the
        # two sides apart is what gives the brain a gradient to steer on.
        odor = np.asarray(obs.get("odor_intensity", np.zeros((2, 4))), dtype=float)
        L, R = [0, 2], [1, 3]
        for ch, label in ((0, "good"), (1, "bad")):
            if odor.shape[0] <= ch:
                continue
            lv = float(np.mean(odor[ch][L])) / r.odor_ref * r.odor_gain
            rv = float(np.mean(odor[ch][R])) / r.odor_ref * r.odor_gain
            stim[f"odor_{label}_l"] = float(np.clip(lv, 0, 1))
            stim[f"odor_{label}_r"] = float(np.clip(rv, 0, 1))

        on_food, d_food = self.arena.on_food(xy)
        on_haz, d_haz = self.arena.on_hazard(xy)
        stim["sugar"] = r.taste_gain if on_food else 0.0
        stim["bitter"] = r.taste_gain if on_haz else 0.0

        contact = float(np.linalg.norm(obs["contact_forces"], axis=1).mean())
        stim["touch"] = np.clip(contact * r.touch_gain / 10.0, 0, 1)

        if self.vision and "vision" in obs:
            v = np.asarray(obs["vision"], dtype=float)          # (2, ommatidia, 2)
            lum = v.mean(axis=(1, 2)) / (v.max() + 1e-9)
            stim["vision_l"] = float(np.clip(lum[0] * r.vision_gain, 0, 1))
            stim["vision_r"] = float(np.clip(lum[1] * r.vision_gain, 0, 1))
        elif r.light_level:
            stim["vision_l"] = stim["vision_r"] = 0.15 * r.light_level

        info = {"on_food": on_food, "on_hazard": on_haz,
                "d_food": d_food, "d_hazard": d_haz, "xy": xy}
        return stim, info

    # ---- decoding: descending neurons -> walking drive -------------------
    def _rate(self, name: str, spike_counts: np.ndarray, window_ms: float) -> float:
        port = self.ports.motor[name]
        if len(port) == 0:
            return 0.0
        hz = spike_counts[port.idx].sum() / len(port) / (window_ms / 1000.0)
        return float(hz)

    def _smooth(self, name: str, value: float) -> float:
        """Exponential moving average over `dn_window_ms`.

        These populations are tiny - two DNa neurons a side - so a single
        control tick reports either 0 Hz or one spike's worth. Averaging over a
        window turns that into the graded signal a real descending pathway
        would carry.
        """
        a = min(1.0, self.rules.control_ms / max(self.rules.dn_window_ms, 1e-6))
        prev = self._dn_ema.get(name, 0.0)
        cur = (1 - a) * prev + a * value
        self._dn_ema[name] = cur
        return cur

    def decode(self, spike_counts: np.ndarray, window_ms: float) -> tuple[np.ndarray, dict]:
        r = self.rules
        left = self._smooth("turn_left", self._rate("turn_left", spike_counts, window_ms))
        right = self._smooth("turn_right", self._rate("turn_right", spike_counts, window_ms))
        stop = self._smooth("stop", self._rate("stop", spike_counts, window_ms))
        back = self._smooth("backward", self._rate("backward", spike_counts, window_ms))
        prob = self._rate("proboscis", spike_counts, window_ms)

        ref = r.dn_rate_ref_hz
        # DNa01/DNa02 activity on one side steers the fly toward that side:
        # it raises the contralateral CPG amplitude. Hand-written convention.
        turn = np.clip((right - left) / ref, -1.0, 1.0)
        drive = np.array([r.drive_base + r.drive_turn_gain * turn,
                          r.drive_base - r.drive_turn_gain * turn])

        gate = np.clip(1.0 - stop / ref, 0.0, 1.0)
        drive *= gate
        if back / ref > 0.5:
            drive = -np.abs(drive)
        drive = np.clip(drive, r.drive_min, r.drive_max)
        return drive, {"dn_left": left, "dn_right": right, "dn_stop": stop,
                       "dn_back": back, "mn9": prob, "turn": turn, "gate": gate}

    # ---- the run ---------------------------------------------------------
    def run(self, verbose: bool = True):
        r = self.rules
        obs, _ = self.sim.reset(seed=r.seed)
        n_ctrl = int(round(r.duration_s * 1000.0 / r.control_ms))
        brain_steps = int(round(r.control_ms / self.brain.p.dt))
        phys_steps = int(round(r.control_ms * 1e-3 / self.sim.timestep))
        t_wall = time.time()

        for tick in range(n_ctrl):
            stim, info = self.encode(obs)
            rates = self.ports.drive(stim, base_hz=r.base_hz)

            # The world's own teaching signal, written straight onto the DANs.
            if info["on_food"] and r.reward_dan_hz:
                rates[self.ports.readout["dan"].idx] = r.reward_dan_hz
            if info["on_hazard"] and r.punish_dan_hz:
                rates[self.ports.readout["dan"].idx] = r.punish_dan_hz

            counts = np.zeros(self.brain.n, dtype=np.float32)
            for _ in range(brain_steps):
                counts[self.brain.step(poisson=rates)] += 1

            drive, dn = self.decode(counts, r.control_ms)

            for _ in range(phys_steps):
                obs, _, term, trunc, _ = self.sim.step(drive)
                if self.camera is not None:
                    self.sim.render()

            # --- our physiology ---
            dt_s = r.control_ms / 1000.0
            self.energy -= r.energy_drain_per_s * dt_s
            if info["on_food"] and dn["mn9"] > 5.0:
                self.energy += r.energy_per_feed_s * dt_s
                self.feeds += 1
            if info["on_hazard"]:
                self.energy -= r.hazard_cost_per_s * dt_s
                self.hazard_hits += 1
            self.energy = float(np.clip(self.energy, -1.0, 2.0))

            self.telemetry.log(
                t_s=(tick + 1) * r.control_ms / 1000.0,
                x=float(info["xy"][0]), y=float(info["xy"][1]),
                heading=float(np.arctan2(*obs["fly_orientation"][:2][::-1])),
                energy=self.energy, spikes=int(counts.sum()),
                active=int((counts > 0).sum()),
                drive_l=float(drive[0]), drive_r=float(drive[1]),
                kc=float(counts[self.ports.readout["kenyon"].idx].sum()),
                mbon=float(counts[self.ports.readout["mbon"].idx].sum()),
                cx=float(counts[self.ports.readout["central_complex"].idx].sum()),
                **{k: float(v) for k, v in dn.items()},
                **{f"in_{k}": float(v) for k, v in stim.items()},
                on_food=bool(info["on_food"]), on_hazard=bool(info["on_hazard"]),
            )

            if verbose and tick % max(1, n_ctrl // 20) == 0:
                print(f"  t={self.telemetry.rows[-1]['t_s']:5.2f}s  "
                      f"xy=({info['xy'][0]:6.2f},{info['xy'][1]:6.2f})  "
                      f"E={self.energy:5.2f}  spikes={int(counts.sum()):6d}  "
                      f"drive=({drive[0]:5.2f},{drive[1]:5.2f})  "
                      f"DN L/R={dn['dn_left']:5.1f}/{dn['dn_right']:5.1f}Hz")

            if self.arena.out_of_bounds(info["xy"]):
                self.ended = "left the arena"
                break
            if self.energy <= r.starve_threshold:
                self.ended = "starved"
                break
        else:
            self.ended = "time"

        self.wall_s = time.time() - t_wall
        return self.telemetry.frame()

    def save_video(self, path) -> bool:
        if self.camera is None:
            return False
        self.camera.save_video(path)
        return True
