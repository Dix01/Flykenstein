"""The closed loop: world -> senses -> connectome -> descending neurons -> body.

One control tick is: read the body's sensors, convert them to Poisson drive on
real sensory neurons, integrate the whole connectome, apply plasticity at the
mushroom body, measure the descending neurons, turn that into a walking drive,
and step the physics.

Four things run alongside the fast synaptic dynamics:

* **Hunger** gates sensory gain, the way a real fly's feeding state changes how
  sensitive its sugar neurons are. It is a slow variable, not a memory.
* **Neuromodulation** - the monoamine field - scales synaptic transmission.
* **Learning** at the KC->MBON synapse, driven by dopamine the world delivers:
  PAM neurons on food, PPL1 on hazards, which is the division of labour in the
  real animal.
* **The compass**, an activity bump around the EPG ring at the fly's heading.

Nothing is trained by gradient descent, and nothing is rewarded by an optimiser.
The only thing that changes is a set of synapses, by a local rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..brain.compass import Compass
from ..brain.lif import Brain, LIFParams
from ..brain.plasticity import MushroomBody, PlasticityParams
from ..brain.ports import Ports
from ..brain.retina import Retina
from .arena import FlykensteinArena
from .rules import WorldRules
from .state import Life


@dataclass
class Telemetry:
    rows: list = field(default_factory=list)

    def log(self, **kw):
        self.rows.append(kw)

    def frame(self):
        import pandas as pd
        return pd.DataFrame(self.rows)


class Release:
    """A fly, a brain that remembers, a world, and a run."""

    def __init__(self, rules: WorldRules | None = None, vision: bool = False,
                 render: bool = True, brain: Brain | None = None,
                 life: Life | None = None):
        from flygym import Fly, Camera
        from flygym.examples.locomotion import HybridTurningController

        self.rules = rules or WorldRules()
        r = self.rules

        self.brain = brain or Brain(params=LIFParams(mod_gain=r.mod_gain), seed=r.seed)
        self.ports = Ports(self.brain)
        self.brain.exempt[self.ports.exempt_sensory()] = True
        self.mb = MushroomBody(self.brain, self.ports,
                               PlasticityParams(eta=r.plasticity_eta))
        self.compass = Compass(self.brain)

        # Reward and punishment are carried by different dopaminergic families.
        pt = self.brain.meta["primary_type"].astype(str)
        self.pam = self.brain.meta.index[pt.str.startswith("PAM")].to_numpy(np.int32)
        self.ppl1 = self.brain.meta.index[pt.str.startswith("PPL1")].to_numpy(np.int32)

        # A life the fly carries into this release.
        self.life = life or Life(r.life_path)
        self.carried = self.life.load(self.mb) if r.carry_memory else False
        self.drives = self.life.drives
        self.drives.energy = r.energy_start if not self.carried else self.drives.energy
        self.life.releases += 1

        self.arena = FlykensteinArena(r)
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
                                           timestep=1e-4, seed=r.seed)
        self.camera = cams[0] if cams else None

        self.vision = vision
        self.retina = None
        if vision:
            self.retina = Retina(self.brain)
            self.retina.fit(self._ommatidia_layout())

        self.telemetry = Telemetry()
        self._dn_ema: dict[str, float] = {}
        self.feeds = 0
        self.hazard_hits = 0
        self.ended = ""

    # ---- retinotopy -----------------------------------------------------
    def _ommatidia_layout(self) -> dict[str, np.ndarray]:
        """Centroid of every ommatidium in the rendered eye image."""
        idmap = np.asarray(self.fly.retina.ommatidia_id_map)
        n = self.fly.retina.num_ommatidia_per_eye
        rows, cols = np.indices(idmap.shape)
        xy = np.zeros((n, 2))
        for k in range(n):
            m = idmap == (k + 1)
            xy[k] = (cols[m].mean(), rows[m].mean()) if m.any() else (0.0, 0.0)
        return {"left": xy, "right": xy.copy()}

    # ---- encoding: world -> sensory neurons -----------------------------
    def encode(self, obs) -> tuple[np.ndarray, dict, dict]:
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

        # Hunger opens the food senses and dulls the aversive ones.
        stim = {k: float(np.clip(v * self.drives.gain(k), 0, 1)) for k, v in stim.items()}

        rates = self.ports.drive(stim, base_hz=r.base_hz)

        # Vision, retinotopically.
        if self.vision and "vision" in obs:
            v = np.asarray(obs["vision"], dtype=float)      # (2, ommatidia, 2)
            peak = max(v.max(), 1e-9)
            for k, eye in enumerate(("left", "right")):
                idx, drive = self.retina.drive(eye, v[k].mean(axis=-1) / peak)
                rates[idx] = np.maximum(rates[idx], drive * r.vision_gain * r.base_hz)

        # The compass bump, at the heading the body actually has.
        heading = float(np.arctan2(obs["fly_orientation"][1], obs["fly_orientation"][0]))
        rates[self.compass.idx] = np.maximum(
            rates[self.compass.idx], self.compass.drive(heading) * r.compass_hz)

        info = {"on_food": on_food, "on_hazard": on_haz, "d_food": d_food,
                "d_hazard": d_haz, "xy": xy, "heading": heading}
        return rates, stim, info

    # ---- decoding: descending neurons -> walking drive -------------------
    def _rate(self, name: str, counts: np.ndarray, window_ms: float) -> float:
        port = self.ports.motor[name]
        if len(port) == 0:
            return 0.0
        return float(counts[port.idx].sum() / len(port) / (window_ms / 1000.0))

    def _smooth(self, name: str, value: float) -> float:
        """Exponential moving average over `dn_window_ms`.

        These populations are tiny - two DNa neurons a side - so a single
        control tick reports either 0 Hz or one spike's worth. Averaging over a
        window turns that into the graded signal a real descending pathway
        would carry.
        """
        a = min(1.0, self.rules.control_ms / max(self.rules.dn_window_ms, 1e-6))
        cur = (1 - a) * self._dn_ema.get(name, 0.0) + a * value
        self._dn_ema[name] = cur
        return cur

    def decode(self, counts: np.ndarray, window_ms: float) -> tuple[np.ndarray, dict]:
        r = self.rules
        left = self._smooth("turn_left", self._rate("turn_left", counts, window_ms))
        right = self._smooth("turn_right", self._rate("turn_right", counts, window_ms))
        stop = self._smooth("stop", self._rate("stop", counts, window_ms))
        back = self._smooth("backward", self._rate("backward", counts, window_ms))
        prob = self._rate("proboscis", counts, window_ms)

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
        """Run to completion and return the telemetry as a frame."""
        for _ in self.iter_run(verbose=verbose):
            pass
        return self.telemetry.frame()

    def iter_run(self, verbose: bool = False):
        """Run the release, yielding each control tick's telemetry as it happens.

        Same loop as `run`, exposed a tick at a time so a caller can draw a
        display while the simulation is still computing.
        """
        r = self.rules
        obs, _ = self.sim.reset(seed=r.seed)
        n_ctrl = int(round(r.duration_s * 1000.0 / r.control_ms))
        brain_steps = int(round(r.control_ms / self.brain.p.dt))
        phys_steps = int(round(r.control_ms * 1e-3 / self.sim.timestep))
        dt_s = r.control_ms / 1000.0
        t_wall = time.time()
        was_food = was_haz = False

        for tick in range(n_ctrl):
            rates, stim, info = self.encode(obs)

            # The world's teaching signal, on the pathways that carry it in a
            # real fly: PAM for reward, PPL1 for punishment.
            if info["on_food"] and r.reward_dan_hz:
                rates[self.pam] = np.maximum(rates[self.pam], r.reward_dan_hz)
            if info["on_hazard"] and r.punish_dan_hz:
                rates[self.ppl1] = np.maximum(rates[self.ppl1], r.punish_dan_hz)

            counts = np.zeros(self.brain.n, dtype=np.float32)
            for _ in range(brain_steps):
                counts[self.brain.step(poisson=rates)] += 1

            plast = self.mb.update(counts, r.control_ms) if r.learning else {
                "da_max": 0.0, "elig_mean": 0.0, "depressed": 0.0}
            drive, dn = self.decode(counts, r.control_ms)

            for _ in range(phys_steps):
                obs, _, term, trunc, _ = self.sim.step(drive)
                if self.camera is not None:
                    self.sim.render()

            # --- our physiology ---
            self.drives.energy -= r.energy_drain_per_s * dt_s
            if info["on_food"] and dn["mn9"] > 5.0:
                self.drives.energy += r.energy_per_feed_s * dt_s
                self.feeds += 1
            if info["on_hazard"]:
                self.drives.energy -= r.hazard_cost_per_s * dt_s
                self.hazard_hits += 1
            self.drives.energy = float(np.clip(self.drives.energy, -1.0, 2.0))
            novelty = 0.2 if (info["on_hazard"] and not was_haz) else 0.0
            self.drives.tick(dt_s, novelty)

            if info["on_food"] and not was_food:
                self.life.note("fed", t=self.drives.age_s,
                               x=float(info["xy"][0]), y=float(info["xy"][1]))
            if info["on_hazard"] and not was_haz:
                self.life.note("hazard", t=self.drives.age_s,
                               x=float(info["xy"][0]), y=float(info["xy"][1]))
            was_food, was_haz = info["on_food"], info["on_hazard"]

            bump, strength = self.compass.read(counts, dt_s)
            self.telemetry.log(
                t_s=(tick + 1) * dt_s,
                x=float(info["xy"][0]), y=float(info["xy"][1]),
                heading=info["heading"], bump=bump, bump_strength=strength,
                energy=self.drives.energy, hunger=self.drives.hunger,
                arousal=self.drives.arousal,
                spikes=int(counts.sum()), active=int((counts > 0).sum()),
                drive_l=float(drive[0]), drive_r=float(drive[1]),
                kc=float(counts[self.ports.readout["kenyon"].idx].sum()),
                kc_active=int((counts[self.mb.kc] > 0).sum()),
                mbon=float(counts[self.ports.readout["mbon"].idx].sum()),
                cx=float(counts[self.ports.readout["central_complex"].idx].sum()),
                modulation=float(self.brain.modulation().mean()),
                **{k: float(v) for k, v in dn.items()},
                **{k: float(v) for k, v in plast.items()},
                **{f"in_{k}": float(v) for k, v in stim.items()},
                on_food=bool(info["on_food"]), on_hazard=bool(info["on_hazard"]),
            )

            yield self.telemetry.rows[-1]

            if verbose and tick % max(1, n_ctrl // 20) == 0:
                print(f"  t={self.telemetry.rows[-1]['t_s']:5.2f}s  "
                      f"xy=({info['xy'][0]:6.2f},{info['xy'][1]:6.2f})  "
                      f"E={self.drives.energy:5.2f} H={self.drives.hunger:4.2f}  "
                      f"KC={self.telemetry.rows[-1]['kc_active']:4d}  "
                      f"mem={plast['depressed']*100:5.2f}%  "
                      f"drive=({drive[0]:5.2f},{drive[1]:5.2f})")

            if self.arena.out_of_bounds(info["xy"]):
                self.ended = "left the arena"
                break
            if self.drives.energy <= r.starve_threshold:
                self.ended = "starved"
                break
        else:
            self.ended = "time"

        self.wall_s = time.time() - t_wall
        self.life.save(self.mb)

    def save_video(self, path) -> bool:
        if self.camera is None:
            return False
        self.camera.save_video(path)
        return True
