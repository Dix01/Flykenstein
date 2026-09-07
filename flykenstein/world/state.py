"""Internal state, and the persistence that makes a sequence of runs one life.

Two different things are called memory here and they should not be confused.

*Memory in the animal* is the KC->MBON weight matrix. It changes because of
what happened, it changes what happens next, and nothing reads it out except
the rest of the brain. That is the real thing, and it lives in plasticity.py.

*A record about the animal* is the episode log below: timestamped notes on what
it met and when. It is for us. The fly cannot read it. It is kept because
without it you cannot tell whether the weights learned anything, but it should
never be mistaken for the animal remembering.

Internal state - hunger, arousal - sits in between. It is not a memory of an
event, it is a slow variable that biases how the brain responds, the way a real
fly's feeding state gates the sensitivity of its sugar neurons.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class Drives:
    """Slow internal variables that modulate sensing, as feeding state does."""
    energy: float = 1.0
    arousal: float = 0.0        # rises with novelty and threat, decays slowly
    age_s: float = 0.0

    # how strongly hunger opens and closes the senses
    hunger_sugar_gain: float = 1.6      # a hungry fly's sugar neurons are more sensitive
    hunger_odor_gain: float = 1.0
    hunger_bitter_relief: float = 0.7   # hunger makes bitter less repellent
    arousal_tau_s: float = 8.0

    @property
    def hunger(self) -> float:
        return float(np.clip(1.0 - self.energy, 0.0, 1.0))

    def gain(self, port: str) -> float:
        h = self.hunger
        if port == "sugar":
            return 1.0 + self.hunger_sugar_gain * h
        if port.startswith("odor_good"):
            return 1.0 + self.hunger_odor_gain * h
        if port == "bitter":
            return max(0.0, 1.0 - self.hunger_bitter_relief * h)
        return 1.0

    def tick(self, dt_s: float, novelty: float = 0.0) -> None:
        self.age_s += dt_s
        self.arousal += novelty - self.arousal * dt_s / self.arousal_tau_s
        self.arousal = float(np.clip(self.arousal, 0.0, 1.0))


class Life:
    """Everything that carries over from one release to the next."""

    def __init__(self, path: str | Path = "life"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.drives = Drives()
        self.episodes: list[dict] = []
        self.releases = 0

    # ---- episodic record (for us, not for the fly) ----------------------
    def note(self, kind: str, **fields) -> None:
        self.episodes.append({"release": self.releases, "kind": kind,
                              "wall": time.time(), **fields})

    # ---- persistence ----------------------------------------------------
    def save(self, mushroom_body=None) -> None:
        state = {"drives": asdict(self.drives), "releases": self.releases}
        (self.path / "drives.json").write_text(json.dumps(state, indent=2))
        with open(self.path / "episodes.jsonl", "w") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")
        if mushroom_body is not None:
            mushroom_body.save(self.path / "memory.npz")

    def load(self, mushroom_body=None) -> bool:
        f = self.path / "drives.json"
        if not f.exists():
            return False
        state = json.loads(f.read_text())
        self.drives = Drives(**state["drives"])
        self.releases = state.get("releases", 0)
        ep = self.path / "episodes.jsonl"
        if ep.exists():
            self.episodes = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
        mem = self.path / "memory.npz"
        if mushroom_body is not None and mem.exists():
            mushroom_body.load(mem)
        return True

    def summary(self) -> str:
        meals = sum(1 for e in self.episodes if e["kind"] == "fed")
        harm = sum(1 for e in self.episodes if e["kind"] == "hazard")
        return (f"{self.releases} releases, {self.drives.age_s:.1f}s lived, "
                f"{meals} meals, {harm} hazard encounters, "
                f"energy {self.drives.energy:.2f}")
