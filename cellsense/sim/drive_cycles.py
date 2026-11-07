"""Synthetic pack-current profiles (city, highway, mixed) built from driving modes.

Positive current = discharge. Levels are for a 150 Ah, ~350 V pack.
"""
from __future__ import annotations

import numpy as np

# mode: (current range in A, duration range in s)
_MODES = {
    "city": {
        "idle": ((1.0, 4.0), (10, 45)), "cruise": ((18.0, 50.0), (20, 70)), "accel": ((110.0, 200.0), (6, 16)),
        "regen": ((-90.0, -40.0), (5, 14)), "park": ((0.3, 1.0), (120, 300)),
    },
    "highway": {
        "idle": ((1.0, 4.0), (5, 15)), "cruise": ((65.0, 120.0), (60, 180)), "accel": ((170.0, 260.0), (8, 22)),
        "regen": ((-80.0, -40.0), (5, 12)), "park": ((0.3, 1.0), (120, 300)),
    },
}
_NEXT = {
    "city": {"idle": ["accel", "accel", "cruise"], "cruise": ["accel", "regen", "cruise", "idle", "idle"], "accel": ["cruise", "regen"], "regen": ["idle", "cruise"], "park": ["accel"]},
    "highway": {"idle": ["accel"], "cruise": ["accel", "cruise", "regen", "cruise"], "accel": ["cruise", "cruise", "regen"], "regen": ["cruise", "idle"], "park": ["accel"]},
}
PROFILES = ("city", "highway", "mixed")


class LiveDriveCycle:
    """Endless current source for the live dashboard: call :meth:`next` once per second."""

    def __init__(self, kind: str = "mixed", seed: int = 0, park_prob: float = 0.04):
        if kind not in PROFILES:
            raise ValueError(f"kind must be one of {PROFILES}")
        self.kind, self.rng, self.park_prob = kind, np.random.default_rng(seed), park_prob
        self.style = "city" if kind != "highway" else "highway"
        self.mode, self.level, self.left = "idle", 2.0, 0

    def park(self, seconds: int = 240):
        """Force a parked period (used by the fault lab to show self-discharge)."""
        self.mode, self.level, self.left = "park", 0.6, int(seconds)

    def _pick(self):
        if self.kind == "mixed" and self.rng.random() < 0.08:
            self.style = "highway" if self.style == "city" else "city"
        if self.mode != "park" and self.rng.random() < self.park_prob:
            mode = "park"
        else:
            opts = _NEXT[self.style][self.mode]
            mode = opts[self.rng.integers(len(opts))]
        (lo, hi), (dlo, dhi) = _MODES[self.style][mode]
        self.mode, self.level = mode, lo + (hi - lo) * self.rng.random()
        self.left = int(dlo + (dhi - dlo) * self.rng.random())

    def next(self) -> float:
        if self.left <= 0:
            self._pick()
        self.left -= 1
        jitter = 0.4 if self.mode == "park" else 1.2
        return float(self.level + jitter * self.rng.standard_normal())


def drive_cycle(kind: str = "mixed", duration_s: int = 1800, seed: int = 0, park_prob: float = 0.04) -> np.ndarray:
    src = LiveDriveCycle(kind, seed, park_prob)
    return np.array([src.next() for _ in range(int(duration_s))])
