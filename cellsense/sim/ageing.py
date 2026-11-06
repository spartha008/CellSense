"""Cycle-level capacity history for every cell, used by the forecast.

A real BMS logs one capacity estimate per cell per charge cycle. Here we generate that log with
the standard empirical fade shape Q(c) = Q0 * (1 - a*sqrt(c) - b*c) (SEI growth + linear wear), plus
an accelerating term for cells that are degrading faster than the pack. The shape is checked
against NASA PCoE and Oxford ageing data in :mod:`cellsense.validation.datasets`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FadeModel:
    a: float = 9.0e-4  # sqrt-of-cycles term
    b: float = 1.5e-5  # linear term
    spread: float = 0.08  # relative cell-to-cell spread of a and b
    estimate_noise: float = 0.0009  # 1-sigma noise of a per-cycle capacity estimate (relative)

    def curve(self, c, a=None, b=None):
        a = self.a if a is None else a
        b = self.b if b is None else b
        c = np.asarray(c, dtype=float)
        return 1.0 - a * np.sqrt(c) - b * c


@dataclass
class CapacityHistory:
    cycles: np.ndarray  # (K,)
    capacity: np.ndarray  # (K, N) Ah, estimated
    now_cycle: int


def capacity_history(cap_now: np.ndarray, now_cycle: int = 320, step: int = 4, seed: int = 0,
                     model: FadeModel = FadeModel(), accel_power: float = 1.7) -> CapacityHistory:
    """Back-cast a plausible per-cycle capacity log that ends at ``cap_now``.

    Cells more than 0.8% below the pack median are treated as degrading faster: their extra deficit
    grows as (c/now)^accel_power. Everything else ages at the pack rate.
    """
    rng = np.random.default_rng(seed)
    cap_now = np.asarray(cap_now, dtype=float)
    n = len(cap_now)
    cycles = np.arange(0, now_cycle + 1, step)
    a = model.a * (1 + model.spread * rng.standard_normal(n))
    b = model.b * (1 + model.spread * rng.standard_normal(n))
    base = model.curve(cycles[:, None], a[None, :], b[None, :])  # (K, N)
    base_now = model.curve(now_cycle, a, b)
    med = np.median(cap_now)
    excess = np.clip(1 - cap_now / med - 0.008, 0, None)  # extra deficit beyond normal spread
    extra = excess[None, :] * (cycles[:, None] / now_cycle) ** accel_power
    shape = base / base_now[None, :] * (1 + excess[None, :] - extra)
    cap = cap_now[None, :] * shape
    cap *= 1 + model.estimate_noise * rng.standard_normal(cap.shape)
    return CapacityHistory(cycles, cap, now_cycle)
