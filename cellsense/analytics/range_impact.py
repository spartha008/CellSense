"""Translate: turn per-cell charge into kilometres of range."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PACK, PackConfig


@dataclass
class RangeImpact:
    remaining_ah: np.ndarray  # (N,) charge left in each cell
    cost_km: np.ndarray  # (N,) range each cell costs vs the median cell
    lost_km: float  # range lost to the limiting cell
    limiting_cell: int  # 1-based
    usable_km: float  # rated range minus what is lost to imbalance


def range_impact(soc: np.ndarray, capacity_ah: np.ndarray, cfg: PackConfig = PACK) -> RangeImpact:
    """km lost ~ (median charge - weakest charge) / nominal charge x rated range.

    In a series pack the cell with the least charge left ends the discharge for everyone, so every
    amp-hour it is short of the median cell is range the driver cannot use.
    """
    q = np.asarray(soc, dtype=float) * np.asarray(capacity_ah, dtype=float)
    cost = np.clip(np.median(q) - q, 0, None) / cfg.capacity_ah * cfg.rated_range_km
    i = int(np.argmax(cost))
    return RangeImpact(q, cost, float(cost[i]), i + 1, float(cfg.rated_range_km - cost[i]))
