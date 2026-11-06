"""Forecast: which cell will limit the pack next, and when it crosses the replace threshold."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PACK, THRESHOLDS, PackConfig, Thresholds
from ..sim.ageing import CapacityHistory


@dataclass
class CellForecast:
    cell: int
    now_km: float
    km_6mo: float
    cross_cycle: float | None  # cycle at which the fitted trend reaches the replace threshold
    months_to_threshold: float | None
    cycles: np.ndarray  # history
    km: np.ndarray
    fx: np.ndarray  # forecast
    fy: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def forecast(hist: CapacityHistory, cfg: PackConfig = PACK, th: Thresholds = THRESHOLDS, horizon: int = 240,
             fit_window: int = 200, cells: list[int] | None = None) -> list[CellForecast]:
    """Fit deficit(cycle) = a + b*cycle + c*cycle^2 per cell on recent history and project it forward."""
    cyc = np.asarray(hist.cycles, dtype=float)
    cap = np.asarray(hist.capacity, dtype=float)
    med = np.median(cap, axis=1, keepdims=True)
    km_all = (med - cap) / med * cfg.rated_range_km  # range this cell would cost if it limited the pack
    now = float(hist.now_cycle)
    step = float(np.median(np.diff(cyc))) if len(cyc) > 1 else 4.0
    fx = np.arange(now, now + horizon + step / 2, step)
    six = now + 6 * th.cycles_per_month
    use = cyc >= now - fit_window
    out = []
    for i in (range(cap.shape[1]) if cells is None else [c - 1 for c in cells]):
        y = km_all[:, i]
        coef = np.polyfit(cyc[use], y[use], 2)
        if coef[0] < 0:  # do not let a noisy fit bend the trend back down; fall back to linear
            coef = np.r_[0.0, np.polyfit(cyc[use], y[use], 1)]
        resid = y[use] - np.polyval(coef, cyc[use])
        sd = float(np.sqrt(np.sum(resid**2) / max(1, use.sum() - 3)))
        fy = np.polyval(coef, fx)
        widen = 2 * sd * (1 + (fx - now) / 80.0)
        hit = np.nonzero(fy >= th.replace_km)[0]
        cross = float(fx[hit[0]]) if len(hit) else None
        out.append(CellForecast(
            cell=i + 1, now_km=float(np.polyval(coef, now)), km_6mo=float(np.polyval(coef, six)),
            cross_cycle=cross, months_to_threshold=None if cross is None else (cross - now) / th.cycles_per_month,
            cycles=cyc, km=y, fx=fx, fy=fy, lo=fy - widen, hi=fy + widen,
        ))
    return out


def ranking(forecasts: list[CellForecast], top: int = 3) -> list[CellForecast]:
    return sorted(forecasts, key=lambda f: f.km_6mo, reverse=True)[:top]
