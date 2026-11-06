"""Forecast: which cell will limit the pack next, and when it crosses the replace threshold."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config import PACK, THRESHOLDS, PackConfig, Thresholds
from ..sim.ageing import CapacityHistory
from ..sim.faults import Fault


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
    step_km: float = 0.0  # permanent shift from injected faults (moves the trend up from today)
    recoverable_km: float = 0.0  # range injected faults cost today that balancing or charging gives back
    fault_label: str = ""  # what was injected, for the chart and the ranking


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


@dataclass
class FaultEffect:
    """What one injected fault does to its cell's range-loss forecast."""
    cell: int
    label: str
    permanent_km: float = 0.0
    recoverable_km: float = 0.0


def fault_effect(fault: Fault, elapsed_s: float, params, r_before_ohm: float, cap_ratio: float = 1.0,
                 cfg: PackConfig = PACK, th: Thresholds = THRESHOLDS) -> FaultEffect:
    """Translate an injected fault into kilometres of range, like the forecast's y-axis.

    capacity fade       permanent: the cell holds ``severity`` less charge from today on.
    resistance growth   permanent: under a sustained ``load_current_a`` the cell reaches ``v_min`` earlier, so
                        the charge between the two cut-off points is lost (read off the cell's OCV curve).
    charge imbalance    recoverable: the SoC offset costs range now; balancing gives it back.
    self-discharge      recoverable: the charge leaked since injection; charging gives it back, then it leaks again.
    ``cap_ratio`` is the cell's capacity over the pack median, so km match the forecast's own scale.
    """
    s, km = fault.severity, cfg.rated_range_km
    if fault.kind == "capacity_fade":
        return FaultEffect(fault.cell, f"-{s * 100:.1f}% capacity", permanent_km=s * cap_ratio * km)
    if fault.kind == "resistance_growth":
        def cutoff_soc(r):
            return float(params.soc_at(cfg.v_min + th.load_current_a * r))
        lost = max(0.0, cutoff_soc(r_before_ohm * (1 + s)) - cutoff_soc(r_before_ohm))
        return FaultEffect(fault.cell, f"+{s * 100:.0f}% resistance", permanent_km=lost * cap_ratio * km)
    if fault.kind == "charge_imbalance":
        return FaultEffect(fault.cell, f"{s * 100:.1f}% SoC offset", recoverable_km=s * cap_ratio * km)
    leaked = min(s * max(elapsed_s, 0.0) / 3600.0, 1.0)
    return FaultEffect(fault.cell, f"{s * 100:.0f}%/h leak", recoverable_km=leaked * cap_ratio * km)


def apply_effects(forecasts: list[CellForecast], effects: list[FaultEffect], th: Thresholds = THRESHOLDS) -> list[CellForecast]:
    """Fold injected faults into the forecasts: permanent km shift the trend up from today, recoverable km
    only add to today's cost. Cells without an injected fault are returned unchanged."""
    by_cell: dict[int, list[FaultEffect]] = {}
    for e in effects:
        by_cell.setdefault(e.cell, []).append(e)
    out = []
    for f in forecasts:
        es = by_cell.get(f.cell)
        if not es:
            out.append(f)
            continue
        step = sum(e.permanent_km for e in es)
        rec = sum(e.recoverable_km for e in es)
        fy = f.fy + step
        hit = np.nonzero(fy >= th.replace_km)[0]
        cross = float(f.fx[hit[0]]) if len(hit) else None
        out.append(replace(f, now_km=f.now_km + step + rec, km_6mo=f.km_6mo + step, fy=fy, lo=f.lo + step, hi=f.hi + step,
                           cross_cycle=cross, months_to_threshold=None if cross is None else (cross - float(f.fx[0])) / th.cycles_per_month,
                           step_km=step, recoverable_km=rec, fault_label=" + ".join(dict.fromkeys(e.label for e in es))))
    return out


def ranking(forecasts: list[CellForecast], top: int = 3) -> list[CellForecast]:
    return sorted(forecasts, key=lambda f: f.km_6mo, reverse=True)[:top]
