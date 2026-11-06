"""Charge-drift estimators: separate capacity fade, self-discharge and sudden charge steps.

All inputs are per-cell OCV offsets vs the pack median, in %SoC.

* Capacity fade: while driving, a weaker cell's SoC falls faster, so its offset drifts in
  proportion to the charge drawn: d(offset)/dAh = -(1/C_cell - 1/C_median).
* Self-discharge: while parked no charge is drawn, yet a leaking cell's offset keeps falling.
* A sudden step (a charge imbalance appearing at once) fits a step far better than a slope;
  in that case the slope is only measured after the step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FadeEstimate:
    slope: np.ndarray  # %SoC per Ah (negative = losing ground as the pack discharges)
    se: np.ndarray  # standard error of the slope
    step: np.ndarray  # size of a detected sudden step, %SoC (nan if none)
    ah_span: float


def _linfit(x: np.ndarray, Y: np.ndarray):
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        nan = np.full(Y.shape[1], np.nan)
        return nan, nan, nan
    slope = xc @ (Y - Y.mean(0)) / sxx
    resid = Y - Y.mean(0) - np.outer(xc, slope)
    sse = (resid**2).sum(0)
    se = np.sqrt(sse / max(len(x) - 2, 1) / sxx)
    return slope, se, sse


def _best_step(Y: np.ndarray, margin: int):
    """Best single breakpoint (piecewise-constant fit) for every column at once."""
    k = len(Y)
    c1, c2 = np.cumsum(Y, 0), np.cumsum(Y**2, 0)
    b = np.arange(margin, k - margin)  # first segment = rows [0, b)
    s1, q1 = c1[b - 1], c2[b - 1]
    s2, q2 = c1[-1] - s1, c2[-1] - q1
    nb, na = b[:, None].astype(float), (k - b)[:, None].astype(float)
    sse = (q1 - s1**2 / nb) + (q2 - s2**2 / na)
    j = np.argmin(sse, 0)
    cols = np.arange(Y.shape[1])
    size = s2[j, cols] / na[j, 0] - s1[j, cols] / nb[j, 0]
    return sse[j, cols], b[j], size


def fade_slope(ah: np.ndarray, Y: np.ndarray, min_points: int = 60) -> FadeEstimate:
    n = Y.shape[1]
    nan = np.full(n, np.nan)
    if len(ah) < min_points:
        return FadeEstimate(nan, nan, nan, 0.0)
    slope, se, sse_lin = _linfit(ah, Y)
    step = nan.copy()
    sse_step, brk, size = _best_step(Y, margin=max(20, len(ah) // 10))
    is_step = (sse_step < 0.6 * sse_lin) & (np.abs(size) > 0.4)
    for i in np.nonzero(is_step)[0]:  # re-measure the slope only after the step
        step[i] = size[i]
        tail = slice(brk[i], None)
        if np.ptp(ah[tail]) >= 8 and len(ah[tail]) >= min_points:
            s, e, _ = _linfit(ah[tail], Y[tail, i:i + 1])
            slope[i], se[i] = s[0], e[0]
        else:
            slope[i], se[i] = np.nan, np.nan
    return FadeEstimate(slope, se, step, float(np.ptp(ah)))


@dataclass
class ParkedDrift:
    rate: np.ndarray  # %SoC per hour while parked
    se: np.ndarray
    parked_s: float


def parked_drift(park_id: np.ndarray, hours: np.ndarray, Y: np.ndarray, min_points: int = 60) -> ParkedDrift:
    """Pooled within-park slope of offset vs time.

    Each parked period is fitted on its own; a period where a sudden step explains the data far better
    than a steady slope (e.g. charge removed at once) is left out for that cell, so a step is never
    mistaken for a leak.
    """
    n = Y.shape[1]
    nan = np.full(n, np.nan)
    if len(hours) < min_points:
        return ParkedDrift(nan, nan, float(len(hours)))
    num, sxx_tot, sse_tot, dof = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for pid in np.unique(park_id):
        m = park_id == pid
        if m.sum() < 30:
            continue
        x, Yp = hours[m], Y[m]
        xc = x - x.mean()
        sxx = float(xc @ xc)
        if sxx <= 0:
            continue
        slope = xc @ (Yp - Yp.mean(0)) / sxx
        resid = Yp - Yp.mean(0) - np.outer(xc, slope)
        sse_lin = (resid**2).sum(0)
        sse_step, _, size = _best_step(Yp, margin=max(10, len(x) // 8))
        ok = ~((sse_step < 0.6 * sse_lin) & (np.abs(size) > 0.3))
        num += np.where(ok, slope * sxx, 0.0)
        sxx_tot += np.where(ok, sxx, 0.0)
        sse_tot += np.where(ok, sse_lin, 0.0)
        dof += np.where(ok, len(x) - 2, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(sxx_tot > 0, num / sxx_tot, np.nan)
        se = np.where(sxx_tot > 0, np.sqrt(sse_tot / np.maximum(dof, 1) / sxx_tot), np.nan)
    return ParkedDrift(rate, se, float(len(hours)))
