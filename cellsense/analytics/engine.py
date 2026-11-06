"""Streaming CellSense engine: feed one sample per second, read a snapshot whenever you like.

Per sample it compares every cell with the pack median (robust z-score), estimates internal
resistance from voltage jumps at current steps (dV/dI), tracks offsets at rest and under load, keeps
a coulomb-counted SoC per cell, and runs a conventional threshold-BMS check alongside for comparison.
Every few seconds it scores cells with the Isolation Forest, diagnoses them, and prices the
imbalance in km.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..config import PACK, THRESHOLDS, PackConfig, Thresholds
from ..sim.ecm import CellParams
from .anomaly import AnomalyModel, feature_matrix
from .diagnosis import CellFinding, diagnose
from .drift import fade_slope, parked_drift
from .range_impact import RangeImpact, range_impact


class _Ring:
    """Fixed-size ring buffer of rows."""

    def __init__(self, size: int, width: int):
        self.buf = np.full((size, width), np.nan)
        self.i = 0
        self.n = 0
        self._sum = np.zeros(width)

    def push(self, row: np.ndarray, track_sum: bool = False):
        if track_sum:
            if self.n == len(self.buf):
                self._sum -= self.buf[self.i]
            self._sum += row
        self.buf[self.i] = row
        self.i = (self.i + 1) % len(self.buf)
        self.n = min(self.n + 1, len(self.buf))

    def mean(self) -> np.ndarray:
        return self._sum / max(self.n, 1)

    def values(self) -> np.ndarray:
        if self.n < len(self.buf):
            return self.buf[: self.n]
        return np.roll(self.buf, -self.i, axis=0)

    def forget(self, cols: np.ndarray, keep_last: int):
        """Blank every row except the newest ``keep_last`` for the columns in boolean mask ``cols``."""
        size = len(self.buf)
        keep = {(self.i - 1 - k) % size for k in range(keep_last)}
        rows = [r for r in range(size) if r not in keep]
        block = self.buf[rows]
        block[:, cols] = np.nan
        self.buf[rows] = block


@dataclass
class Snapshot:
    t: float
    current: float
    pack_voltage: float
    pack_soc: float
    spread_mv: float
    dev90_mv: np.ndarray
    zroll: np.ndarray
    if_score: np.ndarray
    features: dict
    findings: list[CellFinding]
    status: np.ndarray  # per-cell status string ("ok" when no finding)
    impact: RangeImpact
    first_flag_t: np.ndarray
    first_z_t: np.ndarray
    bms_first_t: np.ndarray
    hist_t: np.ndarray
    hist_i: np.ndarray
    hist_v: np.ndarray  # (H, N)
    hist_z: np.ndarray  # (H, N)
    hist_med: np.ndarray
    hist_p05: np.ndarray
    hist_p95: np.ndarray

    def finding(self, cell: int) -> CellFinding | None:
        return next((f for f in self.findings if f.cell == cell), None)


class CellSenseEngine:
    def __init__(self, cfg: PackConfig = PACK, th: Thresholds = THRESHOLDS, params: CellParams | None = None,
                 anomaly: AnomalyModel | None = None, capacity_ah: np.ndarray | None = None,
                 history_s: int = 900, eval_every_s: float = 5.0):
        self.cfg, self.th = cfg, th
        self.p = params or CellParams.load()
        self.anomaly = anomaly if anomaly is not None else AnomalyModel.load()
        n = self.n = cfg.n_series
        self.cap_est = np.full(n, cfg.capacity_ah) if capacity_ah is None else np.asarray(capacity_ah, dtype=float).copy()
        self.eval_every_s = eval_every_s
        self._z = _Ring(th.z_window_s, n)
        self.persist = np.zeros(n, dtype=int)
        self.first_z_t = np.full(n, np.nan)
        self.last_z_t = np.full(n, -np.inf)
        self.first_flag_t = np.full(n, np.nan)
        self.bms_first_t = np.full(n, np.nan)
        self._r = _Ring(11, n)  # last 11 current steps: the median rejects outliers; a real change is followed within 2 steps
        self._r_cache = np.full(n, np.nan)
        self._v1_nom = 0.0  # polarisation voltage of a nominal cell, from the current history
        self.rest_off = np.full(n, np.nan)
        self.load_off = np.full(n, np.nan)
        # raw readings for the charge-drift estimators; compensation is applied at evaluation time with
        # the latest resistance estimates, so early (still-settling) estimates do not leave a bias
        self._drive = _Ring(3000, 2 * n + 4)  # [Ah out, I, v1_nom, pack SoC, dev x n, R at the time x n] while driving
        self._park = _Ring(3000, 2 * n + 5)  # [park id, hours, I, v1_nom, pack SoC, dev x n, R x n] while parked
        self._park_id = 0
        self._park_run = 0.0
        self.ah_out = 0.0
        self._since_high = 1e9
        self._rest_run = 0.0
        self.soc_est: np.ndarray | None = None
        self._prev: tuple[float, np.ndarray] | None = None
        self._ht = _Ring(history_s, 2)
        self._hv = _Ring(history_s, n)
        self._hz = _Ring(history_s, n)
        self.t = 0.0
        self.current = 0.0
        self.v = np.full(n, np.nan)
        self.samples = 0
        self._last_eval = -np.inf
        self._cache: tuple | None = None

    # ------------------------------------------------------------------ per-sample update
    def update(self, t: float, current: float, voltages: np.ndarray, dt: float = 1.0):
        th, cfg = self.th, self.cfg
        V = np.asarray(voltages, dtype=float)
        I = float(current)
        med = float(np.median(V))
        dev = V - med
        mad = max(th.mad_floor_v, 1.4826 * float(np.median(np.abs(dev))))
        z = dev / mad
        self._z.push(z, track_sum=True)
        zroll = self._z.mean()
        alert = zroll <= -th.z_alert  # a cell sitting below the pack (higher voltage is not a weakness)
        self.persist = np.where(alert, self.persist + 1, 0)
        held = self.persist >= th.persistence_s
        self.last_z_t[held] = t
        new = held & np.isnan(self.first_z_t)
        self.first_z_t[new] = t
        self._flag(new, t)

        if self._prev is not None:  # internal resistance from voltage response to a current step
            Ip, Vp = self._prev
            dI = I - Ip
            if abs(dI) >= th.step_di_a:
                self._r.push(-(V - Vp) / dI)
                self._follow_resistance_change()
                self._r_cache = np.nanmedian(self._r.values(), axis=0)

        # OCV offset of each cell vs the pack: undo the IR drop and the RC polarisation, both scaled by
        # the cell's measured resistance, so what is left is a difference in state of charge
        r = self._r_cache
        known = np.isfinite(r)
        rmed = float(np.median(r[known])) if known.any() else self.p.r0
        ratio = np.where(known, r / rmed, 1.0)
        a = np.exp(-dt / self.p.tau)
        self._v1_nom = self._v1_nom * a + I * self.p.r1 * (rmed / self.p.r0) * (1 - a)
        ocv_dev = dev + I * (np.where(known, r, rmed) - rmed) + (ratio - 1.0) * self._v1_nom

        rest = abs(I) < th.rest_current_a
        if abs(I) < 40:
            self.rest_off = np.where(np.isnan(self.rest_off), ocv_dev, 0.97 * self.rest_off + 0.03 * ocv_dev)
        if I >= th.load_current_a:
            self.load_off = np.where(np.isnan(self.load_off), dev, 0.9 * self.load_off + 0.1 * dev)

        # coulomb-counted SoC per cell, re-anchored to OCV after each minute of rest
        self.ah_out += I * dt / 3600.0
        ocv_cell = V + I * np.where(known, r, rmed) + ratio * self._v1_nom
        if self.soc_est is None:
            self.soc_est = self.p.soc_at(ocv_cell)
        else:
            self.soc_est = np.clip(self.soc_est - I * dt / 3600.0 / self.cap_est, 0, 1)
        self._rest_run = self._rest_run + dt if abs(I) < 2.0 else 0.0
        if self._rest_run >= 60 and (self._rest_run % 60) < dt:
            self.soc_est = self.p.soc_at(ocv_cell)
        elif abs(I) <= 60 and known.all():  # gentle OCV observer: pull coulomb counting toward the voltage-based SoC
            self.soc_est = self.soc_est + 0.01 * (self.p.soc_at(ocv_cell) - self.soc_est)

        # OCV offset in %SoC, logged for the charge-drift estimators (driving vs parked)
        self._since_high = 0.0 if abs(I) > 100 else self._since_high + dt
        parked = abs(I) < 2.0
        if parked and self._park_run == 0.0:
            self._park_id += 1
        self._park_run = self._park_run + dt if parked else 0.0
        if abs(I) <= 150:
            soc_med = float(np.median(self.soc_est))
            r_then = self._r_cache if self._r.n >= 9 else np.full(self.n, np.nan)  # only once the estimate has settled
            if parked and self._park_run >= 90:  # wait for polarisation to settle
                self._park.push(np.r_[self._park_id, t / 3600.0, I, self._v1_nom, soc_med, dev, r_then])
            elif not parked and self._since_high >= 5:
                self._drive.push(np.r_[self.ah_out, I, self._v1_nom, soc_med, dev, r_then])

        # conventional BMS: hard voltage window, plus a fixed rest-imbalance threshold
        alarm = (V < cfg.v_min) | (V > cfg.v_max) | (rest & (np.abs(dev) > th.bms_rest_dv))
        newb = alarm & np.isnan(self.bms_first_t)
        self.bms_first_t[newb] = t

        self._ht.push(np.array([t, I]))
        self._hv.push(V)
        self._hz.push(zroll)
        self._prev = (I, V)
        self.t, self.current, self.v = t, I, V
        self.samples += 1
        if t - self._last_eval >= self.eval_every_s:
            self.evaluate()

    def _follow_resistance_change(self):
        """The median of the last 11 steps rejects outliers, but after a real change (a fault) it only moves once
        6 new steps have arrived, which can take minutes of cruising. When a cell's last few steps all sit more than
        ``r_shift_frac`` away from its history, its resistance has changed: drop its older steps so the estimate
        follows the new level now. A single noisy step cannot do this; it takes ``r_shift_steps`` in a row."""
        k, frac = self.th.r_shift_steps, self.th.r_shift_frac
        vals = self._r.values()
        if len(vals) < k + 4:
            return
        with warnings.catch_warnings():  # a column already reset has no history yet: all-NaN is expected
            warnings.simplefilter("ignore", RuntimeWarning)
            base = np.nanmedian(vals[:-k], axis=0)
        with np.errstate(invalid="ignore"):
            rel = vals[-k:] / base - 1.0
            shifted = np.all(rel > frac, axis=0) | np.all(rel < -frac, axis=0)
        if shifted.any():
            self._r.forget(shifted, keep_last=k)

    @property
    def if_threshold(self) -> float:
        return self.anomaly.threshold or self.th.if_alert

    def mark_injection(self, cell: int):
        """Forget earlier alerts on ``cell`` so detection time is measured from a fault injected now."""
        i = cell - 1
        self.first_flag_t[i] = np.nan
        self.first_z_t[i] = np.nan
        self.bms_first_t[i] = np.nan
        self.last_z_t[i] = -np.inf
        self.persist[i] = 0

    def _flag(self, mask: np.ndarray, t: float):
        new = mask & np.isnan(self.first_flag_t)
        self.first_flag_t[new] = t

    # ------------------------------------------------------------------ features + evaluation
    def r_cell(self) -> np.ndarray:
        return self._r_cache.copy()

    def _dsoc(self, I, v1_nom, soc_med, dev, r_then, r, rmed) -> np.ndarray:
        """Per-cell charge offset vs the pack (%SoC): undo the IR drop and polarisation.

        Each sample is corrected with the resistance estimate from its own time (once settled), so a
        resistance that changes during the drive does not leave a fake charge drift; samples logged
        before the estimate settled use the latest estimate.
        """
        if len(I) == 0:
            return np.zeros((0, self.n))
        if not np.isfinite(rmed):
            rmed = self.p.r0
        latest = np.where(np.isfinite(r), r, rmed)
        rr = np.where(np.isfinite(r_then), r_then, latest[None, :])
        rm = np.median(rr, axis=1, keepdims=True)
        ocv = dev + I[:, None] * (rr - rm) + v1_nom[:, None] * (rr / rm - 1.0)
        # invert the OCV curve exactly (a linear slope would bend large offsets on a curved OCV)
        ocv_med = self.p.ocv_at(soc_med)[:, None]
        return (self.p.soc_at(ocv_med + ocv) - soc_med[:, None]) * 100.0

    def features(self) -> dict:
        r = self.r_cell()
        rmed = np.nanmedian(r) if np.any(np.isfinite(r)) else np.nan
        D, P = self._drive.values(), self._park.values()
        n = self.n
        fe = fade_slope(D[:, 0], self._dsoc(D[:, 1], D[:, 2], D[:, 3], D[:, 4:4 + n], D[:, 4 + n:], r, rmed))
        pk = parked_drift(P[:, 0], P[:, 1], self._dsoc(P[:, 2], P[:, 3], P[:, 4], P[:, 5:5 + n], P[:, 5 + n:], r, rmed))
        rest_mv, load_mv = self.rest_off * 1000, self.load_off * 1000
        return {
            "r_ohm": r, "r_ratio": r / rmed, "rest_offset_mv": rest_mv, "load_offset_mv": load_mv,
            "sag_mv": load_mv - rest_mv, "fade_slope": fe.slope, "fade_se": fe.se, "ah_span": fe.ah_span,
            "step_pct": fe.step, "sd_drift": pk.rate, "sd_se": pk.se, "parked_s": pk.parked_s,
        }

    def evaluate(self):
        feat = self.features()
        ifs = self.anomaly.score(feature_matrix(feat)) if self.anomaly.ready else np.full(self.n, np.nan)
        soc = self.soc_est if self.soc_est is not None else np.full(self.n, 0.5)
        z_flag = (self.t - self.last_z_t) <= 120
        impact = range_impact(soc, self.cap_est, self.cfg)
        findings = diagnose(feat, self._z.mean(), z_flag, ifs, impact.cost_km, self.cfg, self.th, self.if_threshold)
        fading = [f.cell - 1 for f in findings if f.kind == "capacity_fade"]
        if fading:  # online capacity estimate: d(dSoC)/dAh = -(1/C_i - 1/C_median)
            cap = self.cap_est.copy()
            slope = feat["fade_slope"][fading] / 100.0
            cap[fading] = np.minimum(cap[fading], 1.0 / (1.0 / np.median(cap) - slope))
            impact = range_impact(soc, cap, self.cfg)
            for f in findings:
                f.km_cost = float(impact.cost_km[f.cell - 1])
        mask = np.zeros(self.n, dtype=bool)
        for f in findings:
            mask[f.cell - 1] = True
        self._flag(mask, self.t)
        self._cache = (feat, ifs, impact, findings)
        self._last_eval = self.t

    def snapshot(self) -> Snapshot:
        if self._cache is None or self.t - self._last_eval > 1.0:
            self.evaluate()
        feat, ifs, impact, findings = self._cache
        ht, hv, hz = self._ht.values(), self._hv.values(), self._hz.values()
        q = np.percentile(hv, [5, 50, 95], axis=1) if len(hv) else np.zeros((3, 0))
        recent = hv[-90:]
        dev90 = (recent - np.median(recent, axis=1, keepdims=True)).mean(axis=0) * 1000 if len(recent) else np.zeros(self.n)
        status = np.array(["ok"] * self.n, dtype=object)
        for f in findings:
            status[f.cell - 1] = f.status
        soc = self.soc_est if self.soc_est is not None else np.full(self.n, np.nan)
        return Snapshot(
            t=self.t, current=self.current, pack_voltage=float(np.nansum(self.v)), pack_soc=float(np.median(soc)),
            spread_mv=float(np.ptp(dev90)) if len(dev90) else 0.0, dev90_mv=dev90, zroll=self._z.mean(), if_score=ifs,
            features=feat, findings=findings, status=status, impact=impact, first_flag_t=self.first_flag_t.copy(),
            first_z_t=self.first_z_t.copy(), bms_first_t=self.bms_first_t.copy(),
            hist_t=ht[:, 0] if len(ht) else np.zeros(0), hist_i=ht[:, 1] if len(ht) else np.zeros(0), hist_v=hv, hist_z=hz,
            hist_med=q[1], hist_p05=q[0], hist_p95=q[2],
        )
