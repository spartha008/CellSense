"""Diagnose: turn per-cell evidence into a plain-language finding and an action.

Rules (first match wins):
  resistance >= 1.2x pack median                 -> replace  (resistance growth)
  charge keeps falling behind while parked       -> inspect  (self-discharge)
  charge falls behind per Ah discharged          -> monitor, then replace (capacity fade)
      ...but no parked data yet to rule out a leak -> monitor (charge loss: fade or leak)
  charge offset vs pack, resistance normal       -> balance  (charge imbalance)
  resistance >= 1.08x, Isolation Forest, z alert -> monitor  (early drift / outlier)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PACK, THRESHOLDS, PackConfig, Thresholds

STATUS_ORDER = {"replace": 0, "inspect": 1, "balance": 2, "monitor": 3, "ok": 9}
SEVERITY = {"replace": "critical", "inspect": "warning", "balance": "warning", "monitor": "monitor", "ok": "ok"}
ACTIONABLE_KINDS = {"resistance_growth", "self_discharge", "capacity_fade", "charge_loss", "charge_imbalance"}


@dataclass
class CellFinding:
    cell: int
    status: str  # replace | inspect | balance | monitor
    kind: str  # resistance_growth | self_discharge | capacity_fade | charge_loss | charge_imbalance | early_drift
    severity: str  # critical | warning | monitor
    headline: str
    detail: str
    action: str
    location: str
    z: float
    if_score: float
    r_ratio: float
    rest_mv: float
    load_mv: float
    sag_mv: float
    fade_slope: float
    sd_drift: float
    km_cost: float

    @property
    def actionable(self) -> bool:
        """A specific diagnosis, as opposed to 'early drift' (a watch-list entry)."""
        return self.kind in ACTIONABLE_KINDS


def _pct(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "n/a"
    d = (ratio - 1) * 100
    return f"{'+' if d >= 0 else '−'}{abs(d):.0f}%"


def _ok(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def _worse(r, rest, slope, drift) -> bool:
    """Is the outlier on the weak side of the pack? (A low-resistance cell is an outlier, not a problem.)"""
    return ((_ok(r) and r >= 1.03) or (_ok(rest) and abs(rest) >= 4.0) or (_ok(slope) and slope <= -0.005)
            or (_ok(drift) and drift <= -1.0))


def diagnose(feat: dict, zroll: np.ndarray, z_flag: np.ndarray, if_score: np.ndarray, km_cost: np.ndarray,
             cfg: PackConfig = PACK, th: Thresholds = THRESHOLDS, if_alert: float | None = None) -> list[CellFinding]:
    if_alert = th.if_alert if if_alert is None else if_alert
    parked_enough = feat.get("parked_s", 0.0) >= th.min_parked_s
    out: list[CellFinding] = []
    for i in range(len(zroll)):
        r, rest, load, sag = feat["r_ratio"][i], feat["rest_offset_mv"][i], feat["load_offset_mv"][i], feat["sag_mv"][i]
        slope, drift, ifs = feat["fade_slope"][i], feat["sd_drift"][i], if_score[i]
        leaking = (_ok(drift) and drift <= th.self_discharge_pct_per_h and drift + 4 * feat["sd_se"][i] < 0 and parked_enough)
        fading = (_ok(slope) and slope <= th.fade_slope_pct_per_ah and slope + 3 * feat["fade_se"][i] < 0
                  and feat["ah_span"] >= th.min_fade_ah)
        cell = i + 1
        if _ok(r) and r >= th.r_replace_ratio:
            status, kind = "replace", "resistance_growth"
            head = "Physically degrading"
            under = f"; drops {abs(load):.0f} mV below the pack under load" if _ok(load) else ""
            detail = f"Internal resistance {_pct(r)} vs pack{under}."
            action = "Replace at next service"
        elif leaking:
            status, kind = "inspect", "self_discharge"
            head = "Losing charge on its own"
            detail = f"Falls {abs(drift):.1f}% SoC per hour behind the pack, even while parked."
            action = "Inspect for internal leakage"
        elif fading and parked_enough:
            status, kind = "monitor", "capacity_fade"
            head = "Capacity fading"
            detail = f"Drifts further below the pack as it discharges: about {-slope * cfg.capacity_ah:.1f}% less capacity."
            action = "Monitor, then replace"
        elif fading:
            status, kind = "monitor", "charge_loss"
            head = "Losing charge faster than the pack"
            detail = "Capacity fade or self-discharge; a parked period will tell them apart."
            action = "Monitor; re-check after parking"
        elif _ok(rest) and abs(rest) >= th.balance_offset_mv and (not _ok(r) or r < th.r_monitor_ratio):
            status, kind = "balance", "charge_imbalance"
            head = "Low on charge, not damaged" if rest < 0 else "Over-charged vs the pack"
            detail = f"Charge sits {abs(rest):.0f} mV {'below' if rest < 0 else 'above'} the pack; resistance normal ({_pct(r)})."
            action = "Balance at next charge"
        elif (_ok(r) and r >= th.r_monitor_ratio) or bool(z_flag[i]) or (_ok(ifs) and ifs >= if_alert and _worse(r, rest, slope, drift)):
            status, kind = "monitor", "early_drift"
            head = "Early drift"
            bits = [f"resistance {_pct(r)}"] if _ok(r) else []
            if _ok(ifs):
                bits.append(f"Isolation Forest score {ifs:.2f}")
            lead = f"Sitting below the pack (z = {zroll[i]:.1f}); " if bool(z_flag[i]) else "Outlier vs the pack: "
            detail = (lead + ", ".join(bits) + ".") if bits else lead.rstrip("; ") + "."
            action = "Re-check in 30 cycles"
        else:
            continue
        f = lambda x: float(x) if _ok(x) else float("nan")  # noqa: E731
        out.append(CellFinding(cell, status, kind, SEVERITY[status], head, detail, action, cfg.label(cell),
                               float(zroll[i]), f(ifs), f(r), f(rest), f(load), f(sag), f(slope), f(drift), float(km_cost[i])))
    out.sort(key=lambda x: (STATUS_ORDER[x.status], -x.km_cost))
    return out
