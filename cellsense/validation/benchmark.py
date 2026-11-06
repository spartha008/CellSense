"""Randomised benchmark: inject faults into simulated packs and score CellSense against them.

Reports the four numbers the pitch promises:
  detection delay      seconds from injection to the first flag on the faulty cell
  precision / recall   specific diagnoses on faulty vs healthy cells
  diagnosis accuracy   whether the action matches the injected fault
  forecast error       predicted vs actual cycles until the replace threshold
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from ..analytics import CellSenseEngine, forecast
from ..config import PACK, PROJECT_DIR, THRESHOLDS
from ..sim import EXPECTED_ACTION, LABELS, PROFILES, SEVERITY_RANGES, Fault, capacity_history, drive_cycle, healthy_pack

ACTION_OF = {"replace": "replace", "balance": "balance", "inspect": "inspect", "monitor": "monitor"}


@dataclass
class RunResult:
    seed: int
    profile: str
    kind: str | None
    cell: int | None
    severity: float | None
    t_inj: float | None
    detected: bool
    delay_s: float | None
    final_status: str | None
    final_kind: str | None
    diagnosis_ok: bool | None
    parked_s: float
    bms_detected: bool
    bms_delay_s: float | None
    healthy_actionable: int  # specific diagnoses on cells with no fault (false alarms)
    healthy_watch: int  # early-drift watch-list entries on cells with no fault
    false_alarm_kinds: list
    unjustified_false_alarms: int  # healthy-cell diagnoses the simulator's ground truth does not support


def _run(job: dict) -> dict:
    seed, kind, profile, duration = job["seed"], job["kind"], job["profile"], job["duration"]
    rng = np.random.default_rng(seed)
    sim, hist = healthy_pack(seed)
    eng = CellSenseEngine(capacity_ah=hist.capacity[-1], eval_every_s=10.0)
    fault = None
    if kind:
        lo, hi = SEVERITY_RANGES[kind]
        fault = Fault(kind, int(rng.integers(1, PACK.n_series + 1)), float(lo + (hi - lo) * rng.random()), float(rng.integers(300, 900)))
        sim.schedule(fault)
    trip = np.r_[drive_cycle(profile, duration, seed), 0.3 + 0.7 * rng.random(job.get("park_s", 600))]  # drive, then park
    for i in trip:
        v = sim.step(i)
        if fault and abs(sim.t - 1 - fault.t_start) < 1e-9:
            eng.mark_injection(fault.cell)
        eng.update(sim.t, i, v)
    s = eng.snapshot()
    fc = fault.cell if fault else None
    healthy = [f for f in s.findings if f.cell != fc]
    r = RunResult(seed, profile, kind, fc, fault.severity if fault else None, fault.t_start if fault else None,
                  detected=False, delay_s=None, final_status=None, final_kind=None, diagnosis_ok=None,
                  parked_s=float(s.features["parked_s"]), bms_detected=False, bms_delay_s=None,
                  healthy_actionable=sum(f.actionable for f in healthy), healthy_watch=sum(not f.actionable for f in healthy),
                  false_alarm_kinds=[f.kind for f in healthy if f.actionable], unjustified_false_alarms=0)
    # ground truth: a non-injected cell can still genuinely be out of line (manufacturing spread)
    soc_off = (sim.soc() - np.median(sim.soc())) * 100
    cap_def = (1 - sim.cap / np.median(sim.cap)) * 100
    r_ratio = sim.r0 / np.median(sim.r0)
    for f in healthy:
        if not f.actionable:
            continue
        i = f.cell - 1
        ok = {"charge_imbalance": abs(soc_off[i]) >= 0.8, "capacity_fade": cap_def[i] >= 1.0 or soc_off[i] <= -0.8,
              "charge_loss": cap_def[i] >= 1.0 or soc_off[i] <= -0.8, "resistance_growth": r_ratio[i] >= 1.12, "self_discharge": False}[f.kind]
        r.unjustified_false_alarms += int(not ok)
    if fault:
        i = fault.cell - 1
        t_flag = s.first_flag_t[i]
        r.detected = bool(np.isfinite(t_flag) and t_flag >= fault.t_start)
        r.delay_s = float(t_flag - fault.t_start) if r.detected else None
        fnd = s.finding(fault.cell)
        if fnd:
            r.final_status, r.final_kind = fnd.status, fnd.kind
        expected = EXPECTED_ACTION[kind]
        r.diagnosis_ok = bool(fnd is not None and fnd.status == expected and (kind != "capacity_fade" or fnd.kind in ("capacity_fade", "charge_loss")))
        tb = s.bms_first_t[i]
        r.bms_detected = bool(np.isfinite(tb) and tb >= fault.t_start)
        r.bms_delay_s = float(tb - fault.t_start) if r.bms_detected else None
    return asdict(r)


def forecast_error(n_cells: int = 60, seed: int = 7) -> dict:
    """Generate long capacity histories, fit on the first 320 cycles, compare predicted vs actual crossing."""
    rng = np.random.default_rng(seed)
    errs = []
    for k in range(n_cells):
        cap_end = np.full(PACK.n_series, PACK.capacity_ah) * (1 + 0.0035 * rng.standard_normal(PACK.n_series))
        cap_end[0] *= 1 - rng.uniform(0.07, 0.12)  # one cell that will degrade well past the threshold
        full = capacity_history(cap_end, now_cycle=640, seed=int(rng.integers(1e6)))
        med = np.median(full.capacity, axis=1)
        km = (med - full.capacity[:, 0]) / med * PACK.rated_range_km
        hit = np.nonzero(km >= THRESHOLDS.replace_km)[0]
        if not len(hit) or full.cycles[hit[0]] <= 360:
            continue
        actual = float(full.cycles[hit[0]])
        cut = full.cycles <= 320
        part = type(full)(full.cycles[cut], full.capacity[cut], 320)
        pred = forecast(part, horizon=400, cells=[1])[0].cross_cycle
        if pred is not None:
            errs.append(pred - actual)
    e = np.array(errs)
    return {"cells": int(len(e)), "mae_cycles": float(np.mean(np.abs(e))) if len(e) else None,
            "mae_months": float(np.mean(np.abs(e)) / THRESHOLDS.cycles_per_month) if len(e) else None,
            "bias_cycles": float(np.mean(e)) if len(e) else None}


def summarise(rows: list[dict], fc: dict) -> dict:
    faulty = [r for r in rows if r["kind"]]
    healthy_runs = [r for r in rows if not r["kind"]]
    tp = sum(1 for r in faulty if r["final_kind"] not in (None, "early_drift"))
    fp = sum(r["healthy_actionable"] for r in rows)
    per = {}
    for k in SEVERITY_RANGES:
        rs = [r for r in faulty if r["kind"] == k]
        d = [r["delay_s"] for r in rs if r["delay_s"] is not None]
        per[k] = {
            "runs": len(rs), "detected_pct": 100 * np.mean([r["detected"] for r in rs]) if rs else None,
            "delay_median_s": float(np.median(d)) if d else None, "delay_p90_s": float(np.percentile(d, 90)) if d else None,
            "diagnosis_ok_pct": 100 * np.mean([r["diagnosis_ok"] for r in rs]) if rs else None,
            "threshold_bms_detected_pct": 100 * np.mean([r["bms_detected"] for r in rs]) if rs else None,
            "final_kinds": {kk: sum(r["final_kind"] == kk for r in rs) for kk in sorted({str(r["final_kind"]) for r in rs})},
        }
    return {
        "runs": len(rows), "fault_runs": len(faulty), "healthy_runs": len(healthy_runs),
        "recall_pct": 100 * tp / max(len(faulty), 1), "precision_pct": 100 * tp / max(tp + fp, 1),
        "false_alarms_per_pack": fp / max(len(rows), 1),
        "unjustified_per_pack": sum(r["unjustified_false_alarms"] for r in rows) / max(len(rows), 1),
        "precision_truth_pct": 100 * (tp + fp - sum(r["unjustified_false_alarms"] for r in rows)) / max(tp + fp, 1),
        "watch_list_per_pack": float(np.mean([r["healthy_watch"] for r in rows])),
        "detected_any_pct": 100 * np.mean([r["detected"] for r in faulty]) if faulty else None,
        "diagnosis_ok_pct": 100 * np.mean([r["diagnosis_ok"] for r in faulty]) if faulty else None,
        "threshold_bms_detected_pct": 100 * np.mean([r["bms_detected"] for r in faulty]) if faulty else None,
        "per_fault": per, "forecast": fc,
        "false_alarm_kinds": {k: sum(r["false_alarm_kinds"].count(k) for r in rows) for k in sorted({k for r in rows for k in r["false_alarm_kinds"]})},
    }


def write_report(summary: dict, rows: list[dict], seconds: float) -> str:
    out = PROJECT_DIR / "reports"
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=1, default=float))
    s = summary
    f = lambda v, d=0, u="": "–" if v is None else f"{v:.{d}f}{u}"  # noqa: E731
    lines = [
        "# CellSense benchmark", "",
        f"{s['fault_runs']} randomised fault runs + {s['healthy_runs']} healthy packs, 96 cells each: a 30 min drive "
        f"(city / highway / mixed) followed by 10 min parked, real-time ECM simulator fitted to PyBaMM. Run time {seconds:.0f} s.", "",
        "| Metric | Result |", "|---|---|",
        f"| Faults detected (any flag on the faulty cell) | {f(s['detected_any_pct'], 1, '%')} |",
        f"| Recall (faulty cell gets a specific diagnosis) | {f(s['recall_pct'], 1, '%')} |",
        f"| Precision vs injected faults only | {f(s['precision_pct'], 1, '%')} |",
        f"| Precision vs ground truth (cell really out of line: ≥0.8% SoC, ≥1% capacity or ≥12% resistance) | {f(s['precision_truth_pct'], 1, '%')} |",
        f"| Diagnoses on non-injected cells per pack | {f(s['false_alarms_per_pack'], 2)} ({', '.join(f'{k}: {v}' for k, v in s['false_alarm_kinds'].items()) or 'none'}) |",
        f"| ...of which not supported by ground truth (true false alarms) | {f(s['unjustified_per_pack'], 2)} per pack |",
        f"| Watch-list entries per pack (early drift, not an alarm) | {f(s['watch_list_per_pack'], 2)} |",
        f"| Diagnosis matches the injected fault | {f(s['diagnosis_ok_pct'], 1, '%')} |",
        f"| Threshold-only BMS detected the fault | {f(s['threshold_bms_detected_pct'], 1, '%')} |",
        f"| Forecast error, cycles to 20 km threshold (MAE) | {f(s['forecast']['mae_cycles'], 0)} cycles (~{f(s['forecast']['mae_months'], 1)} months), {s['forecast']['cells']} cells |",
        "", "## By fault type", "",
        "| Fault | Runs | Detected | Median delay | 90th pct delay | Right action | Threshold BMS | Final diagnoses |", "|---|---|---|---|---|---|---|---|",
    ]
    for k, p in s["per_fault"].items():
        lines.append(f"| {LABELS[k]} | {p['runs']} | {f(p['detected_pct'], 0, '%')} | {f(p['delay_median_s'], 0, ' s')} | {f(p['delay_p90_s'], 0, ' s')} | "
                     f"{f(p['diagnosis_ok_pct'], 0, '%')} | {f(p['threshold_bms_detected_pct'], 0, '%')} | {', '.join(f'{a}: {b}' for a, b in p['final_kinds'].items())} |")
    lines += ["", "Severities are drawn uniformly from: " + ", ".join(f"{LABELS[k]} {lo:g}–{hi:g}" for k, (lo, hi) in SEVERITY_RANGES.items()) + ".",
              "Self-discharge and capacity fade look alike while driving; the parked period at the end of each trip is what separates them. "
              "'charge_loss' means fade-or-leak (not yet separated) and is scored as right for capacity fade, wrong for self-discharge."]
    text = "\n".join(lines) + "\n"
    (out / "benchmark.md").write_text(text, encoding="utf-8")
    return text


def run_benchmark(runs: int = 200, healthy: int = 40, duration: int = 1800, seed: int = 1, workers: int = 0) -> dict:
    kinds = list(SEVERITY_RANGES)
    jobs = [{"seed": seed * 100_000 + k, "kind": kinds[k % len(kinds)], "profile": PROFILES[(k // len(kinds)) % len(PROFILES)], "duration": duration} for k in range(runs)]
    jobs += [{"seed": seed * 100_000 + runs + k, "kind": None, "profile": PROFILES[k % len(PROFILES)], "duration": duration} for k in range(healthy)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    print(f"benchmark: {len(jobs)} runs on {workers} processes ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_run, jobs, chunksize=4))
    fc = forecast_error()
    summary = summarise(rows, fc)
    print(write_report(summary, rows, time.time() - t0))
    return summary
