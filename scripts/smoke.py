"""Quick end-to-end check: demo pack, a healthy pack, and one live injection of each fault type."""
import sys, time
import numpy as np
from cellsense.analytics import CellSenseEngine
from cellsense.analytics.training import train_anomaly_model
from cellsense.sim import Fault, demo_pack, drive_cycle, healthy_pack

if "--train" in sys.argv:
    t0 = time.time(); train_anomaly_model(n_packs=30); print(f"trained IF in {time.time() - t0:.0f}s")

def run(sim, hist, prof, fault=None):
    eng = CellSenseEngine(capacity_ah=hist.capacity[-1])
    if fault: sim.schedule(fault)
    t0 = time.time()
    for i in prof:
        v = sim.step(i)
        if fault and abs(sim.t - fault.t_start - 1) < 1e-9: eng.mark_injection(fault.cell)
        eng.update(sim.t, i, v)
    return eng, eng.snapshot(), (time.time() - t0) / len(prof) * 1000

def show(title, s, ms, fault=None, eng=None):
    print(f"\n== {title}  ({ms:.2f} ms/sample)  lost {s.impact.lost_km:.1f} km, limiting cell {s.impact.limiting_cell}, findings {len(s.findings)}")
    for f in s.findings[:6]:
        print(f"  {f.cell:>2} {f.status:<8} {f.kind:<18} IF {f.if_score:.2f} r {f.r_ratio:.2f} rest {f.rest_mv:6.1f} sag {f.sag_mv:6.1f} "
              f"slope {f.fade_slope:7.4f} drift {f.sd_drift:6.2f} km {f.km_cost:5.1f} | {f.detail}")
    if fault:
        i = fault.cell - 1
        d = s.first_flag_t[i] - fault.t_start
        print(f"  injected {fault.describe()} at t={fault.t_start:.0f}s -> flagged after {d:.0f}s, BMS alarm: {s.bms_first_t[i]}")

sim, hist = demo_pack(); eng, s, ms = run(sim, hist, drive_cycle("mixed", 1800, 3)); show("demo pack", s, ms)
sim, hist = healthy_pack(5); eng, s, ms = run(sim, hist, drive_cycle("city", 1800, 4)); show("healthy pack", s, ms)
for k, (kind, sev) in enumerate([("resistance_growth", 0.35), ("charge_imbalance", 0.02), ("capacity_fade", 0.04), ("self_discharge", 0.06)]):
    sim, hist = healthy_pack(20 + k); f = Fault(kind, 29, sev, 300)
    eng, s, ms = run(sim, hist, drive_cycle("mixed", 1800, 30 + k), f); show(f"inject {kind}", s, ms, f)
