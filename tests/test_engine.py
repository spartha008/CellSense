"""End-to-end: simulated packs through the CellSense engine."""
import numpy as np
import pytest

from cellsense.analytics import CellSenseEngine
from cellsense.sim import Fault, demo_pack, drive_cycle, healthy_pack


def _run(sim, hist, profile, seed, fault=None):
    eng = CellSenseEngine(capacity_ah=hist.capacity[-1])
    if fault:
        sim.schedule(fault)
    for i in drive_cycle(profile, 1800, seed):
        v = sim.step(i)
        if fault and abs(sim.t - 1 - fault.t_start) < 1e-9:
            eng.mark_injection(fault.cell)
        eng.update(sim.t, i, v)
    return eng.snapshot()


@pytest.fixture(scope="module")
def demo():
    sim, hist = demo_pack()
    return sim, _run(sim, hist, "mixed", 3)


def test_demo_pack_diagnoses(demo):
    _, s = demo
    by = {f.cell: f for f in s.findings}
    assert by[47].status == "replace" and by[47].kind == "resistance_growth"
    assert by[12].status == "balance" and by[12].kind == "charge_imbalance"
    assert 83 in by and by[83].status == "monitor"
    assert s.impact.limiting_cell == 47


def test_demo_range_lost_matches_truth(demo):
    sim, s = demo
    q = sim.soc() * sim.cap
    truth = np.clip(np.median(q) - q, 0, None).max() / 150 * 350
    assert s.impact.lost_km == pytest.approx(truth, abs=1.5)


def test_healthy_pack_has_no_specific_diagnoses():
    sim, hist = healthy_pack(5)
    s = _run(sim, hist, "city", 4)
    assert not [f for f in s.findings if f.actionable]


@pytest.mark.parametrize("kind,sev,expect", [("resistance_growth", 0.35, "replace"), ("charge_imbalance", 0.02, "balance")])
def test_injected_fault_is_caught_quickly_and_bms_stays_silent(kind, sev, expect):
    sim, hist = healthy_pack(21)
    f = Fault(kind, 29, sev, 300)
    s = _run(sim, hist, "mixed", 31, f)
    assert s.first_flag_t[28] - f.t_start < 120
    assert s.finding(29).status == expect
    assert np.isnan(s.bms_first_t[28])


@pytest.mark.parametrize("label,v29,current,expect", [
    ("60 mV low at rest", 3.640, 0.5, True),
    ("40 mV low at rest, under the 50 mV limit", 3.660, 0.5, False),
    ("60 mV low while driving: the rest rule only applies at rest", 3.640, 100.0, False),
    ("below the 2.8 V floor", 2.750, 100.0, True),
    ("above the 4.2 V ceiling", 4.250, -50.0, True),
])
def test_threshold_bms_alarm_rule(label, v29, current, expect):
    eng = CellSenseEngine()
    rng = np.random.default_rng(0)
    for t in range(1, 31):
        v = 3.70 + 0.001 * rng.standard_normal(96)
        v[28] = v29
        eng.update(float(t), current, v)
    assert np.isfinite(eng.bms_first_t[28]) == expect, label
    assert np.isnan(np.delete(eng.bms_first_t, 28)).all()


def test_threshold_bms_alarms_on_a_gross_imbalance_long_after_cellsense():
    """Keeps the 'BMS stays silent' tests honest: the baseline does fire once a fault is big enough."""
    sim, hist = healthy_pack(21)
    f = Fault("charge_imbalance", 29, 0.08, 300)
    s = _run(sim, hist, "city", 31, f)
    assert np.isfinite(s.bms_first_t[28]) and np.isfinite(s.first_flag_t[28])
    assert s.first_flag_t[28] < s.bms_first_t[28]
