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
