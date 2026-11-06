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


def test_ring_forget_keeps_only_the_newest_rows_of_the_chosen_columns():
    from cellsense.analytics.engine import _Ring

    ring = _Ring(4, 2)
    for k in range(6):
        ring.push(np.array([k, 10 + k], dtype=float))
    ring.forget(np.array([True, False]), keep_last=2)
    v = ring.values()
    assert np.isnan(v[:2, 0]).all() and list(v[2:, 0]) == [4, 5]
    assert list(v[:, 1]) == [12, 13, 14, 15]


def _square_wave(eng, r, levels, t, hold=3):
    """Hold each current level for ``hold`` seconds; cell voltage = 3.7 V − I·R, no noise."""
    for current in levels:
        for _ in range(hold):
            t += 1
            eng.update(float(t), float(current), 3.7 - current * r)
    return t


def test_resistance_estimate_follows_a_real_change_within_two_steps():
    eng = CellSenseEngine()
    r = np.full(96, 0.8e-3)
    t = _square_wave(eng, r, [0, 150] * 7, 0)  # 13 current steps: the 11-step ring is full of healthy readings
    worse = r.copy()
    worse[28] *= 1.4
    _square_wave(eng, worse, [0, 150, 0], t)  # only 2 steps see the new resistance
    ratio = eng._r_cache[28] / np.median(eng._r_cache)
    assert ratio == pytest.approx(1.4, abs=0.02)  # a plain 11-step median would still read ~1.0 here


def test_one_odd_step_does_not_move_the_resistance_estimate():
    eng = CellSenseEngine()
    r = np.full(96, 0.8e-3)
    t = _square_wave(eng, r, [0, 150] * 7, 0)
    for k in range(3):  # back up to 150 A; only the first sample of the step reads cell 29 high
        t += 1
        v = 3.7 - 150.0 * r
        if k == 0:
            v[28] -= 150.0 * 0.4 * r[28]
        eng.update(float(t), 150.0, v)
    t = _square_wave(eng, r, [0, 150, 0], t)
    ratio = eng._r_cache[28] / np.median(eng._r_cache)
    assert ratio == pytest.approx(1.0, abs=0.02)


def test_fault_lab_reaches_replace_within_two_minutes_of_driving():
    """The live demo: +36% resistance injected mid-drive is called 'replace' quickly, not after minutes of cruising."""
    from cellsense.dashboard.session import LiveSession

    sess = LiveSession(mode="demo", speed=1, warmup_s=600)
    sess.stop()
    for seed in (23, 71):
        sess.reset("demo", seed=seed)
        for _ in range(int(np.random.default_rng(seed).integers(30, 400))):
            sess._tick()
        sess.inject("resistance_growth", 29, 0.36)
        called = None
        for k in range(1, 121):
            sess._tick()
            f = sess.engine.snapshot().finding(29) if k % 5 == 0 else None
            if f is not None and f.status == "replace":
                called = k
                break
        assert called is not None, f"seed {seed}: no 'replace' within 120 s"
