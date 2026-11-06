import numpy as np
import pytest

from cellsense.analytics import range_impact
from cellsense.analytics.anomaly import AnomalyModel, pack_scale
from cellsense.analytics.drift import fade_slope, parked_drift
from cellsense.analytics.forecast import forecast
from cellsense.config import PACK
from cellsense.sim import CellParams, Fault, PackSimulator, capacity_history


def test_ocv_is_invertible():
    p = CellParams.load()
    soc = np.linspace(0.05, 0.95, 19)
    assert np.all(np.diff(p.ocv) > 0)
    assert np.allclose(p.soc_at(p.ocv_at(soc)), soc, atol=1e-3)


def test_faults_change_the_right_cell():
    base = PackSimulator(seed=1, noise_v=0, adc_lsb_v=0)
    sims = {k: PackSimulator(seed=1, noise_v=0, adc_lsb_v=0, conditions=[Fault(k, 10, s)]) for k, s in
            [("resistance_growth", 0.4), ("charge_imbalance", 0.02), ("capacity_fade", 0.05), ("self_discharge", 0.1)]}
    assert sims["resistance_growth"].r0[9] == pytest.approx(base.r0[9] * 1.4)
    assert sims["capacity_fade"].cap[9] == pytest.approx(base.cap[9] * 0.95)
    assert sims["charge_imbalance"].soc()[9] < base.soc()[9] - 0.015
    v_load = {k: s.step(200.0)[9] for k, s in sims.items()}
    v0 = base.step(200.0)[9]
    assert v_load["resistance_growth"] < v0 - 0.05  # 40% more resistance sags ~70 mV at 200 A
    s = sims["self_discharge"]
    q0 = s.q[9]
    for _ in range(600):
        s.step(0.0)
    assert q0 - s.q[9] == pytest.approx(0.1 * s.cap[9] * 600 / 3600, rel=0.02)


def test_range_impact_formula():
    soc = np.full(96, 0.6)
    cap = np.full(96, 150.0)
    cap[46] = 150.0 * 0.969
    imp = range_impact(soc, cap, PACK)
    assert imp.limiting_cell == 47
    assert imp.lost_km == pytest.approx(0.6 * 150 * 0.031 / 150 * 350, rel=1e-6)


def test_fade_slope_recovers_capacity_difference():
    rng = np.random.default_rng(0)
    ah = np.linspace(0, 30, 900)
    true = np.zeros(96)
    true[5] = -0.03  # %SoC per Ah
    Y = np.outer(ah, true) + 0.1 * rng.standard_normal((900, 96))
    est = fade_slope(ah, Y)
    assert est.slope[5] == pytest.approx(-0.03, abs=0.002)
    assert np.nanmax(np.abs(np.delete(est.slope, 5))) < 0.004


def test_sudden_step_is_not_mistaken_for_fade():
    rng = np.random.default_rng(1)
    ah = np.linspace(0, 30, 900)
    Y = 0.1 * rng.standard_normal((900, 96))
    Y[450:, 7] -= 2.0  # 2 %SoC removed at once halfway through
    est = fade_slope(ah, Y)
    assert est.step[7] == pytest.approx(-2.0, abs=0.1)
    assert abs(est.slope[7]) < 0.01


def test_parked_drift_finds_leak_and_ignores_steps():
    rng = np.random.default_rng(2)
    hours = np.r_[np.linspace(0, 0.07, 250), np.linspace(1, 1.07, 250)]
    pid = np.r_[np.zeros(250), np.ones(250)]
    Y = 0.1 * rng.standard_normal((500, 96))
    Y[:, 3] += -6.0 * hours  # leaking 6 %SoC per hour
    Y[125:250, 9] -= 2.0  # a step inside the first park
    d = parked_drift(pid, hours, Y)
    assert d.rate[3] == pytest.approx(-6.0, abs=1.0)
    assert abs(d.rate[9]) < 2.0


def test_forecast_crossing():
    cap_now = np.full(96, 150.0)
    cap_now[0] *= 1 - 0.035
    hist = capacity_history(cap_now, now_cycle=320, seed=3)
    f = forecast(hist, cells=[1])[0]
    assert 9 < f.now_km < 14
    assert f.cross_cycle is not None and 380 < f.cross_cycle < 620


def test_fast_anomaly_score_matches_sklearn():
    m = AnomalyModel.load()
    if not m.ready:
        pytest.skip("anomaly model not trained (python -m cellsense train)")
    X = np.random.default_rng(4).standard_normal((96, 5))
    assert np.allclose(m.score(X), -m.model.score_samples(pack_scale(X).astype(np.float32)))


def _demo_forecasts():
    cap = np.full(PACK.n_series, 150.0)
    cap[46] *= 0.965
    hist = capacity_history(cap, now_cycle=320, seed=3)
    return forecast(hist)


def test_injected_capacity_fade_shifts_the_trend_up_from_today():
    from cellsense.analytics.forecast import apply_effects, fault_effect

    fcs = _demo_forecasts()
    e = fault_effect(Fault("capacity_fade", 29, 0.04), 0.0, CellParams.load(), 1.3e-3)
    assert e.permanent_km == pytest.approx(0.04 * PACK.rated_range_km) and e.recoverable_km == 0
    out = apply_effects(fcs, [e])
    before, after = fcs[28], out[28]
    assert np.allclose(after.fy - before.fy, e.permanent_km)
    assert after.km_6mo == pytest.approx(before.km_6mo + e.permanent_km)
    assert after.fault_label == "-4.0% capacity"
    assert all(a is b for k, (a, b) in enumerate(zip(fcs, out)) if k != 28)  # every other cell untouched


def test_recoverable_faults_cost_range_today_but_leave_the_trend():
    from cellsense.analytics.forecast import apply_effects, fault_effect

    fcs = _demo_forecasts()
    p = CellParams.load()
    imb = fault_effect(Fault("charge_imbalance", 29, 0.02), 0.0, p, 1.3e-3)
    leak_1h = fault_effect(Fault("self_discharge", 29, 0.06), 3600.0, p, 1.3e-3)
    assert imb.permanent_km == 0 and imb.recoverable_km == pytest.approx(0.02 * PACK.rated_range_km)
    assert fault_effect(Fault("self_discharge", 29, 0.06), 0.0, p, 1.3e-3).recoverable_km == 0
    assert leak_1h.recoverable_km == pytest.approx(0.06 * PACK.rated_range_km)
    after = apply_effects(fcs, [imb])[28]
    assert np.array_equal(after.fy, fcs[28].fy) and after.km_6mo == fcs[28].km_6mo
    assert after.now_km == pytest.approx(fcs[28].now_km + imb.recoverable_km)


def test_resistance_growth_costs_range_under_load_and_grows_with_severity():
    from cellsense.analytics.forecast import fault_effect

    p = CellParams.load()
    r = p.r0 + p.r1
    km = [fault_effect(Fault("resistance_growth", 29, s), 0.0, p, r).permanent_km for s in (0.2, 0.4, 0.6)]
    assert 0 < km[0] < km[1] < km[2] < 10  # a few km: the cell reaches cut-off a little earlier under load
