"""Fit the real-time equivalent-circuit model to a PyBaMM SPMe cell.

1. Open-circuit voltage curve: the model's OCV variable during a slow C/20 discharge, vs SoC.
2. R0, R1, tau: a 1C, 30 s pulse at 50% SoC; R0 from the instantaneous drop, R1 and tau from the
   relaxation afterwards.
The result is written to ``cellsense/data/ecm_params.json`` and used by the live simulator and engine.
"""
from __future__ import annotations

import numpy as np

from ..config import PACK
from .ecm import CellParams
from .pybamm_cell import scaled_parameter_values


def _var(sol, *names):
    for n in names:
        try:
            return sol[n].entries
        except KeyError:
            continue
    raise KeyError(f"none of {names} in solution")


def ocv_curve(pv, points: int = 101):
    import pybamm

    model = pybamm.lithium_ion.SPMe()
    exp = pybamm.Experiment(["Discharge at C/20 until 2.6 V"], period="5 minutes")
    sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp, solver=pybamm.CasadiSolver(mode="safe")).solve(initial_soc=1.0)
    q = _var(sol, "Discharge capacity [A.h]")
    ocv = _var(sol, "Battery open-circuit voltage [V]", "Open-circuit voltage [V]")
    soc_pts = 1.0 - q / q[-1]
    order = np.argsort(soc_pts)
    s = np.linspace(0, 1, points)
    return s, np.interp(s, soc_pts[order], ocv[order]), float(q[-1])


def pulse_fit(pv, capacity_ah: float):
    import pybamm
    from scipy.optimize import curve_fit

    model = pybamm.lithium_ion.SPMe()
    exp = pybamm.Experiment([
        pybamm.step.rest(60, period=1),
        pybamm.step.current(capacity_ah, duration=30, period=0.1),
        pybamm.step.rest(300, period=1),
    ])
    sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp, solver=pybamm.CasadiSolver(mode="safe")).solve(initial_soc=0.5)
    t = _var(sol, "Time [s]")
    v = _var(sol, "Voltage [V]", "Terminal voltage [V]")
    I = capacity_ah
    v_rest = v[np.searchsorted(t, 59.9)]
    k0 = np.searchsorted(t, 60.0 + 0.1)  # first sample inside the pulse
    r0 = (v_rest - v[k0]) / I
    end = np.searchsorted(t, 90.0)
    tr, vr = t[end + 1:] - t[end], v[end + 1:]
    f = lambda x, vinf, a, tau: vinf - a * np.exp(-x / tau)  # noqa: E731
    (vinf, a, tau), _ = curve_fit(f, tr, vr, p0=[vr[-1], vr[-1] - vr[0], 20.0], maxfev=20000)
    r1 = a / (I * (1 - np.exp(-30.0 / tau)))
    return float(r0), float(abs(r1)), float(abs(tau))


def calibrate(parameter_set: str = "Chen2020", save: bool = True) -> CellParams:
    pv = scaled_parameter_values(parameter_set)
    soc, ocv, cap = ocv_curve(pv)
    r0, r1, tau = pulse_fit(pv, PACK.capacity_ah)
    p = CellParams(soc, ocv, r0, r1, tau, PACK.capacity_ah, source=f"PyBaMM SPMe ({parameter_set}, scaled to {PACK.capacity_ah:.0f} Ah)")
    if save:
        p.save()
    return p
