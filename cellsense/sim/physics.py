"""Physics-based pack simulation: every cell is a PyBaMM SPMe model, wired in series by liionpack.

Cell-to-cell variation and faults are applied per cell:
  capacity spread / capacity fade -> active-material volume fraction of both electrodes (PyBaMM input)
  charge imbalance / SoC spread   -> initial electrode concentrations (PyBaMM input)
  resistance spread / growth      -> a series tab resistance per cell, which the BMS cell-voltage tap
                                     includes (V_measured = V_cell - I_cell * R_tab; in a series string it
                                     does not change any cell's current, so it is applied to the measurement)
Self-discharge needs a leakage path inside the cell, which SPMe does not model; use the ECM backend
for that fault. Faults here are present from the start of the run (offline datasets).

The result has the same shape as the real-time simulator's, so the CellSense engine runs on either.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PACK, PROJECT_DIR
from .ecm import CellParams, SimResult, Spread
from .faults import Fault
from .pybamm_cell import scaled_parameter_values

AM_NEG = "Negative electrode active material volume fraction"
AM_POS = "Positive electrode active material volume fraction"
C0_NEG = "Initial concentration in negative electrode [mol.m-3]"
C0_POS = "Initial concentration in positive electrode [mol.m-3]"


def _sim_func(parameter_values):
    import liionpack as lp
    import pybamm

    model = lp.add_events_to_model(pybamm.lithium_ion.SPMe())
    return pybamm.Simulation(model, parameter_values=parameter_values, solver=pybamm.CasadiSolver(mode="safe"))


def simulate_pack(current: np.ndarray, faults: list[Fault] | tuple = (), n_cells: int = PACK.n_series, soc0: float = 0.8,
                  seed: int = 0, spread: Spread = Spread(), noise_v: float = 0.0008, adc_lsb_v: float = 0.001, nproc: int = 1) -> SimResult:
    import liionpack as lp
    import pybamm

    if any(f.kind == "self_discharge" for f in faults):
        raise ValueError("self_discharge is not modelled by SPMe; use the ECM backend for it")
    if any(not 1 <= f.cell <= n_cells for f in faults):
        raise ValueError("fault cell outside the simulated pack")
    rng = np.random.default_rng(seed)
    g = lambda: rng.standard_normal(n_cells)  # noqa: E731
    ecm = CellParams.load()
    pv = scaled_parameter_values()
    k = 1 + spread.capacity * g()  # relative capacity
    r_tab = np.clip(0.05e-3 + spread.resistance * ecm.r0 * g(), 0.0, None)  # ohm, part of what the BMS tap measures
    soc_off = spread.soc * g()
    for f in faults:
        i = f.cell - 1
        if f.kind == "capacity_fade":
            k[i] *= 1 - f.severity
        elif f.kind == "resistance_growth":
            r_tab[i] += f.severity * ecm.r0
        elif f.kind == "charge_imbalance":
            soc_off[i] -= f.severity
    cap = PACK.capacity_ah * k
    soc = np.clip(1 - (1 - soc0) * PACK.capacity_ah / cap + soc_off, 0.02, 0.99)  # top-balanced pack
    cn_max = float(pv["Maximum concentration in negative electrode [mol.m-3]"])
    cp_max = float(pv["Maximum concentration in positive electrode [mol.m-3]"])
    x, y = zip(*(lp.get_initial_stoichiometries(float(s), pv) for s in soc))
    # NOTE: liionpack maps per-cell inputs by position, so this order must match the order in which the
    # model meets the input parameters (initial concentrations before active-material fractions)
    inputs = {
        C0_NEG: np.array(x) * cn_max,
        C0_POS: np.array(y) * cp_max,
        AM_NEG: float(pv[AM_NEG]) * k,
        AM_POS: float(pv[AM_POS]) * k,
    }
    pv.update({name: "[input]" for name in inputs}, check_already_exists=False)
    current = np.asarray(current, dtype=float)
    netlist = lp.setup_circuit(Np=1, Ns=n_cells, Rb=1e-6, Rc=1e-6, Ri=ecm.r0, V=float(ecm.ocv_at(soc0)), I=float(current[0]))
    # one constant-current step per second (liionpack mishandles interpolated drive-cycle steps here)
    experiment = pybamm.Experiment([pybamm.step.current(float(i), duration=1, period=1) for i in current], period="1 second")
    out = lp.solve(netlist=netlist, sim_func=_sim_func, parameter_values=pv, experiment=experiment, inputs=inputs,
                   output_variables=["Voltage [V]"], initial_soc=None, nproc=nproc, manager="casadi")
    # liionpack reports the voltage at the end of each step against the current applied in the next one:
    # V[k + 1] is the response to I[k]. Re-align so each row is (current, resulting voltage).
    I_cell = np.asarray(out["Cell current [A]"])[:-1]
    V = np.asarray(out["Voltage [V]"])[1:] - I_cell * r_tab[None, :]
    tt = np.asarray(out["Time [s]"]).ravel()[1:]
    I = np.asarray(out["Pack current [A]"]).ravel()[:-1]
    V = V + noise_v * rng.standard_normal(V.shape)  # the BMS sees noisy, quantised measurements
    if adc_lsb_v:
        V = np.round(V / adc_lsb_v) * adc_lsb_v
    return SimResult(tt, I, V, list(faults), cap, None, None, backend="liionpack")


def save(res: SimResult, path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=res.t, current=res.current, voltage=res.voltage, capacity=res.capacity,
                        faults=np.array([f"{f.kind}:{f.cell}:{f.severity}" for f in res.faults]), backend=res.backend)
    return path


def load(path: str | Path) -> SimResult:
    d = np.load(path, allow_pickle=False)
    faults = [Fault(k, int(c), float(s)) for k, c, s in (x.split(":") for x in d["faults"])]
    return SimResult(d["t"], d["current"], d["voltage"], faults, d["capacity"], backend=str(d["backend"]))


def analyse(res: SimResult, n_print: int = 6):
    from dataclasses import replace

    from ..analytics import CellSenseEngine

    n = res.voltage.shape[1]  # --cells can differ from the 96-cell default
    eng = CellSenseEngine(replace(PACK, n_series=n, modules=1, cells_per_module=n), capacity_ah=res.capacity)
    for t, i, v in zip(res.t, res.current, res.voltage):
        eng.update(float(t), float(i), v)
    s = eng.snapshot()
    print(f"engine on {res.backend} data: range lost {s.impact.lost_km:.1f} km (limiting cell {s.impact.limiting_cell}), {len(s.findings)} findings")
    for f in s.findings[:n_print]:
        print(f"  cell {f.cell:>2} {f.status:<8} {f.kind:<18} {f.headline}: {f.detail}")
    return s


def run_cli(a):
    import time

    from .drive_cycles import drive_cycle

    faults = []
    for spec in a.fault:
        parts = spec.split(":")
        try:
            if len(parts) != 3:
                raise ValueError("expected kind:cell:severity, e.g. resistance_growth:29:0.35")
            f = Fault(parts[0], int(parts[1]), float(parts[2]))
            if not 1 <= f.cell <= a.cells:
                raise ValueError(f"cell must be between 1 and {a.cells}")
            if f.kind == "self_discharge":
                raise ValueError("self_discharge is not modelled by SPMe; inject it in the dashboard's fault lab instead")
        except ValueError as e:
            raise SystemExit(f"cellsense simulate: bad --fault {spec!r}: {e}") from None
        faults.append(f)
    cur = drive_cycle(a.profile, a.duration, a.seed)
    t0 = time.time()
    print(f"liionpack: {a.cells} SPMe cells in series, {a.duration} s {a.profile} drive, faults: {[f.describe() for f in faults] or 'none'}")
    res = simulate_pack(cur, faults, n_cells=a.cells, seed=a.seed)
    print(f"solved in {time.time() - t0:.0f} s -> {save(res, a.out)}")
    if a.analyse:
        analyse(res)
