"""Real-time pack simulator: one equivalent-circuit model (OCV + R0 + one RC pair) per cell,
vectorised over all 96 cells so the dashboard can step it live.

Cell parameters come from ``data/ecm_params.json``. Run ``python -m cellsense calibrate`` to
regenerate that file from a PyBaMM SPMe model of the cell; without it a built-in NMC curve is used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, PACK, PackConfig
from .faults import Fault

PARAMS_FILE = DATA_DIR / "ecm_params.json"


def _default_ocv(soc: np.ndarray) -> np.ndarray:
    soc = np.asarray(soc, dtype=float)
    return 3.45 + 0.62 * soc + 0.13 * soc**3 - 0.08 * np.exp(-20 * soc)


@dataclass
class CellParams:
    soc: np.ndarray
    ocv: np.ndarray
    r0: float  # ohm, instantaneous resistance
    r1: float  # ohm, polarisation resistance
    tau: float  # s, RC time constant
    capacity_ah: float
    source: str = "built-in NMC curve"

    def __post_init__(self):
        self.soc = np.asarray(self.soc, dtype=float)
        # OCV must be strictly increasing so it can be inverted
        ocv = np.maximum.accumulate(np.asarray(self.ocv, dtype=float))
        self.ocv = ocv + np.arange(len(ocv)) * 1e-7

    def ocv_at(self, soc):
        return np.interp(np.clip(soc, 0.0, 1.0), self.soc, self.ocv)

    def soc_at(self, voltage):
        return np.interp(voltage, self.ocv, self.soc)

    @classmethod
    def load(cls, path: Path | None = None) -> "CellParams":
        path = Path(path) if path else PARAMS_FILE
        if path.exists():
            d = json.loads(path.read_text())
            return cls(np.array(d["soc"]), np.array(d["ocv"]), d["r0_ohm"], d["r1_ohm"], d["tau_s"], d["capacity_ah"], d.get("source", str(path)))
        s = np.linspace(0, 1, 101)
        return cls(s, _default_ocv(s), 0.85e-3, 0.51e-3, 30.0, PACK.capacity_ah)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else PARAMS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "source": self.source, "capacity_ah": self.capacity_ah, "r0_ohm": self.r0, "r1_ohm": self.r1, "tau_s": self.tau,
            "soc": [round(float(x), 6) for x in self.soc], "ocv": [round(float(x), 6) for x in self.ocv],
        }, indent=1))
        return path


@dataclass(frozen=True)
class Spread:
    """Cell-to-cell manufacturing spread (1 sigma, relative)."""

    capacity: float = 0.0035
    resistance: float = 0.035
    soc: float = 0.0022


@dataclass
class SimResult:
    t: np.ndarray  # (T,) seconds
    current: np.ndarray  # (T,) amps, + = discharge
    voltage: np.ndarray  # (T, N) measured cell voltages
    faults: list[Fault] = field(default_factory=list)
    capacity: np.ndarray | None = None  # (N,) true capacities at the end
    r0: np.ndarray | None = None  # (N,) true R0 at the end
    soc: np.ndarray | None = None  # (T, N) true SoC (simulation only)
    backend: str = "ecm"


class PackSimulator:
    """Series pack of equivalent-circuit cells.

    ``conditions`` are pre-existing cell issues applied before the pack is top-balanced and charged;
    faults passed to :meth:`schedule` are injected live when the simulation clock reaches ``t_start``.
    """

    def __init__(self, cfg: PackConfig = PACK, params: CellParams | None = None, seed: int = 0, soc0: float = 0.8,
                 spread: Spread = Spread(), conditions: list[Fault] | tuple = (), temperature_c: float = 25.0,
                 noise_v: float = 0.0008, adc_lsb_v: float = 0.001):
        self.cfg = cfg
        self.p = params or CellParams.load()
        self.rng = np.random.default_rng(seed)
        n = cfg.n_series
        g = self.rng.standard_normal
        self.cap = self.p.capacity_ah * (1 + spread.capacity * g(n))
        rr = 1 + spread.resistance * g(n)
        self.r0 = self.p.r0 * rr
        self.r1 = self.p.r1 * rr
        self.tau = np.full(n, self.p.tau)
        self.leak = np.zeros(n)
        self.noise_v, self.adc = noise_v, adc_lsb_v
        self.temperature_c = temperature_c
        self.t = 0.0
        self.conditions = list(conditions)
        for f in self.conditions:
            if f.kind in ("resistance_growth", "capacity_fade", "self_discharge"):
                self._apply(f)
        # top-balanced pack: every cell has had the same charge removed from full
        removed = (1 - soc0) * cfg.capacity_ah
        self.q = np.clip(self.cap - removed + spread.soc * cfg.capacity_ah * g(n), 0, None)
        for f in self.conditions:
            if f.kind == "charge_imbalance":
                self._apply(f)
        self.v1 = np.zeros(n)
        self.pending: list[Fault] = []
        self.injected: list[Fault] = []

    # ---- faults -------------------------------------------------------------------------
    def _apply(self, f: Fault):
        i = f.cell - 1
        if f.kind == "charge_imbalance":
            self.q[i] = max(0.0, self.q[i] - f.severity * self.cap[i])
        elif f.kind == "resistance_growth":
            self.r0[i] *= 1 + f.severity
            self.r1[i] *= 1 + f.severity
        elif f.kind == "capacity_fade":
            if hasattr(self, "q"):
                self.q[i] *= 1 - f.severity
            self.cap[i] *= 1 - f.severity
        elif f.kind == "self_discharge":
            self.leak[i] += f.severity * self.cap[i]  # fraction per hour x Ah = amps

    def schedule(self, fault: Fault):
        self.pending.append(fault)

    def inject_now(self, kind: str, cell: int, severity: float) -> Fault:
        f = Fault(kind, cell, severity, self.t)
        self.schedule(f)
        return f

    # ---- dynamics -----------------------------------------------------------------------
    def soc(self) -> np.ndarray:
        return np.clip(self.q / self.cap, 0.0, 1.0)

    def step(self, current: float, dt: float = 1.0) -> np.ndarray:
        """Advance ``dt`` seconds at pack ``current`` (A, + = discharge); return measured cell voltages."""
        due = [f for f in self.pending if f.t_start <= self.t + 1e-9]
        for f in due:
            self._apply(f)
            self.pending.remove(f)
            self.injected.append(f)
        k = 1 + 0.015 * (25.0 - self.temperature_c)  # resistance rises in the cold, pack-wide
        a = np.exp(-dt / self.tau)
        self.v1 = self.v1 * a + current * self.r1 * k * (1 - a)
        self.q = np.clip(self.q - (current + self.leak) * dt / 3600.0, 0.0, self.cap)
        v = self.p.ocv_at(self.soc()) - current * self.r0 * k - self.v1
        v = v + self.noise_v * self.rng.standard_normal(v.shape)
        if self.adc:
            v = np.round(v / self.adc) * self.adc
        self.t += dt
        return v

    def run(self, current: np.ndarray, dt: float = 1.0, faults: list[Fault] | tuple = (), keep_soc: bool = False) -> SimResult:
        for f in faults:
            self.schedule(f)
        current = np.asarray(current, dtype=float)
        T, n = len(current), self.cfg.n_series
        V = np.empty((T, n))
        S = np.empty((T, n)) if keep_soc else None
        t = np.empty(T)
        for k, i in enumerate(current):
            V[k] = self.step(float(i), dt)
            t[k] = self.t
            if keep_soc:
                S[k] = self.soc()
        return SimResult(t, current, V, list(self.conditions) + list(self.injected), self.cap.copy(), self.r0.copy(), S)
