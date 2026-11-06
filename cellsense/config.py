"""Pack definition and every decision threshold CellSense uses, in one place."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
PROJECT_DIR = PACKAGE_DIR.parent


@dataclass(frozen=True)
class PackConfig:
    """A 96s1p NMC pack: 96 large-format cells in series, 8 modules of 12."""

    n_series: int = 96
    modules: int = 8
    cells_per_module: int = 12
    capacity_ah: float = 150.0
    chemistry: str = "NMC"
    energy_kwh: float = 52.0
    rated_range_km: float = 350.0
    v_min: float = 2.8  # hard cell limits a conventional BMS enforces
    v_max: float = 4.2

    def module_of(self, cell: int) -> int:
        return (cell - 1) // self.cells_per_module + 1

    def position_of(self, cell: int) -> int:
        return (cell - 1) % self.cells_per_module + 1

    def label(self, cell: int) -> str:
        return f"M{self.module_of(cell)} · P{self.position_of(cell)}"


@dataclass(frozen=True)
class Thresholds:
    # Detect: robust z-score against the pack median
    z_alert: float = 3.0
    z_window_s: int = 30
    persistence_s: int = 10
    mad_floor_v: float = 0.0025  # ~ cell-voltage measurement accuracy of a production BMS
    # Detect: Isolation Forest anomaly score (paper definition, 0..1)
    if_alert: float = 0.65
    # Diagnose: dV/dI internal resistance at current steps
    step_di_a: float = 60.0
    r_replace_ratio: float = 1.20
    r_monitor_ratio: float = 1.08
    # Diagnose: offsets (all relative to the pack median)
    rest_current_a: float = 8.0
    load_current_a: float = 120.0
    balance_offset_mv: float = 8.0
    balance_max_sag_mv: float = 6.0
    # charge drift relative to the pack, in %SoC: per Ah discharged (capacity fade) and per hour (self-discharge)
    fade_slope_pct_per_ah: float = -0.010  # ~1.5% less capacity than the median cell
    self_discharge_pct_per_h: float = -3.0
    min_fade_ah: float = 12.0  # charge that must flow before capacity fade is judged
    min_parked_s: float = 300.0  # settled parked data needed before self-discharge is judged
    # Conventional BMS baseline (what we compare against)
    bms_rest_dv: float = 0.050
    # Translate + forecast
    replace_km: float = 20.0
    cycles_per_month: float = 25.0


PACK = PackConfig()
THRESHOLDS = Thresholds()
