"""The four fault types CellSense is built to catch, and how each is injected."""
from __future__ import annotations

from dataclasses import dataclass

# severity meaning per kind:
#   charge_imbalance   fraction of capacity removed from the cell (SoC offset), 0.01-0.03
#   resistance_growth  fractional increase of internal resistance, 0.20-0.60
#   capacity_fade      fractional loss of capacity, 0.02-0.06
#   self_discharge     leakage as a fraction of capacity per hour (accelerated for demos), 0.02-0.10
SEVERITY_RANGES = {
    "charge_imbalance": (0.01, 0.03),
    "resistance_growth": (0.20, 0.60),
    "capacity_fade": (0.02, 0.06),
    "self_discharge": (0.02, 0.10),
}

EXPECTED_ACTION = {
    "charge_imbalance": "balance",
    "resistance_growth": "replace",
    "capacity_fade": "monitor",
    "self_discharge": "inspect",
}

LABELS = {
    "charge_imbalance": "Charge imbalance",
    "resistance_growth": "Resistance growth",
    "capacity_fade": "Capacity fade",
    "self_discharge": "Self-discharge",
}


@dataclass(frozen=True)
class Fault:
    kind: str
    cell: int  # 1-based cell number
    severity: float
    t_start: float = 0.0  # seconds into the session; 0 = present from the start

    def __post_init__(self):
        if self.kind not in SEVERITY_RANGES:
            raise ValueError(f"unknown fault kind {self.kind!r}; expected one of {list(SEVERITY_RANGES)}")
        if self.severity < 0:
            raise ValueError("severity must be >= 0")

    def describe(self) -> str:
        s = self.severity
        detail = {
            "charge_imbalance": f"SoC offset of {s * 100:.1f}%",
            "resistance_growth": f"+{s * 100:.0f}% internal resistance",
            "capacity_fade": f"-{s * 100:.1f}% capacity",
            "self_discharge": f"{s * 100:.0f}% of capacity leaking per hour",
        }[self.kind]
        return f"{LABELS[self.kind]} on cell {self.cell}: {detail}"
