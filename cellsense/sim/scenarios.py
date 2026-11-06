"""Ready-made packs: the demo pack from the pitch deck, and clean healthy packs."""
from __future__ import annotations

from ..config import PACK
from .ageing import capacity_history
from .ecm import CellParams, PackSimulator
from .faults import Fault

# The pack shown in the deck: Cell 47 degrading, Cell 12 low on charge, Cell 83 drifting early.
DEMO_CONDITIONS = (
    Fault("capacity_fade", 47, 0.0355),
    Fault("resistance_growth", 47, 0.38),
    Fault("charge_imbalance", 12, 0.024),
    Fault("capacity_fade", 83, 0.014),
    Fault("resistance_growth", 83, 0.10),
)


def demo_pack(seed: int = 7, params: CellParams | None = None, soc0: float = 0.78):
    sim = PackSimulator(PACK, params, seed=seed, soc0=soc0, conditions=list(DEMO_CONDITIONS))
    hist = capacity_history(sim.cap, now_cycle=320, seed=seed)
    return sim, hist


def healthy_pack(seed: int = 0, params: CellParams | None = None, soc0: float = 0.8):
    sim = PackSimulator(PACK, params, seed=seed, soc0=soc0)
    hist = capacity_history(sim.cap, now_cycle=320, seed=seed)
    return sim, hist
