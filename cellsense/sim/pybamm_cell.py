"""The physics cell: a PyBaMM parameter set scaled up to the 150 Ah large-format cell of the pack.

Chen2020 describes a 5 Ah LG M50 (NMC811 / graphite-SiOx) cylindrical cell. Widening the electrodes
30x gives the same chemistry and voltage behaviour at 150 Ah, with resistances 30x lower.
"""
from __future__ import annotations

from ..config import PACK

BASE_CAPACITY_AH = 5.0
BASE_WIDTH_M = 1.58


def scaled_parameter_values(parameter_set: str = "Chen2020", capacity_ah: float = PACK.capacity_ah):
    import pybamm

    pv = pybamm.ParameterValues(parameter_set)
    k = capacity_ah / float(pv.get("Nominal cell capacity [A.h]", BASE_CAPACITY_AH))
    pv.update({
        "Electrode width [m]": float(pv["Electrode width [m]"]) * k,
        "Nominal cell capacity [A.h]": capacity_ah,
        "Current function [A]": capacity_ah,
        "Lower voltage cut-off [V]": 2.5,
        "Upper voltage cut-off [V]": 4.25,
    })
    return pv
