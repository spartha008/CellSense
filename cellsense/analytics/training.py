"""Train the Isolation Forest on healthy simulated packs (unsupervised: no fault labels used)."""
from __future__ import annotations

from ..sim.drive_cycles import PROFILES, drive_cycle
from ..sim.scenarios import healthy_pack
from .anomaly import AnomalyModel, feature_matrix
from .engine import CellSenseEngine


def healthy_features(n_packs: int = 30, duration_s: int = 1800, seed: int = 1000):
    packs = []
    for k in range(n_packs):
        sim, hist = healthy_pack(seed + k)
        eng = CellSenseEngine(anomaly=AnomalyModel(None), capacity_ah=hist.capacity[-1], eval_every_s=1e12)
        for i in drive_cycle(PROFILES[k % len(PROFILES)], duration_s, seed + k):
            eng.update(sim.t + 1, i, sim.step(i))
        packs.append(feature_matrix(eng.features()))
    return packs


def train_anomaly_model(n_packs: int = 30, duration_s: int = 1800, seed: int = 1000, save: bool = True) -> AnomalyModel:
    model = AnomalyModel().fit(healthy_features(n_packs, duration_s, seed), seed=seed)
    if save:
        model.save()
    return model
