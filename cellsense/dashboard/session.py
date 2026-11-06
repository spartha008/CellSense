"""Live session: runs the pack simulator in a background thread and streams it into the engine."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ..analytics import CellSenseEngine, Snapshot, forecast
from ..analytics.anomaly import AnomalyModel
from ..analytics.training import train_anomaly_model
from ..config import PACK
from ..sim import Fault, LiveDriveCycle, capacity_history, demo_pack, healthy_pack
from ..sim.ecm import CellParams


@dataclass
class Injection:
    fault: Fault
    t_inj: float


class LiveSession:
    """``background=True`` (local server): a thread advances the simulation continuously.
    ``background=False`` (serverless hosts such as Vercel, which freeze the process between requests):
    each read advances the simulation by the wall-clock time since the previous read, at most
    ``max_catchup_s`` simulated seconds at a time so a request after a long idle stays fast."""

    def __init__(self, mode: str = "demo", profile: str = "mixed", speed: float = 5.0, seed: int = 7, warmup_s: int = 900,
                 replay: str | None = None, background: bool = True, max_catchup_s: float = 120.0):
        self.lock = threading.RLock()
        self.background, self.max_catchup_s = background, float(max_catchup_s)
        self._last_wall, self._acc = time.perf_counter(), 0.0
        self.replay_path = replay
        self.replay = None
        if replay:  # play back a physics run (PyBaMM + liionpack) instead of the real-time simulator
            from ..sim.physics import load
            self.replay = load(replay)
            warmup_s = min(warmup_s, 300)
        self.params = CellParams.load()
        self.anomaly = AnomalyModel.load()
        if not self.anomaly.ready:  # first run: learn what a healthy pack looks like (~1 min)
            self.anomaly = train_anomaly_model()
        self.profile, self.speed, self.warmup_s = profile, float(speed), int(warmup_s)
        self.paused = False
        self.reset(mode, seed)
        self._stop = threading.Event()
        self.thread = None
        if background:
            self.thread = threading.Thread(target=self._loop, name="cellsense-sim", daemon=True)
            self.thread.start()

    # ------------------------------------------------------------------ control
    def reset(self, mode: str = "demo", seed: int | None = None):
        with self.lock:
            self.mode = "replay" if self.replay is not None else mode
            self.seed = int(seed if seed is not None else np.random.default_rng().integers(1, 10_000))
            self.paused = False
            if self.replay is not None:
                self.sim, self._ri, self._t = None, 0, 0.0
                self.hist = capacity_history(self.replay.capacity, now_cycle=320, seed=self.seed)
            else:
                self.sim, self.hist = (demo_pack if mode == "demo" else healthy_pack)(self.seed, self.params)
            self.engine = CellSenseEngine(PACK, params=self.params, anomaly=self.anomaly, capacity_ah=self.hist.capacity[-1])
            self.drive = LiveDriveCycle(self.profile, self.seed)
            self.injections: list[Injection] = []
            self.forecasts = forecast(self.hist)
            for _ in range(self.warmup_s):  # start with some history on screen
                self._tick()
            self.snap = self.engine.snapshot()
            self._last_wall, self._acc = time.perf_counter(), 0.0

    def inject(self, kind: str, cell: int, severity: float) -> Fault | None:
        if self.replay is not None:
            return None
        with self.lock:
            f = self.sim.inject_now(kind, int(cell), float(severity))
            self.engine.mark_injection(int(cell))
            self.injections.append(Injection(f, self.sim.t))
            return f

    def park(self, seconds: int = 240):
        with self.lock:
            self.drive.park(seconds)

    def set_speed(self, speed: float):
        self.speed = float(speed)

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------ loop
    def _tick(self):
        if self.replay is not None:
            r = self.replay
            if self._ri >= len(r.t):
                self.paused = True  # end of the recorded run
                return
            self._t += 1.0
            self.engine.update(self._t, float(r.current[self._ri]), r.voltage[self._ri])
            self._ri += 1
            return
        i = self.drive.next()
        v = self.sim.step(i)
        self.engine.update(self.sim.t, i, v)

    def _loop(self):
        last, acc, last_snap = time.perf_counter(), 0.0, 0.0
        while not self._stop.is_set():
            time.sleep(0.1)
            now = time.perf_counter()
            dt, last = now - last, now
            if self.paused:
                continue
            acc += dt * self.speed
            with self.lock:
                while acc >= 1.0:
                    self._tick()
                    acc -= 1.0
                if now - last_snap >= 0.5:
                    self.snap = self.engine.snapshot()
                    last_snap = now

    def _advance(self):
        """Serverless mode: catch the simulation up with the wall clock (bounded)."""
        with self.lock:
            now = time.perf_counter()
            dt, self._last_wall = now - self._last_wall, now
            if self.paused:
                return
            self._acc += dt * self.speed
            steps = int(min(self._acc, self.max_catchup_s))
            self._acc -= int(self._acc)  # drop whatever is beyond the cap: the sim just lags the clock
            for _ in range(steps):
                self._tick()
            if steps:
                self.snap = self.engine.snapshot()

    # ------------------------------------------------------------------ read
    def state(self) -> tuple[Snapshot, list[Injection]]:
        if not self.background:
            self._advance()
        with self.lock:
            return self.snap, list(self.injections)

    def detection(self, inj: Injection) -> dict:
        """How CellSense and a threshold-only BMS reacted to an injected fault."""
        s = self.snap
        i = inj.fault.cell - 1
        cs_t, bms_t = s.first_flag_t[i], s.bms_first_t[i]
        return {
            "fault": inj.fault, "elapsed": s.t - inj.t_inj,
            "cellsense_s": float(cs_t - inj.t_inj) if np.isfinite(cs_t) and cs_t >= inj.t_inj else None,
            "bms_s": float(bms_t - inj.t_inj) if np.isfinite(bms_t) and bms_t >= inj.t_inj else None,
            "finding": s.finding(inj.fault.cell),
        }
