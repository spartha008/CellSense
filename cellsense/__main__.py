"""CellSense command line.

    python -m cellsense dashboard [--mode demo|healthy] [--speed 5] [--port 8050]
    python -m cellsense train                 # fit the Isolation Forest on healthy simulated packs
    python -m cellsense calibrate             # fit the real-time cell model to a PyBaMM SPMe cell
    python -m cellsense simulate ...          # physics-based 96-cell pack run with PyBaMM + liionpack
    python -m cellsense benchmark [--runs 200]
    python -m cellsense validate --nasa DIR --oxford FILE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _number(kind, minimum, inclusive=False):
    """argparse type: a number above ``minimum`` (or at least ``minimum`` when ``inclusive``)."""
    def check(text):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if value < minimum or (value == minimum and not inclusive):
            raise argparse.ArgumentTypeError(f"must be {'at least' if inclusive else 'greater than'} {minimum}: {text!r}")
        return value
    return check


def _port(text):
    port = _number(int, 1, inclusive=True)(text)
    if port > 65535:
        raise argparse.ArgumentTypeError(f"must be 1-65535: {text!r}")
    return port


def _existing(text):
    if not Path(text).exists():
        raise argparse.ArgumentTypeError(f"not found: {text}")
    return text


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252, which cannot print "−", "Ω", "²"
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    positive_int, positive_float, count = _number(int, 0), _number(float, 0.0), _number(int, 0, inclusive=True)
    ap = argparse.ArgumentParser(prog="cellsense", description="CellSense: predictive cell imbalance detection for EV battery packs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dashboard", help="run the live dashboard")
    d.add_argument("--mode", choices=["demo", "healthy"], default="demo")
    d.add_argument("--speed", type=positive_float, default=5.0, help="simulated seconds per real second")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=_port, default=8050)
    d.add_argument("--replay", type=_existing, help="play back a physics run saved by `simulate` (.npz) instead of the live simulator")

    t = sub.add_parser("train", help="train the Isolation Forest on healthy packs")
    t.add_argument("--packs", type=positive_int, default=30)

    c = sub.add_parser("calibrate", help="fit the real-time ECM to a PyBaMM SPMe cell")
    c.add_argument("--parameter-set", default="Chen2020")

    s = sub.add_parser("simulate", help="physics-based pack simulation (PyBaMM + liionpack)")
    s.add_argument("--profile", default="mixed", choices=["city", "highway", "mixed"])
    s.add_argument("--duration", type=positive_int, default=600, help="seconds")
    s.add_argument("--fault", action="append", default=[], help="kind:cell:severity, e.g. resistance_growth:29:0.35 (present from the start of the run)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cells", type=positive_int, default=96)
    s.add_argument("--out", default="data/runs/liionpack_run.npz")
    s.add_argument("--analyse", action="store_true", help="run the CellSense engine over the result")

    b = sub.add_parser("benchmark", help="randomised fault benchmark: delay, precision/recall, diagnosis, forecast")
    b.add_argument("--runs", type=positive_int, default=200)
    b.add_argument("--healthy", type=count, default=40)
    b.add_argument("--duration", type=positive_int, default=1800)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--workers", type=count, default=0, help="processes (0 = one per CPU thread, less one)")

    v = sub.add_parser("validate", help="check the ageing model and forecast against NASA PCoE / Oxford data")
    v.add_argument("--nasa", type=_existing, help="folder with NASA PCoE battery .mat files (B0005.mat ...), or NASA's zip")
    v.add_argument("--oxford", type=_existing, help="path to Oxford_Battery_Degradation_Dataset_1.mat")

    a = ap.parse_args(argv)
    if a.cmd in ("calibrate", "simulate"):
        import importlib.util

        missing = [m for m in ("pybamm", "liionpack") if importlib.util.find_spec(m) is None]
        if missing:
            ap.exit(1, f"`{a.cmd}` needs the physics stack ({', '.join(missing)} not installed): pip install -r requirements-full.txt\n")
    if a.cmd == "dashboard":
        from .dashboard import run
        try:
            run(a.host, a.port, a.mode, a.speed, replay=a.replay)
        except ValueError as e:  # e.g. a replay file with the wrong number of cells
            ap.exit(1, f"cellsense dashboard: {e}\n")
    elif a.cmd == "train":
        from .analytics.training import train_anomaly_model
        m = train_anomaly_model(n_packs=a.packs)
        print(f"saved Isolation Forest (alert threshold {m.threshold:.3f}) to cellsense/data/iforest.joblib")
    elif a.cmd == "calibrate":
        from .sim.calibrate import calibrate
        p = calibrate(a.parameter_set)
        print(f"saved ECM fitted to {p.source}: R0={p.r0 * 1000:.3f} mΩ, R1={p.r1 * 1000:.3f} mΩ, tau={p.tau:.1f} s")
    elif a.cmd == "simulate":
        from .sim.physics import run_cli
        run_cli(a)
    elif a.cmd == "benchmark":
        from .validation.benchmark import run_benchmark
        run_benchmark(a.runs, a.healthy, a.duration, a.seed, a.workers)
    elif a.cmd == "validate":
        from .validation.datasets import run_validation
        run_validation(a.nasa, a.oxford)
    return 0


if __name__ == "__main__":
    sys.exit(main())
