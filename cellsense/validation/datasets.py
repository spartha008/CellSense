"""Check CellSense's ageing model and forecast against public battery-ageing data.

Supported datasets (download them yourself; they are not redistributed here):
  * NASA PCoE Battery Data Set (Saha & Goebel, 2007): B0005.mat, B0006.mat, B0007.mat, B0018.mat ...
    from the NASA Prognostics Center of Excellence data repository.
  * Oxford Battery Degradation Dataset 1 (Birkl, 2017): Oxford_Battery_Degradation_Dataset_1.mat
    from the University of Oxford research archive.

Two checks per cell:
  1. Fade shape: fit Q(c)/Q0 = 1 - a*sqrt(c) - b*c (the shape the simulator uses) and report R^2.
  2. Forecast: fit CellSense's quadratic trend on the first 60% of the history and predict when the
     cell reaches a fade level it actually reaches later; report the error in cycles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import PROJECT_DIR


@dataclass
class AgeingSeries:
    name: str
    cycles: np.ndarray
    capacity: np.ndarray  # Ah
    note: str = ""
    use: bool = True  # False = listed in the report but not used, with ``note`` saying why


# From the README files inside NASA's archive. Only experiments with one fixed discharge protocol for the
# whole test give a clean fade curve; the rest are listed with the reason they are left out.
NASA_PROTOCOL = {
    **{b: ("24 °C, 2 A constant-current discharge, run to 30% fade", True) for b in (5, 6, 7, 18)},
    **{b: ("24 °C, 4 A / 2 A constant-current discharge, run to 20% fade", True) for b in (33, 34, 36)},
    **{b: ("square-wave 4 A load, only 28 cycles", False) for b in (25, 26, 27, 28)},
    **{b: ("43 °C, only 40 cycles", False) for b in (29, 30, 31, 32)},
    **{b: ("load current switched between 1, 2 and 4 A during the test", False) for b in (38, 39, 40)},
    **{b: ("4 °C, load switched between 4 A and 1 A; NASA notes unexplained low-capacity runs", False) for b in (41, 42, 43, 44)},
    **{b: ("4 °C; NASA notes unexplained low-capacity runs", False) for b in (45, 46, 47, 48, 53, 54, 55, 56)},
    **{b: ("4 °C, stopped by a software crash; NASA notes unexplained low-capacity runs", False) for b in (49, 50, 51, 52)},
}


def clean(s: AgeingSeries, tol: float = 0.15, window: int = 15) -> AgeingSeries:
    """Drop start-up/conditioning discharges, aborted ones (<= 0.1 Ah) and partial ones (> tol from the running median)."""
    keep = s.capacity > 0.1
    c, q = s.cycles[keep], s.capacity[keep]
    if len(q) > 20:  # start once the cell first reaches 90% of its early best (skips conditioning cycles)
        start = int(np.argmax(q >= 0.9 * q[:20].max()))
        c, q = c[start:], q[start:]
    if len(q) >= window:
        pad = np.pad(q, window // 2, mode="edge")
        med = np.array([np.median(pad[i:i + window]) for i in range(len(q))])
        ok = np.abs(q - med) <= tol * med
        c, q = c[ok], q[ok]
    return AgeingSeries(s.name, c, q, s.note, s.use)


def extract_nasa(zip_path: str | Path, out_dir: str | Path) -> Path:
    """Pull the .mat files out of NASA's '5. Battery Data Set.zip' (which nests further zips).

    Only plain file names ending in .mat are written, flat into ``out_dir``; nothing else is extracted.
    """
    import io
    import zipfile

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def walk(zf: zipfile.ZipFile):
        for info in zf.infolist():
            name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
            if info.is_dir() or not name:
                continue
            if name.lower().endswith(".zip"):
                with zf.open(info) as f:
                    walk(zipfile.ZipFile(io.BytesIO(f.read())))
            elif name.lower().endswith(".mat") and ".." not in name and ":" not in name:
                with zf.open(info) as src, open(out / name, "wb") as dst:
                    dst.write(src.read())

    with zipfile.ZipFile(zip_path) as zf:
        walk(zf)
    return out


def load_nasa(folder: str | Path) -> list[AgeingSeries]:
    if str(folder).lower().endswith(".zip"):  # accept the downloaded zip directly
        folder = extract_nasa(folder, Path(folder).with_suffix(""))
    from scipy.io import loadmat

    out = []
    for f in sorted(Path(folder).glob("B*.mat")):
        m = loadmat(f)
        key = f.stem
        cycles = m[key][0, 0]["cycle"][0]
        caps = []
        for c in cycles:
            if str(c["type"][0]) == "discharge":
                cap = c["data"][0, 0]["Capacity"]
                if cap.size:
                    caps.append(float(cap.ravel()[0]))
        if len(caps) > 10:
            caps = np.array(caps)
            note, use = NASA_PROTOCOL.get(int(key[1:]), ("protocol not documented here", False))
            out.append(AgeingSeries(f"NASA {key}", np.arange(len(caps), dtype=float), caps, note, use))
    return out


def load_oxford(path: str | Path) -> list[AgeingSeries]:
    from scipy.io import loadmat

    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    out = []
    for k in sorted(x for x in m if x.lower().startswith("cell")):
        cell = m[k]
        names = sorted(n for n in cell._fieldnames if n.startswith("cyc"))
        cyc, cap = [], []
        for n in names:
            dc = getattr(getattr(cell, n), "C1dc", None)
            if dc is None:
                continue
            q = np.abs(np.asarray(dc.q, dtype=float))
            cyc.append(float(n[3:]))
            cap.append(q.max() / 1000.0)  # mAh -> Ah
        if len(cap) > 5:
            out.append(AgeingSeries(f"Oxford {k}", np.array(cyc), np.array(cap), "40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles"))
    return out


def fit_fade_shape(s: AgeingSeries) -> dict:
    """Least-squares fit of Q(c) = Q0 * (1 - a*sqrt(c) - b*c), with Q0 fitted too."""
    c = s.cycles - s.cycles[0]
    X = np.column_stack([np.ones_like(c), np.sqrt(c), c])
    beta, *_ = np.linalg.lstsq(X, s.capacity, rcond=None)
    q0 = float(beta[0])
    pred = X @ beta
    y = s.capacity
    ss_res, ss_tot = float(np.sum((y - pred) ** 2)), float(np.sum((y - y.mean()) ** 2))
    return {"a": float(-beta[1] / q0), "b": float(-beta[2] / q0), "q0": q0, "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "rmse_pct": float(np.sqrt(ss_res / len(y)) / q0 * 100), "final_fade_pct": float((1 - y[-1] / q0) * 100)}


def forecast_check(s: AgeingSeries, train_frac: float = 0.6) -> dict:
    """CellSense-style trend forecast on real data: predict when fade reaches a level seen later."""
    c = s.cycles - s.cycles[0]
    fade = (1 - s.capacity / s.capacity[:3].mean()) * 100
    if len(fade) > 10:  # running median: robust to the occasional partial discharge
        pad = np.pad(fade, 4, mode="edge")
        smooth = np.array([np.median(pad[i:i + 9]) for i in range(len(fade))])
    else:
        smooth = fade
    n_train = max(5, int(len(c) * train_frac))
    target = 0.5 * (smooth[n_train - 1] + smooth.max())  # a level reached after the training window
    hit = np.nonzero(smooth[n_train:] >= target)[0]
    if not len(hit):
        return {"target_fade_pct": float(target), "actual": None, "predicted": None, "error_cycles": None}
    actual = float(c[n_train + hit[0]])
    window = c[:n_train] >= c[n_train - 1] * 0.4
    coef = np.polyfit(c[:n_train][window], fade[:n_train][window], 2)
    if coef[0] < 0:
        coef = np.r_[0.0, np.polyfit(c[:n_train][window], fade[:n_train][window], 1)]
    grid = np.linspace(c[n_train - 1], c[-1] * 3, 4000)
    k = np.nonzero(np.polyval(coef, grid) >= target)[0]
    pred = float(grid[k[0]]) if len(k) else None
    return {"target_fade_pct": float(target), "actual": actual, "predicted": pred,
            "error_cycles": None if pred is None else pred - actual, "span_cycles": float(c[-1])}


def run_validation(nasa: str | None, oxford: str | None) -> str:
    series: list[AgeingSeries] = []
    if nasa:
        series += load_nasa(nasa)
    if oxford:
        series += load_oxford(oxford)
    if not series:
        msg = ("No dataset found. Download the NASA PCoE Battery Data Set (B0005.mat ...) and/or the Oxford Battery "
               "Degradation Dataset 1, then run:\n  python -m cellsense validate --nasa <folder or zip> --oxford <file.mat>")
        print(msg)
        return msg
    used = [clean(s) for s in series if s.use]
    left_out = [s for s in series if not s.use]
    rows = ["| Cell | Test conditions | Cycles | Final fade | Fade-shape fit R² | Fit RMSE | Forecast target | Actual cycle | Predicted | Error |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    stats: dict[str, dict[str, list]] = {}
    for s in used:
        f, fc = fit_fade_shape(s), forecast_check(s)
        e = fc["error_cycles"]
        st = stats.setdefault(s.name.split()[0], {"r2": [], "err": [], "rel": []})
        st["r2"].append(f["r2"])
        if e is not None:
            st["err"].append(e)
            st["rel"].append(abs(e) / max(fc["span_cycles"], 1))
        pred = f"{fc['predicted']:.0f}" if fc["predicted"] is not None else "–"
        act = f"{fc['actual']:.0f}" if fc["actual"] is not None else "–"
        rows.append(f"| {s.name} | {s.note} | {s.cycles[-1] - s.cycles[0]:.0f} | {f['final_fade_pct']:.1f}% | {f['r2']:.3f} | {f['rmse_pct']:.2f}% | "
                    f"{fc['target_fade_pct']:.1f}% | {act} | {pred} | {f'{e:+.0f}' if e is not None else '–'} |")
    summary = [f"**{len(used)} cells used.**", "", "| Dataset | Cells | Fade-shape fit R² (median, range) | Forecast error, median | Predicted early (warns before it happens) |",
               "|---|---|---|---|---|"]
    for src, st in stats.items():
        err = np.array(st["err"])
        summary.append(f"| {src} | {len(st['r2'])} | {np.median(st['r2']):.3f} ({min(st['r2']):.3f}–{max(st['r2']):.3f}) | "
                       + (f"{np.median(np.abs(err)):.0f} cycles ({np.median(st['rel']) * 100:.1f}% of test length) | {int(np.sum(err < 0))} of {len(err)} |" if len(err) else "– | – |"))
    excl = ["| Cell | Reason |", "|---|---|"] + [f"| {s.name} | {s.note} |" for s in left_out]
    text = "\n".join(["# Validation against public ageing data", "",
                      "Two checks per cell. **Fade shape**: least-squares fit of Q(c) = Q0·(1 − a·√c − b·c), the ageing model the simulator uses. "
                      "**Forecast**: CellSense's quadratic trend, fitted on the first 60% of each history, predicting the cycle at which the cell "
                      "reaches a fade level it only reaches later.", "",
                      "Cleaning: start-up conditioning discharges (before the cell first reaches 90% of its early best), aborted discharges (≤ 0.1 Ah) and partial discharges (> 15% from the running median of 15) are dropped.", "",
                      *summary, "", *rows, "",
                      "A negative error means the forecast flagged the cell before it actually got there. A trend fitted on early "
                      "history cannot anticipate a sudden late-life 'knee' (a cell whose fade abruptly accelerates); such cells are "
                      "forecast late.", "",
                      "Data: NASA PCoE Battery Data Set (B. Saha and K. Goebel, 2007, NASA Ames Prognostics Data Repository); Oxford "
                      "Battery Degradation Dataset 1 (C. Birkl, 2017, University of Oxford, ODbL; see also C. R. Birkl, 'Diagnosis and "
                      "Prognosis of Degradation in Lithium-Ion Batteries', PhD thesis, University of Oxford, 2017).", "",
                      "## Not used" if left_out else "", "",
                      "Selection follows the README files shipped with each dataset: only tests with one fixed discharge protocol for the "
                      "whole run give a clean fade curve." if left_out else "", "", *(excl if left_out else [])])
    out = PROJECT_DIR / "reports"
    out.mkdir(exist_ok=True)
    (out / "validation.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return text
