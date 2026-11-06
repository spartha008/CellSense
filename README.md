# CellSense

**Predictive cell imbalance detection for EV battery packs.** Team BitStorm, problem statement #5.

CellSense compares every cell of a 96-cell pack against the rest of the pack, in real time, and turns what it finds into
plain actions: which cell is weak, whether it needs **balancing** or **replacing**, when it will become the pack's
limiting cell, and how many **km of range** it is costing the driver.

Everything runs in software: a physics-based pack model (PyBaMM + liionpack) generates the data, and a real-time
cell model fitted to it drives the live dashboard and the fault lab.

## Quick start (Windows, Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-full.txt
.venv\Scripts\python -m cellsense dashboard
```

`requirements.txt` is the slim dashboard runtime (what a web host installs). `requirements-full.txt` adds the
research stack: PyBaMM + liionpack (physics backend, `calibrate`, `simulate`), pandas, and pytest.

Open http://127.0.0.1:8050. The first start trains the anomaly model (about a minute) if
`cellsense/data/iforest.joblib` is missing.

> The versions in `requirements-full.txt` are pinned on purpose. On Windows 11 with Smart App Control, newer builds of
> pandas (3.x) and PyBaMM's solver package (pybammsolvers 0.10+) are blocked from loading.

## Deploying the dashboard (Vercel)

`app.py` at the repo root is the WSGI entrypoint: it builds the dashboard and exposes the Flask server as `app`,
which Vercel's Python runtime serves. Vercel installs only `requirements.txt` (about 340 MB, inside the 500 MB
Python function limit) and `.vercelignore` keeps decks, reports and tests out of the bundle.

Serverless functions are frozen between requests, so on Vercel the simulation advances on each request (by the
time elapsed since the last one, at most 2 simulated minutes per request) instead of in a background thread.
Each function instance keeps its own pack, so if Vercel runs several instances at once, viewers can see different
packs and an injected fault only shows on the instance that received it. For a single always-on demo, a host that
runs a long-lived process (e.g. `python -m cellsense dashboard --host 0.0.0.0`) avoids that.

## What is in the box

| Deck claim | Where it lives |
|---|---|
| **Detect**: rolling z-scores vs the pack median + Isolation Forest | `cellsense/analytics/engine.py` (robust median/MAD z, 30 s window, 10 s persistence), `analytics/anomaly.py` |
| **Diagnose**: ΔV/ΔI internal resistance → balance vs replace | `engine.py` (resistance from current steps), `analytics/diagnosis.py` (rules) |
| **Forecast**: which cell limits the pack next, and when | `analytics/forecast.py` (quadratic trend + 95% band per cell) |
| **Translate**: range lost in km | `analytics/range_impact.py` |
| PyBaMM + liionpack 96-cell pack with injected faults | `sim/physics.py` (`python -m cellsense simulate`) |
| Real-time simulator for the live demo, fitted to PyBaMM | `sim/ecm.py`, `sim/calibrate.py` (`python -m cellsense calibrate`) |
| Faults: charge imbalance, resistance growth, capacity fade, self-discharge | `sim/faults.py` |
| Dashboard: heatmap, alerts, diagnostics, forecast, live fault lab | `cellsense/dashboard/` (Plotly Dash) |
| Benchmark: detection delay, precision/recall, diagnosis accuracy, forecast error | `validation/benchmark.py` → `reports/benchmark.md` |
| Validation against NASA PCoE and Oxford ageing data | `validation/datasets.py` → `reports/validation.md` |

## Commands

```bash
python -m cellsense dashboard [--mode demo|healthy] [--speed 5] [--port 8050]
python -m cellsense dashboard --replay data/runs/liionpack_demo_96.npz   # play back a PyBaMM + liionpack run
python -m cellsense calibrate            # fit the real-time cell model to a PyBaMM SPMe cell (Chen2020, scaled to 150 Ah)
python -m cellsense train                # re-fit the Isolation Forest on healthy simulated packs
python -m cellsense simulate --cells 96 --duration 900 --fault resistance_growth:47:0.38 --analyse
python -m cellsense benchmark --runs 200 --healthy 40
python -m cellsense validate --nasa <folder with B0005.mat ...> --oxford <Oxford_Battery_Degradation_Dataset_1.mat>
python -m pytest
```

## Results (simulated, see `reports/benchmark.md`)

200 randomised fault injections + 40 healthy packs, 96 cells each, a 30-minute drive then 10 minutes parked:

| Fault | Detected | Median time to detect | Right action |
|---|---|---|---|
| Charge imbalance (1–3% SoC) | 98% | 24 s | 94% |
| Resistance growth (+20–60%) | 100% | 21 s | 92% |
| Capacity fade (2–6%) | 88% | ~10 min | 70% |
| Self-discharge | 100% | ~4 min | 86% |

A threshold-only BMS (2.8–4.2 V window, 50 mV rest-imbalance limit) caught 1% of the same faults. Precision against
the simulator's ground truth is 89% (0.11 false alarms per 96-cell pack); the forecast predicts when a degrading
cell crosses the 20 km threshold to within ~13 cycles. Re-run with `python -m cellsense benchmark`.

## Validation on real ageing data (see `reports/validation.md`)

| Dataset | Cells | Fade-shape fit R² (median) | Forecast error (median) | Forecast warned early |
|---|---|---|---|---|
| NASA PCoE (18650, 24 °C, constant-current tests) | 7 | 0.952 | 15 cycles (8.6% of test length) | 7 of 7 |
| Oxford Dataset 1 (Kokam pouch, 40 °C, Artemis drive cycle) | 8 | 0.996 | 378 cycles (5.3% of test length) | 6 of 8 |

The √cycle + linear fade model the simulator uses fits real cells well, and the forecaster warns early in 13 of 15
cells. It cannot anticipate a sudden late-life knee (Oxford Cell 5 was forecast late). Cells from NASA tests that
changed protocol mid-run, or that NASA flags for unexplained low-capacity runs, are listed but not used.

To reproduce, download the data from the publishers (not redistributed here) into `data/external/`:

* NASA: `https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip` (210 MB), linked from the
  [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
* Oxford: `Oxford_Battery_Degradation_Dataset_1.mat` (266 MB) from the
  [Oxford Research Archive record](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) (ODbL)

```bash
python -m cellsense validate --nasa "data/external/nasa_zip/5_Battery_Data_Set.zip" --oxford data/external/oxford/Oxford_Battery_Degradation_Dataset_1.mat
```

If you use these datasets, cite them: B. Saha and K. Goebel (2007), "Battery Data Set", NASA Prognostics Data
Repository; C. Birkl (2017), Oxford Battery Degradation Dataset 1, University of Oxford, and C. R. Birkl,
"Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries", PhD thesis, University of Oxford, 2017.

## How it works

```
 pack data ──► per sample (1 Hz) ──────────────────────────► every 5 s ─────────────────────► dashboard
 96 cell V,    · deviation from pack median, robust z-score    · Isolation Forest score         · heatmap, alerts
 pack current  · ΔV/ΔI resistance at current steps ≥ 60 A      · diagnosis rules                · cell diagnostics
               · offsets at rest / under load                  · range lost in km               · forecast
               · charge offset with IR + RC drop removed       · capacity forecast (per cycle)  · fault lab
               · coulomb-counted SoC per cell + OCV observer
               · threshold-BMS baseline for comparison
```

**Telling faults apart.** All four faults pull a cell below the pack, but each leaves a different fingerprint:

| Fault | Signature | Action |
|---|---|---|
| Charge imbalance | constant charge offset, normal resistance | Balance |
| Resistance growth | ΔV/ΔI resistance ≥ 1.2× pack median; sags more as current rises | Replace |
| Capacity fade | falls further behind per Ah discharged (−(1/C_cell − 1/C_median) per Ah) | Monitor, then replace |
| Self-discharge | keeps falling behind while parked, when no charge flows | Inspect |

Capacity fade and self-discharge look identical while driving; until the car has been parked for ~5 minutes of
settled data, CellSense reports "losing charge faster than the pack (fade or leak)" rather than guessing.

## Honest notes

* **Simulated data.** All results come from simulation. The live dashboard and the benchmark use the real-time
  equivalent-circuit model, whose OCV curve and R0/R1/τ are fitted to a PyBaMM SPMe cell (`calibrate`).
  The full PyBaMM + liionpack pack runs offline (about 1–3 minutes per simulated minute for 96 cells).
* **Physics backend limits.** SPMe has no internal leakage path, so self-discharge is only available in the
  real-time model. liionpack faults are present from the start of a run (no mid-run injection).
* **Forecast history.** Per-cycle capacity estimates are generated with the standard √cycle + linear fade model;
  `validate` checks that shape and the forecaster against the NASA and Oxford datasets.
* **Datasets are not included** (licensing and size); see "Validation on real ageing data" for the download links.
* **The 150 Ah cell** is Chen2020 (LG M50, NMC811/graphite-SiOx) with its electrodes widened 30× to match the
  deck's 96s1p, 52 kWh pack.

## Project layout

```
cellsense/
  config.py            pack definition and every threshold
  sim/                 ecm.py (real-time), physics.py (PyBaMM+liionpack), calibrate.py, faults.py,
                       drive_cycles.py, ageing.py, scenarios.py (the deck's demo pack)
  analytics/           engine.py, drift.py, anomaly.py, diagnosis.py, forecast.py, range_impact.py, training.py
  dashboard/           app.py, session.py, figures.py, assets/
  validation/          benchmark.py, datasets.py
  data/                ecm_params.json (PyBaMM fit), iforest.joblib (trained model)
tests/                 pytest suite
reports/               benchmark and validation results
mockups/               pitch-deck screenshots
```
