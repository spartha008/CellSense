# CellSense benchmark

200 randomised fault runs + 40 healthy packs, 96 cells each: a 30 min drive (city / highway / mixed) followed by 10 min parked, real-time ECM simulator fitted to PyBaMM. Run time 86 s.

| Metric | Result |
|---|---|
| Faults detected (any flag on the faulty cell) | 96.5% |
| Recall (faulty cell gets a specific diagnosis) | 90.5% |
| Precision vs injected faults only | 73.6% |
| Precision vs ground truth (cell really out of line: ≥0.8% SoC, ≥1% capacity or ≥12% resistance) | 89.0% |
| Diagnoses on non-injected cells per pack | 0.27 (capacity_fade: 22, charge_imbalance: 43) |
| ...of which not supported by ground truth (true false alarms) | 0.11 per pack |
| Watch-list entries per pack (early drift, not an alarm) | 1.26 |
| Diagnosis matches the injected fault | 86.0% |
| Threshold-only BMS detected the fault | 1.0% |
| Forecast error, cycles to 20 km threshold (MAE) | 13 cycles (~0.5 months), 54 cells |

## By fault type

| Fault | Runs | Detected | Median delay | 90th pct delay | Right action | Threshold BMS | Final diagnoses |
|---|---|---|---|---|---|---|---|
| Charge imbalance | 50 | 98% | 24 s | 82 s | 94% | 0% | None: 0, capacity_fade: 1, charge_imbalance: 47, early_drift: 1 |
| Resistance growth | 50 | 100% | 20 s | 72 s | 94% | 0% | capacity_fade: 1, early_drift: 2, resistance_growth: 47 |
| Capacity fade | 50 | 88% | 598 s | 1003 s | 70% | 0% | None: 0, capacity_fade: 35, early_drift: 1 |
| Self-discharge | 50 | 100% | 262 s | 556 s | 86% | 4% | capacity_fade: 5, charge_imbalance: 2, self_discharge: 43 |

Severities are drawn uniformly from: Charge imbalance 0.01–0.03, Resistance growth 0.2–0.6, Capacity fade 0.02–0.06, Self-discharge 0.02–0.1.
Self-discharge and capacity fade look alike while driving; the parked period at the end of each trip is what separates them. 'charge_loss' means fade-or-leak (not yet separated) and is scored as right for capacity fade, wrong for self-discharge.
