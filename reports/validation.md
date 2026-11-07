# Validation against public ageing data

Two checks per cell. **Fade shape**: least-squares fit of Q(c) = Q0·(1 − a·√c − b·c), the ageing model the simulator uses. **Forecast**: CellSense's quadratic trend, fitted on the first 60% of each history, predicting the cycle at which the cell reaches a fade level it only reaches later.

Cleaning: start-up conditioning discharges (before the cell first reaches 90% of its early best), aborted discharges (≤ 0.1 Ah) and partial discharges (> 15% from the running median of 15) are dropped.

**15 cells used.**

| Dataset | Cells | Fade-shape fit R² (median, range) | Forecast error, median | Predicted early (warns before it happens) |
|---|---|---|---|---|
| NASA | 7 | 0.952 (0.416–0.977) | 15 cycles (8.6% of test length) | 7 of 7 |
| Oxford | 8 | 0.996 (0.829–0.998) | 378 cycles (5.3% of test length) | 6 of 8 |

| Cell | Test conditions | Cycles | Final fade | Fade-shape fit R² | Fit RMSE | Forecast target | Actual cycle | Predicted | Error |
|---|---|---|---|---|---|---|---|---|---|
| NASA B0005 | 24 °C, 2 A constant-current discharge, run to 30% fade | 167 | 28.5% | 0.977 | 1.54% | 24.5% | 125 | 116 | -9 |
| NASA B0006 | 24 °C, 2 A constant-current discharge, run to 30% fade | 167 | 44.1% | 0.976 | 1.82% | 35.4% | 136 | 115 | -21 |
| NASA B0007 | 24 °C, 2 A constant-current discharge, run to 30% fade | 167 | 25.3% | 0.976 | 1.29% | 20.9% | 128 | 114 | -14 |
| NASA B0018 | 24 °C, 2 A constant-current discharge, run to 30% fade | 131 | 29.5% | 0.952 | 1.78% | 24.2% | 96 | 81 | -15 |
| NASA B0033 | 24 °C, 4 A / 2 A constant-current discharge, run to 20% fade | 189 | 23.6% | 0.416 | 7.88% | 19.0% | 128 | 117 | -11 |
| NASA B0034 | 24 °C, 4 A / 2 A constant-current discharge, run to 20% fade | 195 | 19.3% | 0.646 | 3.15% | 20.9% | 149 | 119 | -30 |
| NASA B0036 | 24 °C, 4 A / 2 A constant-current discharge, run to 20% fade | 195 | 13.1% | 0.888 | 1.20% | 10.6% | 178 | 161 | -17 |
| Oxford Cell1 | 40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles | 8200 | 25.4% | 0.996 | 0.42% | 20.4% | 6400 | 5912 | -488 |
| Oxford Cell2 | 40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles | 7700 | 26.5% | 0.979 | 1.09% | 20.9% | 5600 | 5760 | +160 |
| Oxford Cell3 | 40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles | 8100 | 23.9% | 0.996 | 0.42% | 19.3% | 6300 | 5927 | -373 |
| Oxford Cell4 | 40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles | 5100 | 21.7% | 0.995 | 0.44% | 17.1% | 3800 | 3511 | -289 |
| Oxford Cell5 | 40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles | 5000 | 37.1% | 0.829 | 2.69% | 24.3% | 5000 | 6189 | +1189 |
| Oxford Cell6 | 40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles | 5000 | 21.0% | 0.996 | 0.33% | 15.9% | 4100 | 3813 | -287 |
| Oxford Cell7 | 40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles | 8100 | 20.4% | 0.998 | 0.25% | 16.5% | 6400 | 6018 | -382 |
| Oxford Cell8 | 40 °C, CC-CV charge + urban Artemis drive-cycle discharge; capacity from the characterisation every 100 cycles | 8100 | 23.5% | 0.997 | 0.38% | 19.0% | 6400 | 6004 | -396 |

A negative error means the forecast flagged the cell before it actually got there. A trend fitted on early history cannot anticipate a sudden late-life 'knee' (a cell whose fade abruptly accelerates); such cells are forecast late.

Data: NASA PCoE Battery Data Set (B. Saha and K. Goebel, 2007, NASA Ames Prognostics Data Repository); Oxford Battery Degradation Dataset 1 (D. Howey and C. Birkl, 2017, University of Oxford, ODbL; see also C. R. Birkl, 'Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries', PhD thesis, University of Oxford, 2017).

## Not used

Selection follows the README files shipped with each dataset: only tests with one fixed discharge protocol for the whole run give a clean fade curve.

| Cell | Reason |
|---|---|
| NASA B0025 | square-wave 4 A load, only 28 cycles |
| NASA B0026 | square-wave 4 A load, only 28 cycles |
| NASA B0027 | square-wave 4 A load, only 28 cycles |
| NASA B0028 | square-wave 4 A load, only 28 cycles |
| NASA B0029 | 43 °C, only 40 cycles |
| NASA B0030 | 43 °C, only 40 cycles |
| NASA B0031 | 43 °C, only 40 cycles |
| NASA B0032 | 43 °C, only 40 cycles |
| NASA B0038 | load current switched between 1, 2 and 4 A during the test |
| NASA B0039 | load current switched between 1, 2 and 4 A during the test |
| NASA B0040 | load current switched between 1, 2 and 4 A during the test |
| NASA B0041 | 4 °C, load switched between 4 A and 1 A; NASA notes unexplained low-capacity runs |
| NASA B0042 | 4 °C, load switched between 4 A and 1 A; NASA notes unexplained low-capacity runs |
| NASA B0043 | 4 °C, load switched between 4 A and 1 A; NASA notes unexplained low-capacity runs |
| NASA B0044 | 4 °C, load switched between 4 A and 1 A; NASA notes unexplained low-capacity runs |
| NASA B0045 | 4 °C; NASA notes unexplained low-capacity runs |
| NASA B0046 | 4 °C; NASA notes unexplained low-capacity runs |
| NASA B0047 | 4 °C; NASA notes unexplained low-capacity runs |
| NASA B0048 | 4 °C; NASA notes unexplained low-capacity runs |
| NASA B0049 | 4 °C, stopped by a software crash; NASA notes unexplained low-capacity runs |
| NASA B0050 | 4 °C, stopped by a software crash; NASA notes unexplained low-capacity runs |
| NASA B0051 | 4 °C, stopped by a software crash; NASA notes unexplained low-capacity runs |
| NASA B0053 | 4 °C; NASA notes unexplained low-capacity runs |
| NASA B0054 | 4 °C; NASA notes unexplained low-capacity runs |
| NASA B0055 | 4 °C; NASA notes unexplained low-capacity runs |
| NASA B0056 | 4 °C; NASA notes unexplained low-capacity runs |
