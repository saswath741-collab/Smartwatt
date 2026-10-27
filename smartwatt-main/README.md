# SmartWatt

Adaptive machine learning for electricity consumption anomaly detection.
Compares five detectors on hourly consumption data:

| Detector | Idea |
|---|---|
| Fixed threshold | mean ± 3·std of training data (baseline) |
| Adaptive threshold | robust z-score vs. the same hour/day-type over the last 14 days |
| Drift-adaptive (EWMA) | online per-slot baseline that follows changing usage |
| Isolation Forest | unsupervised ML on time-aware features |
| Hybrid adaptive | adaptive z-score very high, OR Isolation Forest flags it and z-score is moderately high |

Metrics: precision, recall, F1, false alarm rate (FP / all normal readings),
computed on a chronological test split (first 60% train, last 40% test).
Features and baselines use past data only, so there is no future leakage.

## Run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run_experiment.py          # synthetic data, 10 seeds
.venv\Scripts\streamlit run dashboard.py        # optional dashboard
```

Outputs go to `results/` (metrics CSVs, `detections.png`, `metrics.png`).

Real data: `python run_experiment.py --csv file.csv --time-col Datetime --value-col kWh [--label-col label]`.
Without a label column, metrics are skipped (the abstract's "wherever reliable
labels are available") and only detections are plotted.

## Results (synthetic with usage drift, mean of 10 seeds, test split)

| Detector | Precision | Recall | F1 | False alarm rate |
|---|---|---|---|---|
| Fixed threshold | 0.876 | 0.191 | 0.307 | 0.001 |
| Adaptive threshold (14-day window) | 0.755 | 1.000 | 0.856 | 0.011 |
| Drift-adaptive (EWMA) | 0.923 | 1.000 | 0.959 | 0.003 |
| Isolation Forest | 0.496 | 0.409 | 0.428 | 0.018 |
| Hybrid adaptive | 0.832 | 0.986 | 0.901 | 0.006 |

Reading: the fixed threshold misses most anomalies that are unusual *for their
time of day*; Isolation Forest alone lacks time context and is hurt by drift;
the online drift-adaptive baseline gives the best F1 because it follows the
usage change. The hybrid improves on the static adaptive window and on
Isolation Forest alone.

A Jupyter/Colab version is in `SmartWatt_Colab.ipynb` (`SmartWatt_executed.ipynb`
has outputs saved); `colab_single_cell.py` is the same code as one pasteable cell.

## Caveats (say these in the report)

- The synthetic data has a clear daily/weekly routine, which favours time-aware
  methods. Confirm the finding on a real dataset (e.g. UCI Household Electric
  Power Consumption) with labels you injected or defined, and report that honestly.
- The hybrid thresholds (k_hi=5, k_lo=2.5) were set by hand, not tuned on a
  validation split. Tune them on a validation period before claiming gains.
- Adaptive threshold alone has higher recall than the hybrid; the hybrid's win
  is precision/F1. The ablation row is there to show this.
