"""Run the SmartWatt comparison and save results/plots into ./results."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from smartwatt.data import generate_synthetic, load_csv, load_uci
from smartwatt.evaluate import run_once, run_repeated

p = argparse.ArgumentParser()
p.add_argument("--csv", help="optional real dataset instead of synthetic data")
p.add_argument("--uci", help="path to household_power_consumption.txt (labelled anomalies are injected)")
p.add_argument("--time-col", default="timestamp")
p.add_argument("--value-col", default="consumption")
p.add_argument("--label-col", help="0/1 anomaly labels; omit if unlabelled")
p.add_argument("--seeds", type=int, default=10)
a = p.parse_args()
os.makedirs("results", exist_ok=True)
pd.set_option("display.width", 120)
pd.options.display.float_format = "{:.3f}".format

if a.csv or a.uci:
    df = load_uci(a.uci) if a.uci else load_csv(a.csv, a.time_col, a.value_col, a.label_col)
    table, feat, preds, cut = run_once(df)
    if table is None:
        print("No labels given: skipping metrics, plotting detections only.")
    else:
        print(table)
        table.to_csv("results/metrics_real.csv")
else:
    mean, std = run_repeated(a.seeds)
    print(f"Test-set results, mean over {a.seeds} synthetic datasets\n{mean}\n\nstd\n{std}")
    mean.to_csv("results/metrics_mean.csv")
    std.to_csv("results/metrics_std.csv")
    table, feat, preds, cut = run_once(generate_synthetic(seed=0), seed=0)

win = feat.iloc[cut:cut + 240]
fig, axes = plt.subplots(len(preds), 1, figsize=(12, 9), sharex=True)
for ax, (name, pr) in zip(axes, preds.items()):
    ax.plot(win["timestamp"], win["consumption"], lw=0.8, color="gray")
    t = win[win["is_anomaly"] == 1]
    ax.scatter(t["timestamp"], t["consumption"], s=70, facecolors="none",
               edgecolors="green", label="true anomaly")
    f = win[pr[cut:cut + 240] == 1]
    ax.scatter(f["timestamp"], f["consumption"], s=15, color="red", label="flagged")
    ax.set_ylabel("kWh")
    ax.set_title(name, fontsize=10, loc="left")
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.savefig("results/detections.png", dpi=120)

if table is not None:
    table.plot.bar(figsize=(9, 4), rot=15)
    plt.tight_layout()
    plt.savefig("results/metrics.png", dpi=120)
print("Saved plots/tables in ./results")
