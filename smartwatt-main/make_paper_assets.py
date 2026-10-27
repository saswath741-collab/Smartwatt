"""Generate figures and LaTeX table fragments for the paper into ./paper."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from smartwatt.data import generate_synthetic, load_uci
from smartwatt.evaluate import run_once, run_repeated

UCI = "data_uci/household_power_consumption.txt"
N_SYNTH, N_REAL = 10, 5
os.makedirs("paper", exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
COLS = ["precision", "recall", "f1", "false_alarm_rate"]


def tex_table(mean, std, path):
    names = {"false_alarm_rate": "FAR"}
    best = mean["f1"].idxmax()
    lines = []
    for det in mean.index:
        cells = [f"{mean.loc[det, c]:.3f}$\\pm${std.loc[det, c]:.3f}" for c in COLS]
        if det == best:
            cells = [f"\\textbf{{{x}}}" for x in cells]
            det = f"\\textbf{{{det}}}"
        lines.append(f"{det} & " + " & ".join(cells) + " \\\\")
    open(path, "w", encoding="utf8").write("\n".join(lines) + "\n")


# ---- synthetic: mean/std over independent datasets
s_mean, s_std = run_repeated(N_SYNTH)
tex_table(s_mean, s_std, "paper/table_synth.tex")

# ---- real: mean/std over anomaly-injection seeds on the same real series
runs = [run_once(load_uci(UCI, seed=s), seed=s)[0] for s in range(N_REAL)]
stack = pd.concat(runs, keys=range(N_REAL))
r_mean = stack.groupby(level=1, sort=False).mean()
r_std = stack.groupby(level=1, sort=False).std()
tex_table(r_mean, r_std, "paper/table_real.tex")
s_mean.to_csv("paper/synth_mean.csv"); r_mean.to_csv("paper/real_mean.csv")


# ---- figures
def bars(mean, std, path):
    ax = mean[["precision", "recall", "f1", "false_alarm_rate"]].plot.bar(
        yerr=std[COLS], figsize=(3.4, 2.4), rot=25, capsize=1.5, width=0.8)
    ax.set_ylabel("score"); ax.legend(fontsize=6, ncol=2)
    plt.setp(ax.get_xticklabels(), ha="right")
    plt.tight_layout(); plt.savefig(path); plt.close()


bars(s_mean, s_std, "paper/fig_synth_bar.png")
bars(r_mean, r_std, "paper/fig_real_bar.png")

for tag, df in [("synth", generate_synthetic(seed=0)), ("real", load_uci(UCI, seed=0))]:
    _, feat, preds, cut = run_once(df)
    keep = ["Fixed threshold", "Isolation Forest", "Drift-adaptive (EWMA)"]
    win = feat.iloc[cut:cut + 168]
    fig, axes = plt.subplots(len(keep), 1, figsize=(3.4, 3.6), sharex=True)
    for ax, name in zip(axes, keep):
        ax.plot(win["timestamp"], win["consumption"], lw=0.6, color="gray")
        t = win[win["is_anomaly"] == 1]
        ax.scatter(t["timestamp"], t["consumption"], s=22, facecolors="none", edgecolors="green", lw=0.8)
        f = win[preds[name][cut:cut + 168] == 1]
        ax.scatter(f["timestamp"], f["consumption"], s=6, color="red")
        ax.set_title(name, fontsize=7, loc="left", pad=1)
        ax.set_ylabel("kW" if tag == "real" else "kWh")
    plt.setp(axes[-1].get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout(); plt.savefig(f"paper/fig_det_{tag}.png"); plt.close()

print(s_mean.round(3)); print(r_mean.round(3)); print(r_std.round(3))
