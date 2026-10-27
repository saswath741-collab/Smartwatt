"""Builds SmartWatt_Colab.ipynb (self-contained) from the smartwatt package."""
import json, re

def src(name):
    s = open(f"smartwatt/{name}.py", encoding="utf8").read()
    s = re.sub(r"^from \.\w+ import .*\n", "", s, flags=re.M)
    s = re.sub(r"^(import numpy as np|import pandas as pd)\n", "", s, flags=re.M)
    s = re.sub(r"^from sklearn.*\n", "", s, flags=re.M)
    return s.strip() + "\n"

cells = [
 ("md", "# SmartWatt: drift-adaptive electricity anomaly detection\nRun all cells (Runtime > Run all). Results appear in about a minute."),
 ("code", "import numpy as np, pandas as pd, matplotlib.pyplot as plt\nfrom sklearn.ensemble import IsolationForest\nfrom sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix"),
 ("md", "## Data (synthetic with labelled anomalies and a usage drift)"),
 ("code", src("data")),
 ("md", "## Features"),
 ("code", src("features")),
 ("md", "## Detectors"),
 ("code", src("detectors")),
 ("md", "## Evaluation"),
 ("code", src("evaluate")),
 ("md", "## Run: mean over 10 datasets"),
 ("code", "mean, std = run_repeated(10)\ndisplay(mean.round(3)); display(std.round(3))\nmean.plot.bar(figsize=(10,4), rot=15); plt.show()"),
 ("md", "## Detections on one dataset"),
 ("code", "table, feat, preds, cut = run_once(generate_synthetic(seed=0))\nwin = feat.iloc[cut:cut+240]\nfig, axes = plt.subplots(len(preds), 1, figsize=(12, 10), sharex=True)\nfor ax, (name, pr) in zip(axes, preds.items()):\n    ax.plot(win['timestamp'], win['consumption'], lw=.8, color='gray')\n    t = win[win['is_anomaly']==1]\n    ax.scatter(t['timestamp'], t['consumption'], s=70, facecolors='none', edgecolors='green')\n    f = win[pr[cut:cut+240]==1]\n    ax.scatter(f['timestamp'], f['consumption'], s=15, color='red')\n    ax.set_title(name, loc='left', fontsize=10)\nplt.tight_layout(); plt.show()"),
 ("md", "## Optional: real data\nUpload a CSV, then run `df = load_csv('file.csv', 'timestamp_col', 'value_col')` followed by `run_once(df)`."),
]
nb = {"cells": [{"cell_type": "markdown" if k == "md" else "code", "metadata": {},
                 **({} if k == "md" else {"execution_count": None, "outputs": []}),
                 "source": v.splitlines(keepends=True)} for k, v in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("SmartWatt_Colab.ipynb", "w"), indent=1)
# also a plain script for testing
open("results/_nb_test.py", "w", encoding="utf8").write(
    "\n".join(v for k, v in cells if k == "code").replace("display(", "print(").replace("plt.show()", "plt.close()"))

# single paste-able Colab cell
code = "\n".join(v for k, v in cells if k == "code")
code = code.replace("display(", "print(").replace("plt.show()", "plt.show()")
open("colab_single_cell.py", "w", encoding="utf8").write(code)
