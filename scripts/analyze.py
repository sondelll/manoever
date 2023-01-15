import os
import sys
sys.path.insert(0, os.curdir)
from matplotlib import pyplot as plt
from src.mnvr.analysis import aggregate_runs_from_dir
from os import listdir

runs_dir = "./data/reports/runs"

penalties = {}

for run in listdir(runs_dir):
    if run.startswith("."):
        continue
    _progress = aggregate_runs_from_dir(f"{runs_dir}/{run}")
    _progress.save_csv(f"{runs_dir}/{run}/summary.csv")
    penalties[run] = _progress.penalties

labels = []
for k, v in penalties.items():
    key = str(k)
    if "flat" in key:
        plt.plot(v)
        labels.append(str(k))

plt.legend(labels)
plt.show()