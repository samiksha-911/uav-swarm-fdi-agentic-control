#!/usr/bin/env python3
"""
step74_plot_real_world_sead_results.py
======================================
Plots the empirical proof of realistic 85-93% performance using the
SEAD Real-World UAV Flight Benchmark dataset (step 33):
1. SEAD Real-World Confusion Matrix (92.86% accuracy, showing realistic ambiguity)
2. Real-World vs SITL Performance Comparison Bar Chart
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/home/samiksha/uav_fdi_data")

# 1. Load SEAD Confusion Matrix
cm_df = pd.read_csv(BASE / "step33_sead_fdi_confusion_matrix.csv", index_col=0)
labels = list(cm_df.index)
cm_vals = cm_df.to_numpy()

fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
im = ax.imshow(cm_vals, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
short_labels = [l.replace(" Failure", "") for l in labels]
ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9, fontweight="bold")
ax.set_yticklabels(short_labels, fontsize=9, fontweight="bold")

thresh = cm_vals.max() / 2.0
for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm_vals[i, j]
        color = "white" if val > thresh else "black"
        ax.text(j, i, str(int(val)), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

ax.set_ylabel("Ground Truth (Real-World Flight Runs)", fontsize=10, fontweight="bold")
ax.set_xlabel("Predicted Class (Held-Out Test Set)", fontsize=10, fontweight="bold")
ax.set_title("SEAD Real-World UAV Flight Dataset: Held-Out Test Confusion Matrix\n(Accuracy = 92.86%, Macro-F1 = 0.8367, Weighted-F1 = 0.9592)", fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()

out_cm = BASE / "real_world_sead_confusion_matrix.png"
plt.savefig(out_cm)
plt.close()
print(f"Saved: {out_cm}")

# 2. Comparison Bar Chart: Controlled SITL vs Real-World Flight Benchmark
categories = ["Accelerometer", "Barometer", "GPS", "Gyroscope", "Compass", "No Failure", "Overall"]
sitl_scores = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
sead_scores = [100.0, 100.0, 100.0, 100.0, 75.0,  100.0, 92.86]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
rects1 = ax.bar(x - width/2, sitl_scores, width, label="Controlled SITL (Hard Cutoffs)", color="#2ca02c", alpha=0.85, edgecolor="black")
rects2 = ax.bar(x + width/2, sead_scores, width, label="Real-World Flight (SEAD Benchmark)", color="#1f77b4", alpha=0.85, edgecolor="black")

ax.set_ylabel("Recall / Accuracy (%)", fontsize=10, fontweight="bold")
ax.set_title("FDI Performance: Controlled Simulation vs Real-World Flight Benchmark", fontsize=11, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
ax.set_ylim(0, 115)
ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% Target Baseline")
ax.legend(loc="lower right", fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
out_comp = BASE / "real_world_vs_sitl_comparison.png"
plt.savefig(out_comp)
plt.close()
print(f"Saved: {out_comp}")
print("Done!")
