#!/usr/bin/env python3

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt


BASE = "/home/samiksha/uav_fdi_data"


def save_plot(filename):
    path = os.path.join(BASE, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


# ============================================================
# GRAPH 1 — CORRECT SEAD 70-RUN CLASS DISTRIBUTION
# ============================================================

file = os.path.join(
    BASE,
    "step20_sead_archive_runs.csv"
)

df = pd.read_csv(file)

print("\nGRAPH 1 — SEAD ARCHIVE RUNS")
print(df.head())
print("\nColumns:", list(df.columns))

# Find the label column
label_col = None

for c in df.columns:
    cl = c.lower()

    if (
        "label" in cl
        or "class" in cl
        or "failure" in cl
    ):
        label_col = c
        break

if label_col is None:
    raise ValueError(
        "Could not identify SEAD class/label column."
    )

counts = (
    df[label_col]
    .astype(str)
    .value_counts()
)

plt.figure(figsize=(10, 6))

plt.bar(
    counts.index,
    counts.values
)

plt.title(
    "SEAD Dataset: Fault Class Distribution",
    fontsize=16
)

plt.xlabel(
    "Fault Class",
    fontsize=12
)

plt.ylabel(
    "Number of Runs",
    fontsize=12
)

plt.xticks(
    rotation=30,
    ha="right"
)

for i, value in enumerate(counts.values):
    plt.text(
        i,
        value + 0.2,
        str(value),
        ha="center",
        fontsize=11
    )

save_plot(
    "graph1_sead_class_distribution.png"
)


# ============================================================
# GRAPH 2 — FEATURE REDUCTION
# ============================================================

labels = [
    "Extracted",
    "Non-constant",
    "Selected"
]

values = [
    1551,
    1154,
    750
]

plt.figure(figsize=(8, 6))

plt.bar(
    labels,
    values
)

plt.title(
    "SEAD Feature Reduction",
    fontsize=16
)

plt.xlabel(
    "Feature Processing Stage",
    fontsize=12
)

plt.ylabel(
    "Number of Features",
    fontsize=12
)

for i, value in enumerate(values):
    plt.text(
        i,
        value + 20,
        str(value),
        ha="center",
        fontsize=11
    )

save_plot(
    "graph2_sead_feature_reduction.png"
)


# ============================================================
# GRAPH 3 — FEATURE COUNT VS ACCURACY
# ============================================================

file = os.path.join(
    BASE,
    "step29_sead_feature_count_experiment.csv"
)

df = pd.read_csv(file)

print("\nGRAPH 3 — FEATURE COUNT")
print(df)

plt.figure(figsize=(9, 6))

plt.plot(
    df["top_k"],
    df["accuracy_mean"] * 100,
    marker="o",
    linewidth=2
)

# Highlight 750 features
row750 = df[df["top_k"] == 750]

if not row750.empty:

    acc750 = (
        row750["accuracy_mean"].iloc[0]
        * 100
    )

    plt.scatter(
        [750],
        [acc750],
        s=100,
        zorder=5
    )

    plt.annotate(
        f"750 features\n{acc750:.2f}%",
        (750, acc750),
        xytext=(15, 10),
        textcoords="offset points",
        fontsize=11
    )

plt.title(
    "Effect of Feature Count on Classification Accuracy",
    fontsize=16
)

plt.xlabel(
    "Number of Selected Features",
    fontsize=12
)

plt.ylabel(
    "5-Fold CV Accuracy (%)",
    fontsize=12
)

plt.xticks(
    df["top_k"]
)

plt.ylim(
    50,
    100
)

plt.grid(
    True,
    alpha=0.3
)

save_plot(
    "graph3_feature_count_vs_accuracy.png"
)


# ============================================================
# GRAPH 4 — TOP 15 FEATURES BY ETA-SQUARED
# ============================================================

file = os.path.join(
    BASE,
    "step26_sead_leakage_audit.csv"
)

df = pd.read_csv(file)

print("\nGRAPH 4 — ETA SQUARED")
print(df.head(10))

feature_col = "feature"
eta_col = "eta_squared"

if feature_col not in df.columns:
    raise ValueError(
        "Missing feature column."
    )

if eta_col not in df.columns:
    raise ValueError(
        "Missing eta_squared column."
    )

top = (
    df
    .sort_values(
        eta_col,
        ascending=False
    )
    .head(15)
    .sort_values(
        eta_col,
        ascending=True
    )
)

plt.figure(figsize=(11, 8))

plt.barh(
    top[feature_col].astype(str),
    top[eta_col]
)

plt.title(
    "Top 15 SEAD Features by η²",
    fontsize=16
)

plt.xlabel(
    "η² (Class Discrimination)",
    fontsize=12
)

plt.ylabel(
    "Telemetry Feature",
    fontsize=12
)

save_plot(
    "graph4_top_eta_squared_features.png"
)


# ============================================================
# GRAPH 5 — CORRECT SEAD MODEL COMPARISON
# ============================================================

file = os.path.join(
    BASE,
    "step28_sead_baseline_model_results.csv"
)

df = pd.read_csv(file)

print("\nGRAPH 5 — MODEL COMPARISON")
print(df)

model_col = "model"
accuracy_col = "accuracy"

summary = (
    df
    .groupby(model_col)[accuracy_col]
    .mean()
    .sort_values(
        ascending=False
    )
)

plt.figure(figsize=(9, 6))

plt.bar(
    summary.index,
    summary.values * 100
)

plt.title(
    "SEAD Baseline Model Comparison",
    fontsize=16
)

plt.xlabel(
    "Machine Learning Model",
    fontsize=12
)

plt.ylabel(
    "Mean 5-Fold CV Accuracy (%)",
    fontsize=12
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=15,
    ha="right"
)

for i, value in enumerate(summary.values * 100):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=11
    )

save_plot(
    "graph5_sead_model_comparison.png"
)


# ============================================================
# GRAPH 6 — TRAINING VS VALIDATION ACCURACY
# ============================================================

file = os.path.join(
    BASE,
    "step33A_fold_results.csv"
)

df = pd.read_csv(file)

print("\nGRAPH 6 — GENERALIZATION CHECK")
print(df)

plt.figure(figsize=(9, 6))

plt.plot(
    df["fold"],
    df["train_accuracy"] * 100,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    df["fold"],
    df["validation_accuracy"] * 100,
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.title(
    "ExtraTrees: Training vs Validation Accuracy",
    fontsize=16
)

plt.xlabel(
    "Cross-Validation Fold",
    fontsize=12
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=12
)

plt.xticks(
    df["fold"]
)

plt.ylim(
    80,
    102
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

save_plot(
    "graph6_training_vs_validation_accuracy.png"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("CORRECTED PROJECT GRAPHS GENERATED")
print("=" * 70)

for i in range(1, 7):
    print(
        f"{i}. graph{i}_"
    )

print()
print("Location:")
print(BASE)
