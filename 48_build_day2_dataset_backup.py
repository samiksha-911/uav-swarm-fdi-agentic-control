import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/home/samiksha/uav_fdi_data")

# ============================================================
# INDEPENDENT PX4 RECORDINGS
# ============================================================

RUNS = [
    # Normal
    ("normal_run2", BASE / "step46_normal_run2.csv", "No Failure"),
    ("normal_run3", BASE / "step49_normal_run3.csv", "No Failure"),
    ("normal_run4", BASE / "step49_normal_run4.csv", "No Failure"),
    ("normal_run5", BASE / "step49_normal_run5.csv", "No Failure"),
    ("normal_run6", BASE / "step49_normal_run6.csv", "No Failure"),
    ("normal_run7", BASE / "step49_normal_run7.csv", "No Failure"),
    ("normal_run11", BASE / "step49_normal_run11.csv", "No Failure"),

    # Compass fault
    ("compass_fault_run2", BASE / "step47_compass_fault_run2.csv", "Compass Failure"),
    ("compass_fault_run3", BASE / "step49_compass_fault_run3.csv", "Compass Failure"),
    ("compass_fault_run4", BASE / "step49_compass_fault_run4.csv", "Compass Failure"),
    ("compass_fault_run5", BASE / "step49_compass_fault_run5.csv", "Compass Failure"),
]

OUTPUT_FILE = BASE / "step48_day2_dataset.csv"

# 50 samples at approximately 10 Hz = 5 seconds
WINDOW_SIZE = 50


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_window_features(window):
    """
    Convert one 50-sample PX4 telemetry window into
    mean/std/min/max/range features.
    """

    features = {}

    numeric_cols = window.select_dtypes(
        include=[np.number, "bool"]
    ).columns

    for col in numeric_cols:

        # Convert booleans to numeric
        values = pd.to_numeric(
            window[col],
            errors="coerce"
        ).astype(float)

        # Skip columns that are completely invalid
        if values.isna().all():
            continue

        values = values.dropna()

        if len(values) == 0:
            continue

        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std(ddof=0)
        features[f"{col}_min"] = values.min()
        features[f"{col}_max"] = values.max()
        features[f"{col}_range"] = values.max() - values.min()

    return features


# ============================================================
# BUILD DATASET
# ============================================================

all_windows = []

print("=" * 70)
print("STEP 48 — BUILD PX4-NATIVE DAY 2 DATASET")
print("=" * 70)

for run_id, filepath, label in RUNS:

    print()
    print(f"Processing: {filepath.name}")
    print(f"Run ID:    {run_id}")
    print(f"Label:     {label}")

    if not filepath.exists():
        print("WARNING: File not found — skipping")
        continue

    df = pd.read_csv(filepath)

    print(f"Samples:   {len(df)}")

    # Remove incomplete final window.
    n_complete = len(df) // WINDOW_SIZE

    print(f"Complete windows: {n_complete}")

    for window_number in range(n_complete):

        start = window_number * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window = df.iloc[start:end].copy()

        features = extract_window_features(window)

        features["run_id"] = run_id
        features["window_number"] = window_number
        features["label"] = label

        all_windows.append(features)


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

dataset = pd.DataFrame(all_windows)

if dataset.empty:
    raise RuntimeError("No complete windows were created.")

# Put metadata first
metadata = [
    "run_id",
    "window_number",
    "label",
]

feature_columns = [
    c for c in dataset.columns
    if c not in metadata
]

dataset = dataset[
    metadata + feature_columns
]

# Sort for reproducibility
dataset = dataset.sort_values(
    ["run_id", "window_number"]
).reset_index(drop=True)


# ============================================================
# QUALITY CHECKS
# ============================================================

print()
print("=" * 70)
print("PX4-NATIVE DAY 2 DATASET")
print("=" * 70)

print("Rows/windows:", len(dataset))
print("Columns:", len(dataset.columns))
print("ML features:", len(feature_columns))

print()
print("Class distribution:")
print(dataset["label"].value_counts())

print()
print("Independent runs:")
print(dataset.groupby(
    ["run_id", "label"]
).size())

print()
print("Missing values:", int(dataset.isna().sum().sum()))

print()
print("Windows per run:")
print(dataset.groupby("run_id").size())


# ============================================================
# SAVE
# ============================================================

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("=" * 70)
print("STEP 48 COMPLETE")
print("=" * 70)
