import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "uav_fdi_data"

INPUT = BASE / "step59_px4_native_6class_dataset.csv"
OUTPUT = BASE / "step60_px4_native_6class_fair_dataset.csv"

MAX_WINDOWS_PER_RUN = 10
RANDOM_STATE = 42

df = pd.read_csv(INPUT)

print("=" * 70)
print("STEP 60 — FAIR RUN-BALANCED PX4-NATIVE DATASET")
print("=" * 70)

print("\nOriginal shape:", df.shape)

print("\nOriginal class counts:")
print(df["label"].value_counts())

print("\nOriginal windows per run:")
print(df.groupby(["label", "run_id"]).size())


# ------------------------------------------------------------
# Deterministically cap each independent run.
# ------------------------------------------------------------

parts = []

for (label, run_id), group in df.groupby(
    ["label", "run_id"],
    sort=True
):
    group = group.sample(
        n=min(MAX_WINDOWS_PER_RUN, len(group)),
        random_state=RANDOM_STATE
    )

    parts.append(group)


fair = pd.concat(parts, ignore_index=True)

# Stable ordering for reproducibility.
fair = fair.sort_values(
    ["label", "run_id", "window_id"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Save.
# ------------------------------------------------------------

fair.to_csv(OUTPUT, index=False)


# ------------------------------------------------------------
# Report.
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FAIR DATASET")
print("=" * 70)

print("Output:", OUTPUT)
print("Shape:", fair.shape)

print("\nClass counts:")
print(fair["label"].value_counts())

print("\nRuns per class:")
print(fair.groupby("label")["run_id"].nunique())

print("\nWindows per run:")
print(
    fair.groupby(["label", "run_id"])
        .size()
)

print("\nMissing values:", int(fair.isna().sum().sum()))

print("\nMaximum windows from any run:",
      fair.groupby("run_id").size().max())

print("\nClass list:")
for c in sorted(fair["label"].unique()):
    print(" -", c)

print("\nSaved successfully.")
