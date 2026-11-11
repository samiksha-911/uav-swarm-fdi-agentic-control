import pandas as pd
import numpy as np
import glob
import os

BASE = "/home/samiksha/uav_fdi_data/Processed Data"
VARIABLE_FILE = "/home/samiksha/uav_fdi_data/variable_features.csv"

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

# Number of samples before/after failure transition
PRE_FAILURE = 5000
POST_FAILURE = 5000

# Maximum number of rows from each normal run
NORMAL_ROWS = 10000

OUTPUT = "/home/samiksha/uav_fdi_data/fdi_training_windows.csv"

# ---------------------------------------------------------
# Load feature list
# ---------------------------------------------------------

variable_features = pd.read_csv(VARIABLE_FILE)["feature"].tolist()

# Remove target/time if present
features = [
    f for f in variable_features
    if f not in ["Time", "Status"]
]

print("=" * 70)
print("FDI DATASET PREPARATION")
print("=" * 70)
print("Candidate input features:", len(features))

# ---------------------------------------------------------
# Find runs
# ---------------------------------------------------------

runs = []

for folder in sorted(glob.glob(BASE + "/*")):

    if not os.path.isdir(folder):
        continue

    name = os.path.basename(folder)

    csvs = glob.glob(folder + "/*.csv")

    if not csvs:
        continue

    csvs.sort(key=os.path.getsize, reverse=True)

    if "(" in name and ")" in name:
        failure_type = name.rsplit("(", 1)[1].rstrip(")")
    else:
        failure_type = "Unknown"

    runs.append({
        "name": name,
        "file": csvs[0],
        "failure_type": failure_type
    })

print("Runs:", len(runs))
print()

# ---------------------------------------------------------
# Process runs
# ---------------------------------------------------------

all_windows = []

for i, run in enumerate(runs, 1):

    print(f"[{i}/{len(runs)}] {run['name']}")

    file = run["file"]
    failure_type = run["failure_type"]

    # Read only required columns
    usecols = features + ["Status"]

    try:
        df = pd.read_csv(
            file,
            usecols=lambda c: c in usecols,
            low_memory=False
        )
    except Exception as e:
        print("  ERROR:", e)
        continue

    # -----------------------------------------------------
    # Convert features to numeric
    # -----------------------------------------------------

    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target
    df["Status"] = pd.to_numeric(
        df["Status"],
        errors="coerce"
    )

    # Remove rows with invalid target
    df = df.dropna(subset=["Status"])

    # -----------------------------------------------------
    # Failure / normal handling
    # -----------------------------------------------------

    if failure_type == "No Failure":

        # Keep only a limited number of normal samples
        # spread across the run
        n = min(NORMAL_ROWS, len(df))

        if len(df) > n:
            idx = np.linspace(
                0,
                len(df) - 1,
                n
            ).astype(int)

            df = df.iloc[idx].copy()

        df["failure_type"] = "No Failure"
        df["run"] = run["name"]

        all_windows.append(df)

        continue

    # -----------------------------------------------------
    # Locate failure transition
    # -----------------------------------------------------

    status = df["Status"].to_numpy()

    transition = np.where(
        (status[:-1] == 0) &
        (status[1:] == 1)
    )[0]

    if len(transition) == 0:

        print("  WARNING: no 0 -> 1 transition")

        continue

    transition_row = transition[0] + 1

    # -----------------------------------------------------
    # Extract window around failure
    # -----------------------------------------------------

    start = max(
        0,
        transition_row - PRE_FAILURE
    )

    end = min(
        len(df),
        transition_row + POST_FAILURE
    )

    window = df.iloc[start:end].copy()

    # Relative time/sample index
    window["sample_index"] = np.arange(
        start,
        end
    )

    window["relative_to_failure"] = (
        window["sample_index"] -
        transition_row
    )

    window["failure_type"] = failure_type
    window["run"] = run["name"]

    all_windows.append(window)

# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------

if not all_windows:
    raise RuntimeError("No usable runs were processed.")

result = pd.concat(
    all_windows,
    ignore_index=True
)

# ---------------------------------------------------------
# Clean target
# ---------------------------------------------------------

result["Status"] = (
    result["Status"]
    .round()
    .astype(int)
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 70)
print("DATASET READY")
print("=" * 70)

print("Shape:", result.shape)

print()
print("Rows by failure type:")
print(
    result["failure_type"]
    .value_counts()
)

print()
print("Status distribution:")
print(
    result["Status"]
    .value_counts()
)

print()
print("Saved:")
print(OUTPUT)
