import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# STEP 44 — PX4 WINDOW-LEVEL FEATURE EXTRACTION
# ============================================================

BASE = Path("/home/samiksha/uav_fdi_data")

NORMAL_FILE = BASE / "step43_normal_px4_recording.csv"
FAULT_FILE = BASE / "step43_compass_fault_px4_recording.csv"

OUTPUT_FILE = BASE / "step44_px4_window_features.csv"

# PX4 recorder is approximately 10 Hz.
# 100 samples ~= 10 seconds.
WINDOW_SIZE = 100

# Never use these as ML features.
EXCLUDE_COLUMNS = {
    "time",
}


# ============================================================
# LOAD AND CLEAN
# ============================================================

def prepare_dataframe(df):
    """
    Convert boolean and numeric telemetry into float64.

    This prevents errors such as:
    numpy boolean subtract is not supported
    """

    df = df.copy()

    for col in df.columns:

        if col in EXCLUDE_COLUMNS:
            continue

        # Boolean -> 0.0 / 1.0
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("float64")

        # Numeric -> float64
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("float64")

    return df


# ============================================================
# WINDOW FEATURE EXTRACTION
# ============================================================

def make_window_features(df, label, source_file):

    df = prepare_dataframe(df)

    # Select numeric telemetry columns only.
    feature_columns = [
        col
        for col in df.columns
        if col not in EXCLUDE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    rows = []

    # Ignore incomplete final window.
    n_windows = len(df) // WINDOW_SIZE

    print(f"\nSource: {source_file}")
    print(f"Label: {label}")
    print(f"Samples: {len(df)}")
    print(f"Complete windows: {n_windows}")
    print(f"Telemetry signals: {len(feature_columns)}")

    for window_id in range(n_windows):

        start = window_id * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window = df.iloc[start:end]

        row = {
            "source_file": source_file,
            "window_id": window_id,
            "label": label,
        }

        for col in feature_columns:

            # Force everything to float.
            values = pd.to_numeric(
                window[col],
                errors="coerce"
            ).astype("float64")

            # Remove infinities.
            values = values.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # Remove missing values.
            values = values.dropna()

            if len(values) == 0:
                continue

            minimum = float(values.min())
            maximum = float(values.max())

            row[f"{col}__mean"] = float(values.mean())
            row[f"{col}__std"] = float(values.std(ddof=0))
            row[f"{col}__min"] = minimum
            row[f"{col}__max"] = maximum
            row[f"{col}__range"] = maximum - minimum

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("STEP 44 — PX4 WINDOW-LEVEL FEATURE EXTRACTION")
print("=" * 70)

# ------------------------------------------------------------
# Check input files
# ------------------------------------------------------------

if not NORMAL_FILE.exists():
    raise FileNotFoundError(
        f"Normal recording not found:\n{NORMAL_FILE}"
    )

if not FAULT_FILE.exists():
    raise FileNotFoundError(
        f"Compass fault recording not found:\n{FAULT_FILE}"
    )


# ------------------------------------------------------------
# Load recordings
# ------------------------------------------------------------

print("\nLoading recordings...")

normal = pd.read_csv(NORMAL_FILE)
fault = pd.read_csv(FAULT_FILE)

print(f"Normal recording: {len(normal)} samples")
print(f"Compass fault recording: {len(fault)} samples")

print(f"Normal columns: {len(normal.columns)}")
print(f"Fault columns: {len(fault.columns)}")


# ------------------------------------------------------------
# Verify recording schemas
# ------------------------------------------------------------

normal_columns = set(normal.columns)
fault_columns = set(fault.columns)

if normal_columns != fault_columns:

    only_normal = sorted(normal_columns - fault_columns)
    only_fault = sorted(fault_columns - normal_columns)

    print("\nWARNING: Recording schemas differ.")

    if only_normal:
        print("Only in normal recording:")
        print(only_normal)

    if only_fault:
        print("Only in fault recording:")
        print(only_fault)

else:
    print("\nRecording schemas match.")


# ------------------------------------------------------------
# Create window-level features
# ------------------------------------------------------------

print("\nCreating windows...")

normal_features = make_window_features(
    normal,
    "No Failure",
    NORMAL_FILE.name,
)

fault_features = make_window_features(
    fault,
    "Compass Failure",
    FAULT_FILE.name,
)


# ------------------------------------------------------------
# Combine datasets
# ------------------------------------------------------------

dataset = pd.concat(
    [
        normal_features,
        fault_features,
    ],
    ignore_index=True,
)


# ------------------------------------------------------------
# Clean numerical values
# ------------------------------------------------------------

dataset = dataset.replace(
    [np.inf, -np.inf],
    np.nan
)


# ------------------------------------------------------------
# Identify feature columns
# ------------------------------------------------------------

metadata_columns = {
    "source_file",
    "window_id",
    "label",
}

feature_columns = [
    col
    for col in dataset.columns
    if col not in metadata_columns
]


# ------------------------------------------------------------
# Missing-value analysis
# ------------------------------------------------------------

missing_counts = dataset[feature_columns].isna().sum()

features_with_missing = int(
    (missing_counts > 0).sum()
)

total_missing = int(
    missing_counts.sum()
)

print("\nMissing-value summary:")
print(
    f"Features with missing values: "
    f"{features_with_missing}"
)

print(
    f"Total missing values: "
    f"{total_missing}"
)


# ------------------------------------------------------------
# Remove completely missing features
# ------------------------------------------------------------

usable_features = [
    col
    for col in feature_columns
    if dataset[col].notna().any()
]

removed_features = [
    col
    for col in feature_columns
    if not dataset[col].notna().any()
]

if removed_features:

    print(
        f"\nRemoving {len(removed_features)} "
        "completely missing features."
    )

dataset = dataset[
    [
        "source_file",
        "window_id",
        "label",
    ]
    + usable_features
]


# ------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total windows: {len(dataset)}")

print(
    f"Telemetry-derived ML features: "
    f"{len(usable_features)}"
)

print("\nClass distribution:")

print(
    dataset["label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# Check feature values
# ------------------------------------------------------------

print("\nFeature matrix shape:")

print(
    dataset[usable_features].shape
)


# ------------------------------------------------------------
# Check for remaining NaN / infinity
# ------------------------------------------------------------

remaining_nan = int(
    dataset[usable_features].isna().sum().sum()
)

remaining_inf = int(
    np.isinf(
        dataset[usable_features].to_numpy(
            dtype="float64"
        )
    ).sum()
)

print("\nFinal numerical validation:")

print(
    f"Remaining NaN values: {remaining_nan}"
)

print(
    f"Remaining infinite values: {remaining_inf}"
)


# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")

print(OUTPUT_FILE)


# ------------------------------------------------------------
# Show sample
# ------------------------------------------------------------

print("\nFirst 5 rows:")

print(
    dataset.head().to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Show feature names
# ------------------------------------------------------------

print("\nFirst 30 generated features:")

for feature in usable_features[:30]:
    print(feature)


# ------------------------------------------------------------
# Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 44 COMPLETE")
print("=" * 70)

print(
    "\nPX4 raw telemetry has been converted into "
    "fixed-length window-level ML features."
)

print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)
