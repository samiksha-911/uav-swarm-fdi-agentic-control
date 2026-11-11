import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "uav_fdi_data"
WINDOW = 50

# Valid controlled PX4-native classes.
RUNS = {
    "Accelerometer Failure": sorted(Path(BASE).glob("step_auto_accel_fault_run*.csv")),
    "Barometer Failure": sorted(Path(BASE).glob("step_auto_baro_fault_run*.csv")),
    "Compass Failure": sorted(Path(BASE).glob("step_auto_compass_fault_run*.csv")),
    "GPS Failure": sorted(Path(BASE).glob("step_auto_gps_fault_run*.csv")),
    "Gyroscope Failure": sorted(Path(BASE).glob("step_auto_gyro_fault_run*.csv")),
    "No Failure": sorted(Path(BASE).glob("step58_normal_run*.csv")),
}



def add_window_features(df, start, end):
    w = df.iloc[start:end].copy()

    features = {}

    numeric_cols = w.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()

    # Never use time directly as a model feature.
    exclude = {"time"}

    for col in numeric_cols:
        if col in exclude:
            continue

        x = pd.to_numeric(
            w[col],
            errors="coerce"
        ).to_numpy(dtype=float)

        if len(x) == 0 or np.all(np.isnan(x)):
            continue

        x = x[np.isfinite(x)]

        if len(x) == 0:
            continue

        features[f"{col}_mean"] = float(np.mean(x))
        features[f"{col}_std"] = float(np.std(x))
        features[f"{col}_min"] = float(np.min(x))
        features[f"{col}_max"] = float(np.max(x))
        features[f"{col}_range"] = float(
            np.max(x) - np.min(x)
        )

        if len(x) > 1:
            dx = np.diff(x)

            features[f"{col}_first"] = float(x[0])
            features[f"{col}_last"] = float(x[-1])
            features[f"{col}_delta"] = float(
                x[-1] - x[0]
            )
            features[f"{col}_mean_abs_change"] = float(
                np.mean(np.abs(dx))
            )
            features[f"{col}_max_abs_change"] = float(
                np.max(np.abs(dx))
            )
            features[f"{col}_change_std"] = float(
                np.std(dx)
            )
            features[f"{col}_change_range"] = float(
                np.max(dx) - np.min(dx)
            )

    return features


def process_fixed_label_file(path, label):
    df = pd.read_csv(path)

    # Remove rows containing missing values before windowing.
    df = df.dropna(axis=0).reset_index(drop=True)

    rows = []

    n_windows = len(df) // WINDOW

    for i in range(n_windows):
        start = i * WINDOW
        end = start + WINDOW

        features = add_window_features(
            df,
            start,
            end
        )

        features["label"] = label
        features["run_id"] = path.stem
        features["window_id"] = i

        rows.append(features)

    return rows


def process_compass_file(path):
    df = pd.read_csv(path)

    # The long Run 5 has one missing sample.
    df = df.dropna(axis=0).reset_index(drop=True)

    rows = []

    n_windows = len(df) // WINDOW

    for i in range(n_windows):
        start = i * WINDOW
        end = start + WINDOW

        w = df.iloc[start:end]

        heading = (
            w["mag_heading_ok"]
            .astype(bool)
            .to_numpy()
        )

        decl = (
            w["mag_declination_ok"]
            .astype(bool)
            .to_numpy()
        )

        # Healthy only if both indicators are valid
        # for the complete window.
        healthy = (
            np.all(heading)
            and np.all(decl)
        )

        # Faulty only if both indicators are invalid
        # for the complete window.
        faulty = (
            np.all(~heading)
            and np.all(~decl)
        )

        # Mixed window = transition window.
        # Discard it.
        if not healthy and not faulty:
            continue

        label = (
            "No Failure"
            if healthy
            else "Compass Failure"
        )

        features = add_window_features(
            df,
            start,
            end
        )

        features["label"] = label
        features["run_id"] = path.stem
        features["window_id"] = i

        rows.append(features)

    return rows


all_rows = []

# ------------------------------------------------------------
# Fixed-label recordings
# ------------------------------------------------------------

for label, filenames in RUNS.items():

    for filename in filenames:

        path = BASE / filename

        if not path.exists():
            print(
                f"WARNING: missing {filename}"
            )
            continue

        print(
            f"Processing {filename} -> {label}"
        )

        all_rows.extend(
            process_fixed_label_file(
                path,
                label
            )
        )



# ------------------------------------------------------------
# Add healthy live windows
# ------------------------------------------------------------

healthy_file = BASE / "live_healthy_model_features.csv"

if healthy_file.exists():
    healthy = pd.read_csv(healthy_file)
    all_rows.extend(
        healthy.to_dict(orient="records")
    )
    print(
        f"Added {len(healthy)} healthy windows from {healthy_file.name}"
    )
else:
    raise FileNotFoundError(
        f"Missing healthy feature file: {healthy_file}"
    )


# ------------------------------------------------------------
# Build dataset
# ------------------------------------------------------------

dataset = pd.DataFrame(all_rows)

if dataset.empty:
    raise RuntimeError(
        "No windows were generated."
    )


# Metadata columns.
metadata = [
    "label",
    "run_id",
    "window_id",
]

feature_cols = [
    c
    for c in dataset.columns
    if c not in metadata
]


# ------------------------------------------------------------
# Remove constant features
# ------------------------------------------------------------

constant_features = [
    c
    for c in feature_cols
    if dataset[c].nunique(
        dropna=False
    ) <= 1
]

dataset = dataset.drop(
    columns=constant_features
)


# ------------------------------------------------------------
# Clean remaining invalid values
# ------------------------------------------------------------

dataset = dataset.replace(
    [np.inf, -np.inf],
    np.nan
)

# Some newer recordings may contain additional telemetry fields
# that older recordings do not have. Do not delete every row just
# because those optional fields are absent elsewhere.
# Keep only features that are populated for every window.
feature_cols_now = [
    c for c in dataset.columns
    if c not in metadata
]

incomplete_features = [
    c for c in feature_cols_now
    if dataset[c].isna().any()
]

if incomplete_features:
    print(
        "\nDropping features unavailable in some recordings:"
    )
    for c in incomplete_features:
        print("  ", c)

    dataset = dataset.drop(
        columns=incomplete_features
    )

dataset = dataset.dropna(
    axis=0
).reset_index(drop=True)


# ------------------------------------------------------------
# Metadata first
# ------------------------------------------------------------

dataset = dataset[
    ["label", "run_id", "window_id"]
    + [
        c
        for c in dataset.columns
        if c not in metadata
    ]
]


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out = (
    BASE
    / "step59_px4_native_6class_dataset.csv"
)

dataset.to_csv(
    out,
    index=False
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PX4-NATIVE 6-CLASS DATASET")
print("=" * 70)

print("Output:", out)
print("Shape:", dataset.shape)

print(
    "Constant features removed:",
    len(constant_features)
)

print("\nClass counts:")
print(
    dataset["label"].value_counts()
)

print("\nRuns per class:")
print(
    dataset.groupby("label")["run_id"]
    .nunique()
)

print("\nWindows per run:")
print(
    dataset.groupby(
        ["label", "run_id"]
    ).size()
)

print(
    "\nMissing values:",
    int(dataset.isna().sum().sum())
)

print("\nAccelerometer files used:")
for filename in RUNS["Accelerometer Failure"]:
    print("  ", filename)

print("=" * 70)
