#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/home/samiksha/uav_fdi_data"

WINDOW_SIZE = 50

OUTPUT_FILE = os.path.join(
    BASE,
    "day2_px4_native_window_dataset.csv"
)

# Each tuple:
# (run_id, csv_file, default_label)
#
# IMPORTANT:
# compass_fault_run5 is handled specially because that recording
# contains BOTH faulty and healthy periods.
RUNS = [
    (
        "normal_run2",
        os.path.join(BASE, "step46_normal_run2.csv"),
        "No Failure"
    ),

    (
        "normal_run3",
        os.path.join(BASE, "step49_normal_run3.csv"),
        "No Failure"
    ),

    (
        "normal_run4",
        os.path.join(BASE, "step49_normal_run4.csv"),
        "No Failure"
    ),

    (
        "normal_run5",
        os.path.join(BASE, "step49_normal_run5.csv"),
        "No Failure"
    ),

    (
        "normal_run6",
        os.path.join(BASE, "step49_normal_run6.csv"),
        "No Failure"
    ),

    (
        "normal_run7",
        os.path.join(BASE, "step49_normal_run7.csv"),
        "No Failure"
    ),

    (
        "normal_run11",
        os.path.join(BASE, "step49_normal_run11.csv"),
        "No Failure"
    ),

    (
        "compass_fault_run2",
        os.path.join(BASE, "step47_compass_fault_run2.csv"),
        "Compass Failure"
    ),

    (
        "compass_fault_run3",
        os.path.join(BASE, "step49_compass_fault_run3.csv"),
        "Compass Failure"
    ),

    (
        "compass_fault_run4",
        os.path.join(BASE, "step49_compass_fault_run4.csv"),
        "Compass Failure"
    ),

    (
        "compass_fault_run5",
        os.path.join(BASE, "step49_compass_fault_run5.csv"),
        "Compass Failure"
    ),
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_window_features(window):
    """
    Extract statistical features from one window.

    For every numeric/bool column:
        mean
        std
        min
        max
        range

    This intentionally keeps the feature definition simple and
    reproducible for the PX4-native baseline.
    """

    features = {}

    for column in window.columns:

        series = window[column]

        # Convert boolean values to numeric.
        if pd.api.types.is_bool_dtype(series):
            values = series.astype(float)

        # Include numeric columns.
        elif pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(
                series,
                errors="coerce"
            ).astype(float)

        # Ignore non-numeric columns.
        else:
            continue

        values = values.to_numpy(dtype=float)

        # Safety check.
        if len(values) == 0:
            continue

        # Remove NaN/inf values for statistical calculation.
        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            continue

        features[f"{column}_mean"] = float(
            np.mean(values)
        )

        features[f"{column}_std"] = float(
            np.std(values)
        )

        features[f"{column}_min"] = float(
            np.min(values)
        )

        features[f"{column}_max"] = float(
            np.max(values)
        )

        features[f"{column}_range"] = float(
            np.max(values) - np.min(values)
        )

    return features


# ============================================================
# SPECIAL LABEL LOGIC FOR MIXED COMPASS RUN
# ============================================================

def get_window_label(run_id, window):
    """
    Determine the label for a window.

    Normal runs:
        use the predefined run-level label.

    Compass fault run #5:
        use the actual PX4 magnetometer state.

    For compass_fault_run5:

        mag_heading_ok == False
        OR
        mag_declination_ok == False

            => compass faulty

        both True
            => healthy

        If a 50-sample window contains both states,
        it crosses the transition and is discarded.
    """

    # --------------------------------------------------------
    # Normal/default behavior
    # --------------------------------------------------------

    if run_id != "compass_fault_run5":
        return None

    # --------------------------------------------------------
    # Required state columns
    # --------------------------------------------------------

    required = [
        "mag_heading_ok",
        "mag_declination_ok"
    ]

    missing = [
        col for col in required
        if col not in window.columns
    ]

    if missing:
        raise RuntimeError(
            f"{run_id}: missing required state columns: {missing}"
        )

    heading = (
        window["mag_heading_ok"]
        .astype(bool)
        .to_numpy()
    )

    declination = (
        window["mag_declination_ok"]
        .astype(bool)
        .to_numpy()
    )

    # Fault condition:
    # either heading validity OR declination validity is false.
    fault_state = (
        (~heading) |
        (~declination)
    )

    # Entire window faulty.
    if np.all(fault_state):
        return "Compass Failure"

    # Entire window healthy.
    if np.all(~fault_state):
        return "No Failure"

    # Mixed window crosses the transition.
    return None


# ============================================================
# PROCESS ONE RUN
# ============================================================

def process_run(run_id, file_path, default_label):

    print()
    print("=" * 70)
    print(f"Processing: {run_id}")
    print(f"File:       {file_path}")
    print(f"Default:    {default_label}")
    print("=" * 70)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    df = pd.read_csv(file_path)

    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if len(df) < WINDOW_SIZE:
        print(
            f"WARNING: only {len(df)} samples; "
            f"less than one complete {WINDOW_SIZE}-sample window."
        )

        return []

    # --------------------------------------------------------
    # Identify numeric/bool feature source columns.
    # --------------------------------------------------------

    feature_columns = []

    for column in df.columns:

        series = df[column]

        if (
            pd.api.types.is_numeric_dtype(series)
            or
            pd.api.types.is_bool_dtype(series)
        ):
            feature_columns.append(column)

    print(
        f"Numeric/bool source columns: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Build non-overlapping windows.
    # --------------------------------------------------------

    n_complete_windows = len(df) // WINDOW_SIZE

    print(
        f"Complete {WINDOW_SIZE}-sample windows: "
        f"{n_complete_windows}"
    )

    rows = []

    discarded_mixed = 0

    label_counts = {}

    for window_number in range(n_complete_windows):

        start = window_number * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window = df.iloc[start:end].copy()

        # ----------------------------------------------------
        # Determine label.
        # ----------------------------------------------------

        if run_id == "compass_fault_run5":

            label = get_window_label(
                run_id,
                window
            )

            # Mixed state window.
            if label is None:
                discarded_mixed += 1

                continue

        else:

            label = default_label

        # ----------------------------------------------------
        # Extract features.
        # ----------------------------------------------------

        features = extract_window_features(
            window
        )

        # ----------------------------------------------------
        # Metadata.
        # ----------------------------------------------------

        row = {
            "run_id": run_id,
            "window_number": window_number,
            "label": label
        }

        row.update(features)

        rows.append(row)

        label_counts[label] = (
            label_counts.get(label, 0) + 1
        )

    # --------------------------------------------------------
    # Run summary.
    # --------------------------------------------------------

    print()
    print("Run summary:")
    print(f"  Generated windows : {len(rows)}")
    print(f"  Discarded mixed   : {discarded_mixed}")

    if label_counts:
        for label, count in label_counts.items():
            print(
                f"  {label:<20}: {count}"
            )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PX4-NATIVE DAY 2 FEATURE EXTRACTION")
    print("=" * 70)

    print(f"Base directory : {BASE}")
    print(f"Window size    : {WINDOW_SIZE}")
    print(f"Output         : {OUTPUT_FILE}")

    all_rows = []

    # --------------------------------------------------------
    # Process every run.
    # --------------------------------------------------------

    for run_id, file_path, default_label in RUNS:

        run_rows = process_run(
            run_id,
            file_path,
            default_label
        )

        all_rows.extend(run_rows)

    # --------------------------------------------------------
    # Check that something was produced.
    # --------------------------------------------------------

    if not all_rows:
        raise RuntimeError(
            "No windows were generated."
        )

    # --------------------------------------------------------
    # Create final dataset.
    # --------------------------------------------------------

    dataset = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # Sort for reproducibility.
    # --------------------------------------------------------

    dataset = dataset.sort_values(
        by=[
            "run_id",
            "window_number"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Replace non-finite values.
    # --------------------------------------------------------

    numeric_columns = dataset.select_dtypes(
        include=[np.number]
    ).columns

    dataset[numeric_columns] = (
        dataset[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # --------------------------------------------------------
    # Report missing values.
    # --------------------------------------------------------

    missing_total = int(
        dataset.isna().sum().sum()
    )

    if missing_total > 0:

        print()
        print(
            f"WARNING: {missing_total} missing values found."
        )

        # Fill numeric missing values with column median.
        for column in numeric_columns:

            if dataset[column].isna().any():

                median = dataset[column].median()

                if pd.isna(median):
                    median = 0.0

                dataset[column] = (
                    dataset[column]
                    .fillna(median)
                )

    else:

        print()
        print("Missing values: 0")

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL DATASET REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print(
        f"Rows              : {len(dataset)}"
    )

    print(
        f"Columns            : {len(dataset.columns)}"
    )

    print(
        f"ML feature columns : "
        f"{len(dataset.columns) - 3}"
    )

    print(
        f"Independent runs   : "
        f"{dataset['run_id'].nunique()}"
    )

    print()
    print("Label distribution:")
    print(
        dataset["label"]
        .value_counts()
        .to_string()
    )

    print()
    print("Windows per run:")

    run_summary = (
        dataset
        .groupby(
            ["run_id", "label"]
        )
        .size()
        .reset_index(
            name="windows"
        )
    )

    print(
        run_summary.to_string(
            index=False
        )
    )

    print()
    print("Missing values:")
    print(
        int(
            dataset.isna()
            .sum()
            .sum()
        )
    )

    print()
    print("Duplicate rows:")
    print(
        int(
            dataset.duplicated()
            .sum()
        )
    )

    print()
    print("=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print()


if __name__ == "__main__":
    main()
