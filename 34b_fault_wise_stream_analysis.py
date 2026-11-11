#!/usr/bin/env python3

"""
STEP 34B
FAULT-WISE STREAM AND COLUMN ANALYSIS

Purpose:
    Analyze ALL discovered telemetry streams across ALL fault classes.

Input:
    70 labeled UAV runs
    Step 34A column statistics

For every stream:
    - compare columns across fault classes
    - calculate class-wise statistics
    - calculate eta-squared fault discrimination
    - rank columns
    - generate useful stream-level graphs

This is exploratory/data-analysis work.
It does not modify the existing ML pipeline.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/home/samiksha/uav_fdi_data"
)

INPUT_FILE = (
    BASE /
    "step34_stream_analysis" /
    "all_column_statistics.csv"
)

OUTPUT_ROOT = (
    BASE /
    "step34B_fault_analysis"
)

TOP_N = 10


# ============================================================
# FAULT CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "No Failure",
    "RC Failure",
    "GPS Failure",
    "Accelerometer Failure",
    "Gyro Failure",
    "Compass Failure",
    "Barometer Failure",
]


# ============================================================
# ETA-SQUARED
# ============================================================

def eta_squared_from_groups(group_values):
    """
    Calculate eta-squared from group means.

    group_values:
        dictionary:
            class -> numeric values
    """

    all_values = []

    groups = []

    for cls, values in group_values.items():

        values = np.asarray(
            values,
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            continue

        groups.append(
            (cls, values)
        )

        all_values.extend(
            values.tolist()
        )

    if len(all_values) == 0:
        return 0.0

    all_values = np.asarray(
        all_values,
        dtype=float
    )

    grand_mean = np.mean(
        all_values
    )

    ss_total = np.sum(
        (all_values - grand_mean) ** 2
    )

    if ss_total <= 0:
        return 0.0

    ss_between = 0.0

    for _, values in groups:

        n = len(values)

        group_mean = np.mean(
            values
        )

        ss_between += (
            n *
            (group_mean - grand_mean) ** 2
        )

    eta = (
        ss_between /
        ss_total
    )

    return float(
        np.clip(
            eta,
            0.0,
            1.0
        )
    )


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "run",
        "stream",
        "column",
        "column_type",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "range",
        "rms",
    }

    missing = (
        required -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


# ============================================================
# EXTRACT FAULT CLASS FROM RUN NAME
# ============================================================

def get_fault_class(run_name):

    text = str(run_name).upper()

    if "(NO FAILURE)" in text:
        return "No Failure"

    if "(RC FAILURE)" in text:
        return "RC Failure"

    if "(GPS FAILURE)" in text:
        return "GPS Failure"

    if "(ACCELEROMETER FAILURE)" in text:
        return "Accelerometer Failure"

    if "(GYRO FAILURE)" in text:
        return "Gyro Failure"

    if "(COMPASS FAILURE)" in text:
        return "Compass Failure"

    if "(BAROMETER FAILURE)" in text:
        return "Barometer Failure"

    return "Unknown"


# ============================================================
# BUILD RUN-LEVEL CLASS LABEL
# ============================================================

def add_class_column(df):

    df = df.copy()

    df["fault_class"] = (
        df["run"]
        .apply(get_fault_class)
    )

    unknown = (
        df["fault_class"] ==
        "Unknown"
    )

    if unknown.any():

        print(
            "WARNING:",
            int(unknown.sum()),
            "records have unknown fault class."
        )

    return df


# ============================================================
# COLUMN / CLASS SUMMARY
# ============================================================

def build_class_statistics(df):

    numeric = df[
        df["column_type"] == "numeric"
    ].copy()

    # Each row already represents a run-column.
    # We use the run-level mean as the main comparison signal.

    rows = []

    for (
        stream,
        column
    ), group in numeric.groupby(
        ["stream", "column"]
    ):

        classes_present = []

        for cls in EXPECTED_CLASSES:

            values = group.loc[
                group["fault_class"] == cls,
                "mean"
            ].dropna()

            if len(values) == 0:
                continue

            classes_present.append(
                cls
            )

            rows.append({

                "stream": stream,
                "column": column,
                "fault_class": cls,

                "n_runs": len(values),

                "mean": values.mean(),

                "std": values.std(),

                "median": values.median(),

                "min": values.min(),

                "max": values.max(),

                "range": (
                    values.max() -
                    values.min()
                ),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# DISCRIMINATION SCORE
# ============================================================

def build_discrimination_table(
    class_stats
):

    results = []

    for (
        stream,
        column
    ), group in class_stats.groupby(
        ["stream", "column"]
    ):

        class_means = group[
            "mean"
        ].dropna()

        if len(class_means) < 2:
            continue

        # Between-class variance
        grand_mean = class_means.mean()

        n_total = group[
            "n_runs"
        ].sum()

        if n_total <= 0:
            continue

        ss_between = 0.0
        ss_total = 0.0

        for _, row in group.iterrows():

            n = row["n_runs"]

            group_mean = row["mean"]

            group_std = row["std"]

            ss_between += (
                n *
                (group_mean - grand_mean) ** 2
            )

            # Approximate within-class total variation
            if np.isfinite(group_std):

                ss_total += (
                    max(
                        n - 1,
                        1
                    ) *
                    group_std ** 2
                )

        ss_total += ss_between

        if ss_total <= 0:

            eta = 0.0

        else:

            eta = (
                ss_between /
                ss_total
            )

        eta = float(
            np.clip(
                eta,
                0.0,
                1.0
            )
        )

        # Identify strongest and weakest class means
        sorted_means = (
            group[
                [
                    "fault_class",
                    "mean"
                ]
            ]
            .sort_values("mean")
        )

        low_class = (
            sorted_means.iloc[0][
                "fault_class"
            ]
        )

        high_class = (
            sorted_means.iloc[-1][
                "fault_class"
            ]
        )

        low_value = (
            sorted_means.iloc[0][
                "mean"
            ]
        )

        high_value = (
            sorted_means.iloc[-1][
                "mean"
            ]
        )

        results.append({

            "stream": stream,

            "column": column,

            "eta_squared": eta,

            "classes_present":
                len(class_means),

            "lowest_class":
                low_class,

            "lowest_class_mean":
                low_value,

            "highest_class":
                high_class,

            "highest_class_mean":
                high_value,

            "between_class_range":
                high_value - low_value,
        })

    result = pd.DataFrame(
        results
    )

    if not result.empty:

        result = result.sort_values(
            [
                "stream",
                "eta_squared"
            ],
            ascending=[
                True,
                False
            ]
        )

        result["stream_rank"] = (
            result
            .groupby("stream")
            .cumcount() + 1
        )

    return result


# ============================================================
# OVERALL STREAM IMPORTANCE
# ============================================================

def build_stream_summary(
    discrimination
):

    if discrimination.empty:

        return pd.DataFrame()

    summary = (
        discrimination
        .groupby("stream")
        .agg(

            columns_analyzed=(
                "column",
                "nunique"
            ),

            mean_eta_squared=(
                "eta_squared",
                "mean"
            ),

            max_eta_squared=(
                "eta_squared",
                "max"
            ),

            median_eta_squared=(
                "eta_squared",
                "median"
            ),

            highly_discriminative_features=(
                "eta_squared",
                lambda x: int(
                    (x >= 0.50).sum()
                )
            ),
        )
        .reset_index()
        .sort_values(
            "mean_eta_squared",
            ascending=False
        )
    )

    return summary


# ============================================================
# GRAPH 1 — TOP FEATURES PER STREAM
# ============================================================

def plot_top_features(
    discrimination
):

    for stream in sorted(
        discrimination["stream"].unique()
    ):

        data = discrimination[
            discrimination["stream"] ==
            stream
        ].head(TOP_N)

        if data.empty:
            continue

        output_dir = (
            OUTPUT_ROOT /
            stream
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        plot_data = (
            data
            .sort_values(
                "eta_squared"
            )
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.barh(
            plot_data["column"].astype(str),
            plot_data["eta_squared"]
        )

        plt.xlabel(
            "η² — Between-Class Discrimination"
        )

        plt.ylabel(
            "Telemetry Column"
        )

        plt.title(
            f"{stream} — Top {TOP_N} Fault-Discriminative Columns"
        )

        plt.xlim(
            0,
            1
        )

        plt.grid(
            True,
            axis="x",
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            output_dir /
            f"{stream}_top_features.png",
            dpi=250,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# GRAPH 2 — STREAM IMPORTANCE
# ============================================================

def plot_stream_importance(
    stream_summary
):

    if stream_summary.empty:
        return

    data = (
        stream_summary
        .sort_values(
            "mean_eta_squared"
        )
    )

    plt.figure(
        figsize=(11, 9)
    )

    plt.barh(
        data["stream"],
        data["mean_eta_squared"]
    )

    plt.xlabel(
        "Mean η² Across Columns"
    )

    plt.ylabel(
        "Telemetry Stream"
    )

    plt.title(
        "Fault Discrimination by Telemetry Stream"
    )

    plt.xlim(
        0,
        1
    )

    plt.grid(
        True,
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT /
        "overall_stream_fault_discrimination.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# GRAPH 3 — TOP FEATURES OVERALL
# ============================================================

def plot_overall_top_features(
    discrimination
):

    data = (
        discrimination
        .sort_values(
            "eta_squared",
            ascending=False
        )
        .head(25)
        .sort_values(
            "eta_squared"
        )
    )

    plt.figure(
        figsize=(12, 10)
    )

    labels = (
        data["stream"]
        + " | "
        + data["column"]
    )

    plt.barh(
        labels,
        data["eta_squared"]
    )

    plt.xlabel(
        "η²"
    )

    plt.ylabel(
        "Telemetry Feature"
    )

    plt.title(
        "Top 25 Fault-Discriminative Telemetry Features"
    )

    plt.xlim(
        0,
        1
    )

    plt.grid(
        True,
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT /
        "overall_top_25_features.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# GRAPH 4 — CLASS MEAN HEATMAP-LIKE MATRIX
# ============================================================

def build_top_feature_matrix(
    class_stats,
    discrimination
):

    top = (
        discrimination
        .sort_values(
            "eta_squared",
            ascending=False
        )
        .head(25)
    )

    rows = []

    for _, feature in top.iterrows():

        stream = feature[
            "stream"
        ]

        column = feature[
            "column"
        ]

        subset = class_stats[
            (
                class_stats["stream"] ==
                stream
            )
            &
            (
                class_stats["column"] ==
                column
            )
        ]

        row = {
            "feature":
                f"{stream} | {column}"
        }

        for cls in EXPECTED_CLASSES:

            value = subset.loc[
                subset["fault_class"] ==
                cls,
                "mean"
            ]

            if len(value):

                row[cls] = (
                    value.iloc[0]
                )

            else:

                row[cls] = np.nan

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def plot_top_feature_matrix(
    matrix
):

    if matrix.empty:
        return

    numeric = matrix.set_index(
        "feature"
    )

    # Standardize each row so different units
    # can be compared visually.
    standardized = (
        numeric
        .subtract(
            numeric.mean(axis=1),
            axis=0
        )
    )

    row_std = numeric.std(
        axis=1
    )

    row_std = row_std.replace(
        0,
        np.nan
    )

    standardized = (
        standardized
        .div(
            row_std,
            axis=0
        )
    )

    plt.figure(
        figsize=(12, 11)
    )

    plt.imshow(
        standardized.to_numpy(),
        aspect="auto"
    )

    plt.colorbar(
        label="Standardized Class Mean"
    )

    plt.xticks(
        range(
            len(numeric.columns)
        ),
        numeric.columns,
        rotation=35,
        ha="right"
    )

    plt.yticks(
        range(
            len(numeric.index)
        ),
        numeric.index,
        fontsize=8
    )

    plt.title(
        "Fault-Class Signatures of Top Telemetry Features"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT /
        "top_feature_class_signature_matrix.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "STEP 34B — FAULT-WISE STREAM ANALYSIS"
    )
    print("=" * 75)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    print()
    print(
        "Records loaded:",
        len(df)
    )

    df = add_class_column(
        df
    )

    print(
        "Runs:",
        df["run"].nunique()
    )

    print(
        "Streams:",
        df["stream"].nunique()
    )

    print(
        "Fault classes:",
        df["fault_class"].nunique()
    )

    print()

    print(
        df[
            "fault_class"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # CLASS STATISTICS
    # --------------------------------------------------------

    print()
    print(
        "BUILDING CLASS-WISE STATISTICS"
    )

    class_stats = (
        build_class_statistics(
            df
        )
    )

    class_stats.to_csv(
        OUTPUT_ROOT /
        "fault_class_column_statistics.csv",
        index=False
    )

    print(
        "Class-stat records:",
        len(class_stats)
    )

    # --------------------------------------------------------
    # DISCRIMINATION
    # --------------------------------------------------------

    print()
    print(
        "CALCULATING FAULT DISCRIMINATION"
    )

    discrimination = (
        build_discrimination_table(
            class_stats
        )
    )

    discrimination.to_csv(
        OUTPUT_ROOT /
        "stream_column_discrimination.csv",
        index=False
    )

    # --------------------------------------------------------
    # STREAM SUMMARY
    # --------------------------------------------------------

    stream_summary = (
        build_stream_summary(
            discrimination
        )
    )

    stream_summary.to_csv(
        OUTPUT_ROOT /
        "stream_fault_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # PRINT STREAM RANKING
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "STREAM FAULT-DISCRIMINATION RANKING"
    )
    print("=" * 75)

    print(
        stream_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PRINT TOP FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "TOP 30 FAULT-DISCRIMINATIVE FEATURES"
    )
    print("=" * 75)

    print(
        discrimination
        .head(30)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # GRAPHS
    # --------------------------------------------------------

    print()
    print(
        "GENERATING GRAPHS"
    )

    print("-" * 75)

    plot_top_features(
        discrimination
    )

    plot_stream_importance(
        stream_summary
    )

    plot_overall_top_features(
        discrimination
    )

    matrix = build_top_feature_matrix(
        class_stats,
        discrimination
    )

    plot_top_feature_matrix(
        matrix
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "STEP 34B COMPLETE"
    )
    print("=" * 75)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "Generated:"
    )

    print(
        "1. fault_class_column_statistics.csv"
    )

    print(
        "2. stream_column_discrimination.csv"
    )

    print(
        "3. stream_fault_summary.csv"
    )

    print(
        "4. Per-stream top-feature graphs"
    )

    print(
        "5. Overall stream discrimination graph"
    )

    print(
        "6. Overall top-feature graph"
    )

    print(
        "7. Top-feature class-signature matrix"
    )


if __name__ == "__main__":
    main()
