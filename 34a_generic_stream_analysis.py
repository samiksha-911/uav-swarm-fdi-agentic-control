#!/usr/bin/env python3

"""
STEP 34A
GENERALIZED UAV TELEMETRY DATA ANALYSIS

Design:
- Automatically scans every run folder.
- Automatically discovers CSV files.
- Groups files by normalized filename.
- Treats the same filename across different runs as the same stream.
- Analyzes every discovered column.
- Handles numeric and categorical columns generically.
- Detects schema differences.
- Produces statistics for ALL runs.
- Produces only STREAM-LEVEL plots to avoid thousands of files.
- Ignores obvious run-level label/export CSVs.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    "/home/samiksha/uav_fdi_data/Processed Data"
)

OUTPUT_ROOT = Path(
    "/home/samiksha/uav_fdi_data/step34_stream_analysis"
)

# Maximum number of columns displayed in one aggregate plot.
MAX_COLUMNS_PER_PLOT = 20

# Number of representative runs shown in time-series plots.
MAX_REPRESENTATIVE_RUNS = 3

# Maximum points retained for plotting one time series.
MAX_TIME_POINTS = 5000


# ============================================================
# FILE / STREAM HELPERS
# ============================================================

def normalize_stream_name(path: Path) -> str:
    """
    Convert filenames into a consistent stream name.

    Examples:
        GPS.csv       -> GPS
        gps.csv       -> GPS
        GPS (1).csv   -> GPS

    Files that look like complete run exports are handled
    separately by should_ignore_file().
    """

    name = path.stem.strip()

    # Remove "(1)", "(2)", etc.
    name = re.sub(
        r"\s*\(\d+\)$",
        "",
        name
    )

    return name.upper()


def should_ignore_file(path: Path, run_name: str) -> bool:
    """
    Ignore files that appear to be a complete run export or
    label file rather than a telemetry stream.

    We DO NOT ignore normal telemetry files such as:
        ATT.csv
        GPS.csv
        IMU.csv
        XKF1.csv

    We ignore a CSV when its filename is essentially the
    same as the run folder name.
    """

    normalized_file = re.sub(
        r"\s+",
        " ",
        path.stem.strip().upper()
    )

    normalized_run = re.sub(
        r"\s+",
        " ",
        run_name.strip().upper()
    )

    if normalized_file == normalized_run:
        return True

    # Remove common failure labels and retry.
    cleaned_file = re.sub(
        r"\s*\(.*?\)\s*",
        "",
        normalized_file
    )

    cleaned_run = re.sub(
        r"\s*\(.*?\)\s*",
        "",
        normalized_run
    )

    if cleaned_file == cleaned_run:
        return True

    return False


# ============================================================
# DISCOVER RUNS / FILES
# ============================================================

def discover_files():
    """
    Discover CSV files underneath run folders.
    """

    records = []

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATA_ROOT}"
        )

    for path in sorted(
        DATA_ROOT.rglob("*.csv")
    ):

        if not path.is_file():
            continue

        relative = path.relative_to(
            DATA_ROOT
        )

        # Need at least:
        # RunFolder/file.csv
        if len(relative.parts) < 2:
            continue

        run_name = relative.parts[0]

        if should_ignore_file(
            path,
            run_name
        ):
            continue

        records.append({
            "run": run_name,
            "path": path,
            "stream": normalize_stream_name(path),
        })

    return records


def group_streams(records):
    """
    Group all matching filenames into stream families.
    """

    streams = {}

    for record in records:

        stream = record["stream"]

        streams.setdefault(
            stream,
            []
        ).append(record)

    return streams


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_time_column(df):
    """
    Detect likely time column automatically.
    """

    preferred = [
        "Time",
        "time",
        "TIME",
        "Timestamp",
        "timestamp",
        "TIMESTAMP",
    ]

    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:

        name = str(col).lower()

        if (
            "timestamp" in name
            or name == "time"
            or name.endswith("_time")
        ):
            return col

    return None


def classify_columns(df):

    time_col = find_time_column(df)

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:

        if col == time_col:
            continue

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.80:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return (
        time_col,
        numeric_cols,
        categorical_cols
    )


# ============================================================
# NUMERIC STATISTICS
# ============================================================

def numeric_statistics(series):

    x = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = x.dropna()

    if valid.empty:

        return {
            "rows": len(x),
            "valid": 0,
            "missing": len(x),
            "unique": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "range": np.nan,
            "rms": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

    return {
        "rows": len(x),
        "valid": len(valid),
        "missing": int(x.isna().sum()),
        "unique": int(valid.nunique()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "median": float(valid.median()),
        "range": float(
            valid.max() - valid.min()
        ),
        "rms": float(
            np.sqrt(
                np.mean(
                    valid.to_numpy() ** 2
                )
            )
        ),
        "q25": float(
            valid.quantile(0.25)
        ),
        "q75": float(
            valid.quantile(0.75)
        ),
    }


# ============================================================
# CATEGORICAL STATISTICS
# ============================================================

def categorical_statistics(series):

    s = series.astype("string")

    counts = (
        s.value_counts(
            dropna=False
        )
    )

    categories = " | ".join(
        f"{str(k)}:{int(v)}"
        for k, v in counts.head(20).items()
    )

    return {
        "rows": len(s),
        "valid": int(s.notna().sum()),
        "missing": int(s.isna().sum()),
        "unique": int(
            s.nunique(dropna=False)
        ),
        "mean": np.nan,
        "std": np.nan,
        "min": np.nan,
        "max": np.nan,
        "median": np.nan,
        "range": np.nan,
        "rms": np.nan,
        "q25": np.nan,
        "q75": np.nan,
        "categories": categories,
    }


# ============================================================
# FILE ANALYSIS
# ============================================================

def analyze_file(record):

    run = record["run"]
    path = record["path"]
    stream = record["stream"]

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(
            f"READ ERROR | {path} | {exc}"
        )
        return [], None

    (
        time_col,
        numeric_cols,
        categorical_cols
    ) = classify_columns(df)

    rows = []

    # -------------------------
    # Numeric columns
    # -------------------------

    for col in numeric_cols:

        stats = numeric_statistics(
            df[col]
        )

        rows.append({
            "run": run,
            "stream": stream,
            "column": col,
            "column_type": "numeric",
            **stats,
        })

    # -------------------------
    # Categorical columns
    # -------------------------

    for col in categorical_cols:

        stats = categorical_statistics(
            df[col]
        )

        rows.append({
            "run": run,
            "stream": stream,
            "column": col,
            "column_type": "categorical",
            **stats,
        })

    metadata = {
        "df": df,
        "time_col": time_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }

    return rows, metadata


# ============================================================
# STREAM-LEVEL AGGREGATION
# ============================================================

def build_stream_summary(stats_df):

    if stats_df.empty:
        return pd.DataFrame()

    numeric = stats_df[
        stats_df["column_type"] == "numeric"
    ].copy()

    if numeric.empty:
        return pd.DataFrame()

    summary = (
        numeric
        .groupby(
            [
                "stream",
                "column"
            ]
        )
        .agg(
            runs_analyzed=(
                "run",
                "nunique"
            ),
            total_rows=(
                "rows",
                "sum"
            ),
            mean_of_means=(
                "mean",
                "mean"
            ),
            mean_std=(
                "std",
                "mean"
            ),
            global_min=(
                "min",
                "min"
            ),
            global_max=(
                "max",
                "max"
            ),
            mean_median=(
                "median",
                "mean"
            ),
            mean_range=(
                "range",
                "mean"
            ),
            mean_rms=(
                "rms",
                "mean"
            ),
            total_missing=(
                "missing",
                "sum"
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# AGGREGATE GRAPH 1
# COLUMN MEANS
# ============================================================

def plot_stream_means(
    stream,
    summary,
    output_dir
):

    df = summary[
        summary["stream"] == stream
    ].copy()

    if df.empty:
        return

    df = df.head(
        MAX_COLUMNS_PER_PLOT
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        df["column"].astype(str),
        df["mean_of_means"]
    )

    plt.title(
        f"{stream} - Mean Value by Column"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Mean"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{stream}_column_means.png",
        dpi=200
    )

    plt.close()


# ============================================================
# AGGREGATE GRAPH 2
# RANGES
# ============================================================

def plot_stream_ranges(
    stream,
    summary,
    output_dir
):

    df = summary[
        summary["stream"] == stream
    ].copy()

    if df.empty:
        return

    df = df.head(
        MAX_COLUMNS_PER_PLOT
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        df["column"].astype(str),
        df["mean_range"]
    )

    plt.title(
        f"{stream} - Average Range by Column"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Average Range"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{stream}_column_ranges.png",
        dpi=200
    )

    plt.close()


# ============================================================
# AGGREGATE GRAPH 3
# MISSING VALUES
# ============================================================

def plot_stream_missingness(
    stream,
    summary,
    output_dir
):

    df = summary[
        summary["stream"] == stream
    ].copy()

    if df.empty:
        return

    df = df.head(
        MAX_COLUMNS_PER_PLOT
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        df["column"].astype(str),
        df["total_missing"]
    )

    plt.title(
        f"{stream} - Missing Values Across Runs"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Total Missing Values"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{stream}_missingness.png",
        dpi=200
    )

    plt.close()


# ============================================================
# REPRESENTATIVE TIME SERIES
# ============================================================

def choose_representative_records(
    records
):

    # Evenly sample runs so that the plots
    # do not contain every run.
    records = sorted(
        records,
        key=lambda x: x["run"]
    )

    if len(records) <= MAX_REPRESENTATIVE_RUNS:
        return records

    indices = np.linspace(
        0,
        len(records) - 1,
        MAX_REPRESENTATIVE_RUNS
    ).astype(int)

    return [
        records[i]
        for i in indices
    ]


def plot_representative_timeseries(
    stream,
    records,
    output_dir
):

    selected = choose_representative_records(
        records
    )

    # Identify common numeric columns
    # from the representative runs.
    common_columns = None

    prepared = []

    for record in selected:

        try:
            df = pd.read_csv(
                record["path"]
            )
        except Exception:
            continue

        time_col = find_time_column(
            df
        )

        if time_col is None:
            continue

        _, numeric_cols, _ = classify_columns(
            df
        )

        if common_columns is None:
            common_columns = set(
                numeric_cols
            )
        else:
            common_columns &= set(
                numeric_cols
            )

        prepared.append(
            (
                record,
                df,
                time_col
            )
        )

    if not common_columns:
        return

    # Limit visualization.
    columns = sorted(
        common_columns
    )[:MAX_COLUMNS_PER_PLOT]

    # One graph per representative run,
    # with multiple columns on the same graph.
    for record, df, time_col in prepared:

        run = record["run"]

        plt.figure(
            figsize=(12, 7)
        )

        for col in columns:

            x = pd.to_numeric(
                df[time_col],
                errors="coerce"
            )

            y = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            valid = (
                x.notna()
                & y.notna()
            )

            x = x[valid]
            y = y[valid]

            if len(x) < 2:
                continue

            # Downsample for plotting only.
            if len(x) > MAX_TIME_POINTS:

                indices = np.linspace(
                    0,
                    len(x) - 1,
                    MAX_TIME_POINTS
                ).astype(int)

                x = x.iloc[indices]
                y = y.iloc[indices]

            # Normalize each column so very different
            # units can be visualized together.
            std = y.std()

            if std == 0 or not np.isfinite(std):
                continue

            y_norm = (
                y - y.mean()
            ) / std

            plt.plot(
                x,
                y_norm,
                linewidth=1,
                label=str(col)
            )

        plt.title(
            f"{stream} - Representative Run\n{run}"
        )

        plt.xlabel(
            time_col
        )

        plt.ylabel(
            "Standardized Value"
        )

        plt.legend(
            fontsize=7,
            ncol=2
        )

        plt.grid(
            True,
            alpha=0.3
        )

        filename = (
            f"{stream}__"
            f"{run}__representative_timeseries.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_dir /
            filename,
            dpi=200
        )

        plt.close()


# ============================================================
# SCHEMA AUDIT
# ============================================================

def build_schema_audit(
    streams
):

    rows = []

    for stream, records in sorted(
        streams.items()
    ):

        signatures = {}

        union_columns = set()

        for record in records:

            try:

                df = pd.read_csv(
                    record["path"],
                    nrows=3
                )

            except Exception:
                continue

            signature = tuple(
                df.columns
            )

            signatures.setdefault(
                signature,
                []
            ).append(
                record["run"]
            )

            union_columns.update(
                df.columns
            )

        rows.append({

            "stream": stream,

            "files_found":
                len(records),

            "distinct_schemas":
                len(signatures),

            "columns_found":
                " | ".join(
                    sorted(
                        map(
                            str,
                            union_columns
                        )
                    )
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "STEP 34A — GENERALIZED UAV TELEMETRY ANALYSIS"
    )
    print("=" * 75)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. DISCOVERY
    # --------------------------------------------------------

    records = discover_files()

    if not records:

        raise RuntimeError(
            "No telemetry CSV files found."
        )

    streams = group_streams(
        records
    )

    runs = sorted(
        set(
            r["run"]
            for r in records
        )
    )

    print()
    print(
        f"Runs discovered : {len(runs)}"
    )

    print(
        f"CSV files       : {len(records)}"
    )

    print(
        f"Streams         : {len(streams)}"
    )

    print()

    print("STREAM INVENTORY")
    print("-" * 75)

    for stream in sorted(streams):

        print(
            f"{stream:20s} "
            f"{len(streams[stream]):4d} files"
        )

    # --------------------------------------------------------
    # 2. SCHEMA AUDIT
    # --------------------------------------------------------

    schema_df = build_schema_audit(
        streams
    )

    schema_df.to_csv(
        OUTPUT_ROOT /
        "stream_schema_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # 3. ANALYZE ALL FILES
    # --------------------------------------------------------

    all_rows = []

    for stream in sorted(streams):

        print()
        print(
            f"ANALYZING STREAM: {stream}"
        )

        print("-" * 75)

        stream_dir = (
            OUTPUT_ROOT /
            stream
        )

        stream_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        stream_records = streams[
            stream
        ]

        for record in stream_records:

            rows, _ = analyze_file(
                record
            )

            all_rows.extend(
                rows
            )

        print(
            f"Files analyzed: "
            f"{len(stream_records)}"
        )

    # --------------------------------------------------------
    # 4. SAVE ALL COLUMN STATS
    # --------------------------------------------------------

    stats_df = pd.DataFrame(
        all_rows
    )

    stats_file = (
        OUTPUT_ROOT /
        "all_column_statistics.csv"
    )

    stats_df.to_csv(
        stats_file,
        index=False
    )

    # --------------------------------------------------------
    # 5. STREAM SUMMARY
    # --------------------------------------------------------

    summary_df = build_stream_summary(
        stats_df
    )

    summary_file = (
        OUTPUT_ROOT /
        "stream_column_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # 6. STREAM-LEVEL VISUALIZATION
    # --------------------------------------------------------

    print()
    print(
        "GENERATING STREAM-LEVEL GRAPHS"
    )

    print("-" * 75)

    for stream in sorted(streams):

        stream_dir = (
            OUTPUT_ROOT /
            stream
        )

        plot_stream_means(
            stream,
            summary_df,
            stream_dir
        )

        plot_stream_ranges(
            stream,
            summary_df,
            stream_dir
        )

        plot_stream_missingness(
            stream,
            summary_df,
            stream_dir
        )

        plot_representative_timeseries(
            stream,
            streams[stream],
            stream_dir
        )

        print(
            f"Graphs complete: {stream}"
        )

    # --------------------------------------------------------
    # 7. FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("STEP 34A COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Runs discovered       : {len(runs)}"
    )

    print(
        f"Telemetry streams     : {len(streams)}"
    )

    print(
        f"Files analyzed        : {len(records)}"
    )

    print(
        f"Column records        : {len(stats_df)}"
    )

    print()
    print(
        "Statistics saved to:"
    )

    print(
        stats_file
    )

    print()
    print(
        "Summary saved to:"
    )

    print(
        summary_file
    )

    print()
    print(
        "Schema audit saved to:"
    )

    print(
        OUTPUT_ROOT /
        "stream_schema_summary.csv"
    )

    print()
    print(
        "Graphs saved under:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "Design rule:"
    )

    print(
        "Same filename = same telemetry stream analysis."
    )

    print(
        "New run folders are automatically included."
    )

    print(
        "New stream filenames are automatically discovered."
    )

    print(
        "Schema differences are reported."
    )

    print(
        "No per-column/per-run image explosion."
    )


if __name__ == "__main__":
    main()
