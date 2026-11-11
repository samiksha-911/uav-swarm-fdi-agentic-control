#!/usr/bin/env python3

"""
======================================================================
STEP 21 — SEAD-ONLY TELEMETRY INSPECTION
======================================================================

Purpose:
    Inspect the actual telemetry structure inside the SEAD archive.

IMPORTANT:
    - NO model training
    - NO dataset merging
    - NO modification of fdi_train.csv
    - NO modification of fdi_test.csv
    - NO full archive extraction
    - Extracts ONLY a small sample from selected runs

This step determines:
    1. What telemetry files exist per SEAD run.
    2. Which files contain usable tabular data.
    3. Their columns and sample counts.
    4. Whether the 70 labeled runs have consistent structure.
    5. Which telemetry streams are worth using in the next step.
======================================================================
"""

import os
import re
import json
import csv
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

RAR_PATH = Path.home() / "Downloads" / "Processed Data.rar"

OUTPUT_DIR = Path.cwd()

SAMPLE_RUNS_PER_CLASS = 1

# We only extract a few small telemetry files from representative runs.
# These are the streams most relevant to FDI investigation.
PRIORITY_STREAMS = [
    "ATT.csv",
    "AHR2.csv",
    "IMU.csv",
    "MAG.csv",
    "GPS.csv",
    "GPA.csv",
    "BARO.csv",
    "RATE.csv",
    "XKF1.csv",
    "XKF2.csv",
    "XKF3.csv",
    "XKF4.csv",
    "XKF5.csv",
    "XKQ.csv",
    "XKTV.csv",
    "XKY0.csv",
    "XKY1.csv",
    "MAV.csv",
]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_command(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout


def get_archive_listing():
    """
    Read archive listing only.
    Nothing is extracted here.
    """

    print("Reading archive listing...")

    output = run_command([
        "unrar",
        "lb",
        str(RAR_PATH)
    ])

    return [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]


def parse_run_folders(entries):
    """
    Extract:

        Processed Data/<run folder>/

    from the archive listing.
    """

    run_folders = set()

    prefix = "Processed Data/"

    for entry in entries:

        if not entry.startswith(prefix):
            continue

        remainder = entry[len(prefix):]

        parts = remainder.split("/")

        if len(parts) >= 2:

            run_folder = parts[0]

            if run_folder:
                run_folders.add(run_folder)

    return sorted(run_folders)


def get_label(run_folder):
    """
    Labels are encoded directly in archive folder names:

        timestamp (Failure Type)
    """

    match = re.search(r"\(([^()]*)\)\s*$", run_folder)

    if match:
        return match.group(1).strip()

    return "UNKNOWN"


def choose_representative_runs(run_folders):
    """
    Choose one run from every SEAD class.

    This keeps extraction very small.
    """

    by_class = defaultdict(list)

    for run in run_folders:
        by_class[get_label(run)].append(run)

    selected = []

    for label in sorted(by_class):

        runs = sorted(by_class[label])

        selected.extend(
            runs[:SAMPLE_RUNS_PER_CLASS]
        )

    return selected, by_class


def archive_entries_for_run(entries, run_folder):
    prefix = f"Processed Data/{run_folder}/"

    return [
        e for e in entries
        if e.startswith(prefix)
    ]


def extract_single_file(archive_entry, temp_dir):
    """
    Extract exactly one file from the RAR.

    No full extraction.
    """

    run_command([
        "unrar",
        "e",
        "-idq",
        "-o+",
        str(RAR_PATH),
        archive_entry,
        str(temp_dir) + os.sep
    ])


def inspect_csv(path):
    """
    Read only enough information to inspect structure.

    For potentially large CSVs:
        - header is read first
        - a small sample is read
        - row count is obtained separately when practical
    """

    result = {
        "file": path.name,
        "size_kb": round(path.stat().st_size / 1024, 2),
        "columns": [],
        "column_count": 0,
        "sample_rows": 0,
        "rows_estimate": None,
        "numeric_columns": [],
        "datetime_like_columns": [],
    }

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as f:

            reader = csv.reader(f)

            header = next(reader)

        result["columns"] = header
        result["column_count"] = len(header)

    except Exception as exc:

        result["error"] = f"Header read failed: {exc}"
        return result

    # -------------------------------------------------------------
    # Small sample
    # -------------------------------------------------------------

    try:

        sample = pd.read_csv(
            path,
            nrows=1000
        )

        result["sample_rows"] = len(sample)

        numeric = sample.select_dtypes(
            include="number"
        ).columns.tolist()

        result["numeric_columns"] = numeric

        # Look for timestamp/time-like columns
        for col in sample.columns:

            name = str(col).lower()

            if any(
                token in name
                for token in [
                    "time",
                    "timestamp",
                    "datetime",
                    "tus",
                    "timeus"
                ]
            ):
                result["datetime_like_columns"].append(col)

    except Exception as exc:

        result["sample_error"] = str(exc)

    # -------------------------------------------------------------
    # Lightweight row count
    # -------------------------------------------------------------

    try:

        with open(
            path,
            "rb"
        ) as f:

            rows = sum(
                1 for _ in f
            )

        # subtract header
        result["rows_estimate"] = max(
            rows - 1,
            0
        )

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print_header(
        "STEP 21 — SEAD-ONLY TELEMETRY INSPECTION"
    )

    print()
    print("NO MODEL TRAINING")
    print("NO DATA MERGING")
    print("NO FULL ARCHIVE EXTRACTION")
    print()
    print("Only one representative run per SEAD class will be sampled.")

    # -----------------------------------------------------------------
    # 1. FILE CHECK
    # -----------------------------------------------------------------

    print_header("1. FILE CHECK")

    if not RAR_PATH.exists():

        raise FileNotFoundError(
            f"SEAD archive not found:\n{RAR_PATH}"
        )

    size_gb = RAR_PATH.stat().st_size / (1024 ** 3)

    print(f"Processed Data.rar:")
    print(f"  {RAR_PATH}")
    print(f"  Size: {size_gb:.2f} GB")

    # -----------------------------------------------------------------
    # 2. ARCHIVE LISTING
    # -----------------------------------------------------------------

    print_header("2. READING ARCHIVE STRUCTURE")

    entries = get_archive_listing()

    print(f"Total archive entries: {len(entries)}")

    run_folders = parse_run_folders(entries)

    print(f"SEAD run folders: {len(run_folders)}")

    # -----------------------------------------------------------------
    # 3. CLASS DISTRIBUTION
    # -----------------------------------------------------------------

    print_header("3. SEAD CLASS DISTRIBUTION")

    selected_runs, by_class = choose_representative_runs(
        run_folders
    )

    for label in sorted(by_class):

        print(
            f"{label:<35} "
            f"{len(by_class[label])}"
        )

    print()
    print(
        f"Representative runs selected: "
        f"{len(selected_runs)}"
    )

    # -----------------------------------------------------------------
    # 4. SELECTED RUNS
    # -----------------------------------------------------------------

    print_header("4. REPRESENTATIVE RUNS")

    for run in selected_runs:

        print(
            f"{get_label(run):<30} "
            f"{run}"
        )

    # -----------------------------------------------------------------
    # 5. STREAM INVENTORY WITHOUT EXTRACTION
    # -----------------------------------------------------------------

    print_header(
        "5. STREAM INVENTORY FROM ARCHIVE LISTING"
    )

    all_streams = Counter()

    for run in run_folders:

        run_entries = archive_entries_for_run(
            entries,
            run
        )

        for entry in run_entries:

            filename = Path(entry).name

            if filename.lower().endswith(".csv"):

                all_streams[filename] += 1

    print(
        f"Unique CSV stream names: {len(all_streams)}"
    )

    print()

    for stream, count in all_streams.most_common():

        print(
            f"{stream:<20} "
            f"{count:>3} runs"
        )

    # -----------------------------------------------------------------
    # 6. EXTRACT ONLY PRIORITY STREAMS
    # -----------------------------------------------------------------

    print_header(
        "6. SMALL REPRESENTATIVE EXTRACTION"
    )

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="sead_step21_"
        )
    )

    inspection_records = []

    try:

        for run_index, run in enumerate(
            selected_runs,
            start=1
        ):

            print()
            print(
                f"[{run_index}/{len(selected_runs)}] "
                f"{run}"
            )

            run_entries = archive_entries_for_run(
                entries,
                run
            )

            available_names = {
                Path(e).name: e
                for e in run_entries
            }

            run_temp = temp_root / (
                f"run_{run_index}"
            )

            run_temp.mkdir(
                parents=True,
                exist_ok=True
            )

            # ---------------------------------------------------------
            # Select priority streams actually present
            # ---------------------------------------------------------

            selected_stream_entries = []

            for stream in PRIORITY_STREAMS:

                if stream in available_names:

                    selected_stream_entries.append(
                        available_names[stream]
                    )

            # Also include the run-level combined CSV if present.
            combined_candidates = [
                e for e in run_entries
                if Path(e).name == run
            ]

            for e in combined_candidates:

                if e not in selected_stream_entries:

                    selected_stream_entries.append(e)

            print(
                f"  Priority CSVs available: "
                f"{len(selected_stream_entries)}"
            )

            # ---------------------------------------------------------
            # Extract selected files only
            # ---------------------------------------------------------

            for entry in selected_stream_entries:

                filename = Path(entry).name

                print(
                    f"    extracting: {filename}"
                )

                try:

                    extract_single_file(
                        entry,
                        run_temp
                    )

                    extracted = run_temp / filename

                    if not extracted.exists():

                        # unrar may preserve directory structure
                        matches = list(
                            run_temp.rglob(filename)
                        )

                        if matches:
                            extracted = matches[0]

                    if not extracted.exists():

                        print(
                            "      WARNING: extracted "
                            "file not found"
                        )
                        continue

                    info = inspect_csv(
                        extracted
                    )

                    info["run"] = run
                    info["failure_type"] = get_label(
                        run
                    )

                    inspection_records.append(
                        info
                    )

                except Exception as exc:

                    print(
                        f"      ERROR: {exc}"
                    )

    finally:

        shutil.rmtree(
            temp_root,
            ignore_errors=True
        )

    # -----------------------------------------------------------------
    # 7. STREAM CONSISTENCY
    # -----------------------------------------------------------------

    print_header(
        "7. TELEMETRY STREAM CONSISTENCY"
    )

    stream_run_counts = Counter()

    for record in inspection_records:

        stream_run_counts[
            record["file"]
        ] += 1

    for stream in sorted(stream_run_counts):

        print(
            f"{stream:<20} "
            f"{stream_run_counts[stream]:>2} "
            f"/ {len(selected_runs)} "
            f"representative runs"
        )

    # -----------------------------------------------------------------
    # 8. COLUMN STRUCTURE
    # -----------------------------------------------------------------

    print_header(
        "8. COLUMN STRUCTURE"
    )

    structure = defaultdict(list)

    for record in inspection_records:

        structure[
            record["file"]
        ].append(
            tuple(record["columns"])
        )

    for stream in sorted(structure):

        variants = structure[stream]

        unique_variants = set(variants)

        print()
        print(
            f"{stream}"
        )
        print(
            f"  unique column layouts: "
            f"{len(unique_variants)}"
        )

        first = variants[0]

        print(
            f"  columns ({len(first)}):"
        )

        print(
            "   ",
            list(first)
        )

    # -----------------------------------------------------------------
    # 9. ROW COUNTS
    # -----------------------------------------------------------------

    print_header(
        "9. TELEMETRY ROW COUNTS"
    )

    for record in inspection_records:

        print(
            f"{record['failure_type']:<28} "
            f"{record['file']:<15} "
            f"{str(record.get('rows_estimate')):>8} rows "
            f"{record['size_kb']:>10.1f} KB"
        )

    # -----------------------------------------------------------------
    # 10. NUMERIC FEATURE INVENTORY
    # -----------------------------------------------------------------

    print_header(
        "10. NUMERIC FEATURE INVENTORY"
    )

    numeric_by_stream = {}

    for record in inspection_records:

        stream = record["file"]

        numeric_by_stream.setdefault(
            stream,
            set()
        )

        for col in record.get(
            "numeric_columns",
            []
        ):

            numeric_by_stream[
                stream
            ].add(str(col))

    for stream in sorted(numeric_by_stream):

        cols = sorted(
            numeric_by_stream[stream]
        )

        print()
        print(
            f"{stream}: "
            f"{len(cols)} numeric columns"
        )

        print(
            "  ",
            cols
        )

    # -----------------------------------------------------------------
    # 11. SAVE INSPECTION RESULTS
    # -----------------------------------------------------------------

    print_header(
        "11. SAVING INSPECTION RESULTS"
    )

    output_csv = (
        OUTPUT_DIR /
        "step21_sead_telemetry_inventory.csv"
    )

    if inspection_records:

        df = pd.DataFrame(
            inspection_records
        )

        # Convert list-like fields into readable strings
        for col in [
            "columns",
            "numeric_columns",
            "datetime_like_columns"
        ]:

            if col in df.columns:

                df[col] = df[col].apply(
                    lambda x: json.dumps(
                        x,
                        default=str
                    )
                    if isinstance(
                        x,
                        (list, tuple)
                    )
                    else x
                )

        df.to_csv(
            output_csv,
            index=False
        )

        print(
            f"Saved: {output_csv.name}"
        )

    # -----------------------------------------------------------------
    # 12. FINAL INTERPRETATION
    # -----------------------------------------------------------------

    print_header(
        "12. STEP 21 INTERPRETATION"
    )

    print(
        """
This step inspected SEAD independently.

No SEAD data was merged with the existing FDI dataset.
No test data was modified.
No classifier was trained.

The important outputs are:

    - archive stream inventory
    - representative telemetry files
    - column layouts
    - row counts
    - numeric feature inventory

NEXT DECISION:

    If the telemetry streams are structurally usable,
    Step 22 will build SEAD-only run-level features.

    We will NOT merge SEAD with fdi_train.csv until
    the SEAD-only experiment has been completed and
    evaluated.

DO NOT manually extract the full 8 GB archive.
"""
    )

    print_header(
        "STEP 21 COMPLETE"
    )

    print(
        f"Representative runs inspected: "
        f"{len(selected_runs)}"
    )

    print(
        f"Telemetry files inspected: "
        f"{len(inspection_records)}"
    )

    print()
    print(
        "Paste the COMPLETE Step 21 output here."
    )


if __name__ == "__main__":
    main()
