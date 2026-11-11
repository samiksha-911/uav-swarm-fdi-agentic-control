#!/usr/bin/env python3

"""
======================================================================
STEP 20 — SEAD MAPPING / ANNOTATION ANALYSIS
======================================================================

Purpose:
    Analyze SEAD mapping.json and the Processed Data.rar structure
    WITHOUT extracting the 8 GB archive.

Inputs:
    ~/Downloads/mapping.json
    ~/Downloads/Processed Data.rar

Outputs:
    step20_sead_class_summary.csv
    step20_sead_annotation_summary.csv
    step20_sead_archive_runs.csv
    step20_sead_mapping_raw.csv

This step:
    1. Inspects mapping.json
    2. Determines the actual annotation structure
    3. Counts classes
    4. Identifies possible fault/normal labels
    5. Lists archive run folders without extraction
    6. Checks overlap between mapping.json and archive runs

IMPORTANT:
    No model training.
    No merging with fdi_train/fdi_test.
    No test-set modification.
    No full archive extraction.
======================================================================
"""

import os
import re
import json
import csv
import subprocess
from collections import Counter, defaultdict

import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

MAPPING_PATH = os.path.expanduser(
    "~/Downloads/mapping.json"
)

RAR_PATH = os.path.expanduser(
    "~/Downloads/Processed Data.rar"
)

OUTPUT_CLASS = "step20_sead_class_summary.csv"
OUTPUT_ANNOTATION = "step20_sead_annotation_summary.csv"
OUTPUT_ARCHIVE = "step20_sead_archive_runs.csv"
OUTPUT_RAW = "step20_sead_mapping_raw.csv"


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def subsection(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ======================================================================
# NORMALIZATION
# ======================================================================

def clean_string(value):
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value)

    return str(value).strip()


def normalize_for_matching(value):
    value = clean_string(value).lower()

    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ======================================================================
# RECURSIVE INSPECTION OF JSON
# ======================================================================

def recursive_values(obj, path=""):
    """
    Yield:
        path, value

    for all scalar values inside a JSON object.
    """

    if isinstance(obj, dict):

        for key, value in obj.items():

            new_path = (
                f"{path}.{key}"
                if path
                else str(key)
            )

            if isinstance(value, (dict, list)):
                yield from recursive_values(
                    value,
                    new_path
                )
            else:
                yield new_path, value

    elif isinstance(obj, list):

        for i, value in enumerate(obj):

            new_path = f"{path}[{i}]"

            if isinstance(value, (dict, list)):
                yield from recursive_values(
                    value,
                    new_path
                )
            else:
                yield new_path, value


# ======================================================================
# FIND "CLASS" FIELDS
# ======================================================================

def extract_class_values(obj):
    """
    Recursively find every key named 'class'
    and return its values.
    """

    results = []

    def walk(node, path=""):

        if isinstance(node, dict):

            for key, value in node.items():

                new_path = (
                    f"{path}.{key}"
                    if path
                    else str(key)
                )

                if str(key).lower() == "class":

                    if isinstance(value, list):

                        for item in value:
                            results.append(
                                (
                                    new_path,
                                    clean_string(item)
                                )
                            )

                    elif isinstance(value, dict):

                        results.append(
                            (
                                new_path,
                                json.dumps(
                                    value,
                                    ensure_ascii=False
                                )
                            )
                        )

                    else:

                        results.append(
                            (
                                new_path,
                                clean_string(value)
                            )
                        )

                if isinstance(value, (dict, list)):
                    walk(value, new_path)

        elif isinstance(node, list):

            for i, value in enumerate(node):

                new_path = f"{path}[{i}]"

                if isinstance(value, (dict, list)):
                    walk(value, new_path)

    walk(obj)

    return results


# ======================================================================
# POSSIBLE FAULT KEYWORDS
# ======================================================================

FAULT_KEYWORDS = [
    "fault",
    "failure",
    "fail",
    "anomaly",
    "attack",
    "fdi",
    "spoof",
    "sensor",
    "gps",
    "imu",
    "gyro",
    "accelerometer",
    "accel",
    "compass",
    "mag",
    "magnetometer",
    "barometer",
    "baro",
    "rc",
    "radio",
    "control",
]


NORMAL_KEYWORDS = [
    "normal",
    "nominal",
    "healthy",
    "no fault",
    "no failure",
    "normal flight",
]


# ======================================================================
# ARCHIVE LISTING
# ======================================================================

def list_rar_files(rar_path):

    print()
    print("Reading RAR directory listing...")
    print("This does NOT extract the archive.")

    command = [
        "unrar",
        "lb",
        rar_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace"
    )

    if result.returncode != 0:

        print("WARNING: unrar listing returned code:",
              result.returncode)

    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    return lines


# ======================================================================
# GET RUN FOLDERS
# ======================================================================

def extract_archive_runs(file_list):

    runs = set()

    prefix = "Processed Data/"

    for path in file_list:

        if not path.startswith(prefix):
            continue

        remainder = path[len(prefix):]

        parts = remainder.split("/")

        if len(parts) >= 1 and parts[0]:

            runs.add(parts[0])

    return sorted(runs)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("STEP 20 — SEAD MAPPING / ANNOTATION ANALYSIS")
    print("=" * 70)

    print()
    print("NO MODEL TRAINING")
    print("NO DATA MERGING")
    print("NO FULL ARCHIVE EXTRACTION")

    # ==================================================================
    # 1. FILE CHECK
    # ==================================================================

    section("1. FILE CHECK")

    print("mapping.json:")
    print(" ", MAPPING_PATH)

    if not os.path.isfile(MAPPING_PATH):
        raise FileNotFoundError(
            f"mapping.json not found:\n{MAPPING_PATH}"
        )

    print(
        " Size:",
        round(os.path.getsize(MAPPING_PATH) / 1024, 2),
        "KB"
    )

    print()
    print("Processed Data.rar:")
    print(" ", RAR_PATH)

    if not os.path.isfile(RAR_PATH):
        raise FileNotFoundError(
            f"Processed Data.rar not found:\n{RAR_PATH}"
        )

    print(
        " Size:",
        round(os.path.getsize(RAR_PATH) / (1024 ** 3), 2),
        "GB"
    )

    # ==================================================================
    # 2. LOAD MAPPING
    # ==================================================================

    section("2. LOADING MAPPING.JSON")

    with open(
        MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    print(
        "Top-level type:",
        type(mapping).__name__
    )

    if isinstance(mapping, dict):

        print(
            "Top-level entries:",
            len(mapping)
        )

        top_keys = list(mapping.keys())

        print()
        print("First 20 run keys:")

        for key in top_keys[:20]:
            print(" ", key)

    elif isinstance(mapping, list):

        print(
            "Top-level list length:",
            len(mapping)
        )

    # ==================================================================
    # 3. STRUCTURE ANALYSIS
    # ==================================================================

    section("3. MAPPING STRUCTURE ANALYSIS")

    key_counter = Counter()
    value_counter = Counter()

    scalar_counter = Counter()

    for path, value in recursive_values(mapping):

        key = path.split(".")[-1]

        key_counter[key] += 1

        if not isinstance(value, (dict, list)):

            scalar_counter[
                f"{key}={clean_string(value)}"
            ] += 1

    print()
    print("Most common JSON field names:")

    for key, count in key_counter.most_common(30):

        print(
            f"  {key:<30} {count}"
        )

    # ==================================================================
    # 4. CLASS EXTRACTION
    # ==================================================================

    section("4. CLASS FIELD ANALYSIS")

    class_values = extract_class_values(mapping)

    print(
        "Total 'class' values found:",
        len(class_values)
    )

    class_counter = Counter()

    for path, value in class_values:

        if value:
            class_counter[value] += 1

    print()
    print("Unique class values:",
          len(class_counter))

    print()
    print("Class distribution:")

    if class_counter:

        for cls, count in class_counter.most_common():

            print(
                f"  {cls:<50} {count}"
            )

    else:

        print("  No scalar class values found.")

    # ==================================================================
    # 5. CLASS CSV
    # ==================================================================

    class_rows = []

    for cls, count in class_counter.most_common():

        normalized = normalize_for_matching(cls)

        matched_fault_keywords = [
            keyword
            for keyword in FAULT_KEYWORDS
            if keyword in normalized
        ]

        matched_normal_keywords = [
            keyword
            for keyword in NORMAL_KEYWORDS
            if keyword in normalized
        ]

        class_rows.append({
            "class": cls,
            "count": count,
            "normalized_class": normalized,
            "fault_keyword_matches":
                "|".join(matched_fault_keywords),
            "normal_keyword_matches":
                "|".join(matched_normal_keywords),
        })

    class_df = pd.DataFrame(class_rows)

    class_df.to_csv(
        OUTPUT_CLASS,
        index=False
    )

    print()
    print("Saved:",
          OUTPUT_CLASS)

    # ==================================================================
    # 6. ANNOTATION STRUCTURE
    # ==================================================================

    section("5. ANNOTATION ANALYSIS")

    annotation_counter = Counter()

    if isinstance(mapping, dict):

        for run_name, run_data in mapping.items():

            if not isinstance(run_data, dict):
                continue

            annotations = run_data.get(
                "annotations"
            )

            if annotations is None:

                annotation_counter[
                    "NO_ANNOTATIONS_FIELD"
                ] += 1

                continue

            annotation_counter[
                type(annotations).__name__
            ] += 1

    print(
        "Annotation container types:"
    )

    for key, count in annotation_counter.items():

        print(
            f"  {key:<25} {count}"
        )

    # ==================================================================
    # 7. RAW MAPPING CSV
    # ==================================================================

    section("6. BUILDING RUN-LEVEL MAPPING TABLE")

    raw_rows = []

    if isinstance(mapping, dict):

        for run_name, run_data in mapping.items():

            row = {
                "run_key": run_name
            }

            if isinstance(run_data, dict):

                for key, value in run_data.items():

                    if isinstance(value, (dict, list)):

                        row[key] = json.dumps(
                            value,
                            ensure_ascii=False
                        )

                    else:

                        row[key] = value

            raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)

    raw_df.to_csv(
        OUTPUT_RAW,
        index=False
    )

    print(
        "Mapping run rows:",
        len(raw_df)
    )

    print(
        "Columns:",
        list(raw_df.columns)
    )

    print()
    print("Saved:",
          OUTPUT_RAW)

    # ==================================================================
    # 8. ARCHIVE LISTING
    # ==================================================================

    section("7. PROCESSED DATA.RAR ANALYSIS")

    archive_files = list_rar_files(
        RAR_PATH
    )

    print(
        "Archive entries:",
        len(archive_files)
    )

    archive_runs = extract_archive_runs(
        archive_files
    )

    print(
        "Archive run folders:",
        len(archive_runs)
    )

    # ==================================================================
    # 9. ARCHIVE RUN LABEL EXTRACTION
    # ==================================================================

    archive_rows = []

    archive_label_counter = Counter()

    pattern = re.compile(
        r"^(.*?)\s*\((.*?)\)$"
    )

    for run in archive_runs:

        match = pattern.match(run)

        if match:

            run_name = match.group(1).strip()
            label = match.group(2).strip()

        else:

            run_name = run
            label = ""

        archive_label_counter[label] += 1

        archive_rows.append({
            "archive_run": run,
            "run_name": run_name,
            "label_from_folder": label
        })

    archive_df = pd.DataFrame(
        archive_rows
    )

    archive_df.to_csv(
        OUTPUT_ARCHIVE,
        index=False
    )

    print()
    print(
        "Archive label distribution:"
    )

    for label, count in archive_label_counter.most_common():

        print(
            f"  {label:<40} {count}"
        )

    print()
    print("Saved:",
          OUTPUT_ARCHIVE)

    # ==================================================================
    # 10. MAPPING ↔ ARCHIVE OVERLAP
    # ==================================================================

    section("8. MAPPING ↔ ARCHIVE OVERLAP")

    mapping_runs = set()

    if isinstance(mapping, dict):

        mapping_runs = {
            str(k).strip()
            for k in mapping.keys()
        }

    archive_run_set = set(
        archive_runs
    )

    exact_overlap = (
        mapping_runs &
        archive_run_set
    )

    mapping_only = (
        mapping_runs -
        archive_run_set
    )

    archive_only = (
        archive_run_set -
        mapping_runs
    )

    print(
        "Mapping runs:",
        len(mapping_runs)
    )

    print(
        "Archive runs:",
        len(archive_run_set)
    )

    print(
        "Exact name overlap:",
        len(exact_overlap)
    )

    print(
        "Mapping-only runs:",
        len(mapping_only)
    )

    print(
        "Archive-only runs:",
        len(archive_only)
    )

    # ==================================================================
    # 11. SHOW MAPPING-ONLY
    # ==================================================================

    subsection("Mapping-only examples")

    for run in sorted(mapping_only)[:30]:

        print(" ", run)

    # ==================================================================
    # 12. SHOW ARCHIVE-ONLY
    # ==================================================================

    subsection("Archive-only examples")

    for run in sorted(archive_only)[:30]:

        print(" ", run)

    # ==================================================================
    # 13. FAULT KEYWORD ANALYSIS
    # ==================================================================

    section("9. POSSIBLE FAULT / NORMAL CLASSES")

    possible_faults = []
    possible_normal = []

    for cls, count in class_counter.items():

        normalized = normalize_for_matching(
            cls
        )

        if any(
            keyword in normalized
            for keyword in FAULT_KEYWORDS
        ):

            possible_faults.append(
                (cls, count)
            )

        if any(
            keyword in normalized
            for keyword in NORMAL_KEYWORDS
        ):

            possible_normal.append(
                (cls, count)
            )

    print()
    print("Classes containing fault-related keywords:")

    if possible_faults:

        for cls, count in sorted(
            possible_faults,
            key=lambda x: -x[1]
        ):

            print(
                f"  {cls:<50} {count}"
            )

    else:

        print("  NONE")

    print()
    print("Classes containing normal-related keywords:")

    if possible_normal:

        for cls, count in sorted(
            possible_normal,
            key=lambda x: -x[1]
        ):

            print(
                f"  {cls:<50} {count}"
            )

    else:

        print("  NONE")

    # ==================================================================
    # 14. ARCHIVE FAULT LABELS
    # ==================================================================

    section("10. ARCHIVE FOLDER LABELS")

    known_labels = [
        label
        for label in archive_label_counter
        if label
    ]

    print(
        "Unique folder labels:",
        len(known_labels)
    )

    for label in sorted(
        known_labels
    ):

        print(
            f"  {label:<40}"
            f"{archive_label_counter[label]}"
        )

    # ==================================================================
    # 15. IMPORTANT INTERPRETATION
    # ==================================================================

    section("11. STEP 20 INTERPRETATION")

    print(
        "This step intentionally does NOT train a classifier."
    )

    print()
    print(
        "The purpose is to establish:"
    )

    print(
        "  1. What SEAD's 'class' field means."
    )

    print(
        "  2. Whether mapping.json labels correspond to"
    )

    print(
        "     archive run folders."
    )

    print(
        "  3. How many SEAD runs/classes are available."
    )

    print(
        "  4. Which SEAD data can safely become training data."
    )

    print(
        "  5. Whether there is enough information to design"
    )

    print(
        "     the >95% accuracy augmentation experiment."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT merge SEAD into fdi_train.csv yet."
    )

    print(
        "Do NOT modify fdi_test.csv."
    )

    print(
        "Do NOT extract the full 8 GB archive."
    )

    # ==================================================================
    # 16. OUTPUT SUMMARY
    # ==================================================================

    section("12. OUTPUT FILES")

    outputs = [
        OUTPUT_CLASS,
        OUTPUT_ANNOTATION,
        OUTPUT_ARCHIVE,
        OUTPUT_RAW
    ]

    for output in outputs:

        if os.path.exists(output):

            print(
                f"  {output:<45}"
                f"{os.path.getsize(output) / 1024:.1f} KB"
            )

    print()
    print("=" * 70)
    print("STEP 20 COMPLETE")
    print("=" * 70)

    print()
    print(
        "Paste the COMPLETE output here."
    )

    print(
        "Do not extract the archive manually yet."
    )


if __name__ == "__main__":
    main()
