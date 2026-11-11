#!/usr/bin/env python3

import os
import re
import io
import csv
import math
from collections import defaultdict, Counter

import pandas as pd

try:
    import rarfile
except ImportError:
    raise SystemExit(
        "rarfile is not installed.\n"
        "Run: pip install rarfile"
    )


# ================================================================
# CONFIG
# ================================================================

ARCHIVE = "/home/samiksha/Downloads/Processed Data.rar"

OUTPUT_FEATURES = "step22_sead_run_features.csv"
OUTPUT_INVENTORY = "step22_sead_feature_inventory.csv"

SEAD_CLASSES = [
    "No Failure",
    "RC Failure",
    "GPS Failure",
    "Accelerometer Failure",
    "Gyro Failure",
    "Compass Failure",
    "Barometer Failure",
]

REQUIRED_STREAMS = [
    "ATT.csv",
    "IMU.csv",
    "MAG.csv",
    "GPS.csv",
    "BARO.csv",
    "RATE.csv",
    "XKF1.csv",
    "XKF2.csv",
    "XKF3.csv",
    "XKF4.csv",
    "XKF5.csv",
]

# Keep Time out of feature statistics.
# We use numeric telemetry columns only.
MAX_FEATURES_PER_STREAM = 1000


# ================================================================
# HELPERS
# ================================================================

def clean_name(x):
    return str(x).strip().replace("\\", "/")


def identify_label(path):
    """
    Find SEAD failure label anywhere in the archive path.
    Longest/specific matching is used safely.
    """
    p = clean_name(path)

    matches = []
    for label in SEAD_CLASSES:
        if f"({label})" in p or label in p:
            matches.append(label)

    if not matches:
        return None

    # Prefer parenthesized exact label if present.
    for label in SEAD_CLASSES:
        if f"({label})" in p:
            return label

    return matches[-1]


def identify_run_folder(path):
    """
    Extract:
        YYYY-MM-DD HH-MM-SS (Failure Label)

    from an archive path.
    """
    p = clean_name(path)

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2})"
        r"\s+\(([^()]*(?:Failure|No Failure)[^()]*)\)"
    )

    m = pattern.search(p)

    if not m:
        return None

    timestamp = m.group(1)
    label = m.group(2).strip()

    if label not in SEAD_CLASSES:
        return None

    return f"{timestamp} ({label})"


def stream_from_path(path):
    base = os.path.basename(clean_name(path))
    if base in REQUIRED_STREAMS:
        return base
    return None


def safe_float(x):
    try:
        if x is None:
            return None

        s = str(x).strip()

        if s == "":
            return None

        v = float(s)

        if not math.isfinite(v):
            return None

        return v

    except Exception:
        return None


def sanitize_feature_name(stream, column, stat):
    return f"{stream[:-4]}__{column}__{stat}"


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def calculate_column_features(values):
    """
    Calculate compact robust run-level statistics.
    """
    if not values:
        return {}

    s = pd.Series(values, dtype="float64")

    result = {}

    result["mean"] = float(s.mean())
    result["std"] = float(s.std(ddof=0))
    result["min"] = float(s.min())
    result["max"] = float(s.max())
    result["median"] = float(s.median())

    q25 = float(s.quantile(0.25))
    q75 = float(s.quantile(0.75))

    result["q25"] = q25
    result["q75"] = q75
    result["range"] = result["max"] - result["min"]
    result["iqr"] = q75 - q25

    # Mean absolute magnitude.
    result["abs_mean"] = float(s.abs().mean())

    # RMS.
    result["rms"] = float(math.sqrt(float((s * s).mean())))

    return result


def read_stream_from_rar(rf, archive_member):
    """
    Read one CSV directly from the RAR.
    Nothing is extracted to disk.
    """

    with rf.open(archive_member) as f:
        raw = f.read()

    if not raw:
        return pd.DataFrame()

    # Decode safely.
    text = raw.decode("utf-8", errors="replace")

    return pd.read_csv(
        io.StringIO(text),
        low_memory=False
    )


def build_stream_features(rf, archive_member, stream_name):
    """
    Build run-level features for one stream.
    """

    try:
        df = read_stream_from_rar(rf, archive_member)
    except Exception as e:
        print(f"      WARNING: could not read {stream_name}: {e}")
        return {}

    if df.empty:
        return {}

    features = {}

    numeric_columns = []

    for col in df.columns:
        if str(col).strip().lower() == "time":
            continue

        converted = pd.to_numeric(df[col], errors="coerce")

        valid_count = int(converted.notna().sum())

        if valid_count == 0:
            continue

        numeric_columns.append((col, converted))

    for col, series in numeric_columns:

        values = series.dropna().tolist()

        if not values:
            continue

        stats = calculate_column_features(values)

        for stat_name, value in stats.items():
            feature_name = sanitize_feature_name(
                stream_name,
                str(col),
                stat_name
            )

            features[feature_name] = value

    return features


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("STEP 22 — SEAD RUN-LEVEL FEATURE BUILDING")
    print("=" * 70)
    print()
    print("SEAD ONLY")
    print("NO FDI MERGE")
    print("NO TEST MODIFICATION")
    print("NO MODEL TRAINING")
    print("NO FULL ARCHIVE EXTRACTION")
    print()

    # ------------------------------------------------------------
    # 1. FILE CHECK
    # ------------------------------------------------------------

    print("1. FILE CHECK")
    print("-" * 70)

    if not os.path.exists(ARCHIVE):
        raise FileNotFoundError(
            f"Archive not found:\n{ARCHIVE}"
        )

    size_gb = os.path.getsize(ARCHIVE) / (1024 ** 3)

    print(f"Processed Data.rar:")
    print(f"  {ARCHIVE}")
    print(f"  Size: {size_gb:.2f} GB")
    print()

    # ------------------------------------------------------------
    # 2. OPEN ARCHIVE
    # ------------------------------------------------------------

    print("2. READING ARCHIVE LISTING")
    print("-" * 70)

    rf = rarfile.RarFile(ARCHIVE)

    members = rf.infolist()

    print(f"Total archive entries: {len(members)}")

    # ------------------------------------------------------------
    # 3. DISCOVER SEAD RUNS
    # ------------------------------------------------------------

    run_streams = defaultdict(dict)
    run_labels = {}

    for info in members:

        if info.isdir():
            continue

        path = clean_name(info.filename)

        run_folder = identify_run_folder(path)

        if run_folder is None:
            continue

        label = identify_label(path)

        if label is None:
            continue

        stream = stream_from_path(path)

        if stream is None:
            continue

        run_labels[run_folder] = label
        run_streams[run_folder][stream] = info.filename

    runs = sorted(run_streams.keys())

    print(f"SEAD runs found: {len(runs)}")
    print()

    if not runs:
        print("DEBUG: Example archive paths:")
        shown = 0

        for info in members:
            if info.isdir():
                continue

            print("  ", info.filename)
            shown += 1

            if shown >= 30:
                break

        raise RuntimeError(
            "No SEAD runs were detected. "
            "Archive naming format did not match."
        )

    # ------------------------------------------------------------
    # 4. CLASS DISTRIBUTION
    # ------------------------------------------------------------

    print("3. CLASS DISTRIBUTION")
    print("-" * 70)

    class_counts = Counter(
        run_labels[r]
        for r in runs
    )

    for label in SEAD_CLASSES:
        print(
            f"{label:<32} "
            f"{class_counts.get(label, 0)}"
        )

    print()

    # ------------------------------------------------------------
    # 5. REQUIRED STREAM CHECK
    # ------------------------------------------------------------

    print("4. REQUIRED STREAM CHECK")
    print("-" * 70)

    for stream in REQUIRED_STREAMS:

        count = sum(
            1
            for r in runs
            if stream in run_streams[r]
        )

        print(
            f"{stream:<15} "
            f"{count}/{len(runs)} runs"
        )

    print()

    # ------------------------------------------------------------
    # 6. BUILD FEATURES
    # ------------------------------------------------------------

    print("5. BUILDING RUN-LEVEL FEATURES")
    print("-" * 70)

    all_rows = []
    inventory_rows = []

    for idx, run in enumerate(runs, start=1):

        label = run_labels[run]

        print()
        print(
            f"[{idx}/{len(runs)}] "
            f"{run}"
        )

        row = {
            "run_key": run,
            "label": label,
        }

        stream_count = 0
        feature_count = 0

        for stream in REQUIRED_STREAMS:

            member = run_streams[run].get(stream)

            if member is None:
                continue

            print(
                f"    processing {stream}"
            )

            stream_features = build_stream_features(
                rf,
                member,
                stream
            )

            row.update(stream_features)

            stream_count += 1
            feature_count += len(stream_features)

            inventory_rows.append({
                "run_key": run,
                "label": label,
                "stream": stream,
                "features_generated": len(stream_features),
            })

        row["available_required_streams"] = stream_count
        row["total_generated_features"] = feature_count

        all_rows.append(row)

        print(
            f"    streams: {stream_count}/{len(REQUIRED_STREAMS)}"
        )

        print(
            f"    features: {feature_count}"
        )

    rf.close()

    # ------------------------------------------------------------
    # 7. FEATURE TABLE
    # ------------------------------------------------------------

    print()
    print("6. FEATURE TABLE SUMMARY")
    print("-" * 70)

    if not all_rows:
        raise RuntimeError(
            "No SEAD runs were processed."
        )

    feature_df = pd.DataFrame(all_rows)

    # Put metadata first.
    metadata = [
        "run_key",
        "label",
        "available_required_streams",
        "total_generated_features",
    ]

    other_columns = [
        c for c in feature_df.columns
        if c not in metadata
    ]

    feature_df = feature_df[
        metadata + sorted(other_columns)
    ]

    print(
        f"Run rows: {len(feature_df)}"
    )

    print(
        f"Total columns: {len(feature_df.columns)}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_df.columns) - len(metadata)}"
    )

    print()

    print("Class distribution in feature table:")

    for label in SEAD_CLASSES:
        n = int(
            (feature_df["label"] == label).sum()
        )

        print(
            f"  {label:<32} {n}"
        )

    print()

    # ------------------------------------------------------------
    # 8. MISSING DATA SUMMARY
    # ------------------------------------------------------------

    print("7. FEATURE DATA QUALITY")
    print("-" * 70)

    feature_columns = [
        c for c in feature_df.columns
        if c not in metadata
    ]

    if feature_columns:

        missing_pct = (
            feature_df[feature_columns]
            .isna()
            .mean()
            * 100
        )

        print(
            f"Feature columns: {len(feature_columns)}"
        )

        print(
            f"Columns with any missing values: "
            f"{int((missing_pct > 0).sum())}"
        )

        print(
            f"Columns >50% missing: "
            f"{int((missing_pct > 50).sum())}"
        )

    print()

    # ------------------------------------------------------------
    # 9. SAVE
    # ------------------------------------------------------------

    print("8. SAVING RESULTS")
    print("-" * 70)

    feature_df.to_csv(
        OUTPUT_FEATURES,
        index=False
    )

    inventory_df = pd.DataFrame(
        inventory_rows
    )

    inventory_df.to_csv(
        OUTPUT_INVENTORY,
        index=False
    )

    print(
        f"Saved: {OUTPUT_FEATURES}"
    )

    print(
        f"Saved: {OUTPUT_INVENTORY}"
    )

    print()

    # ------------------------------------------------------------
    # 10. FINAL INTERPRETATION
    # ------------------------------------------------------------

    print("9. STEP 22 INTERPRETATION")
    print("-" * 70)

    print()
    print(
        "SEAD run-level features were built independently."
    )

    print(
        "No FDI data was merged."
    )

    print(
        "fdi_train.csv was not modified."
    )

    print(
        "fdi_test.csv was not modified."
    )

    print(
        "No classifier was trained."
    )

    print(
        "The RAR archive was accessed directly."
    )

    print(
        "The full 8 GB archive was NOT extracted."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Review the SEAD feature table before any merge."
    )

    print(
        "Do NOT merge SEAD into the FDI dataset yet."
    )

    print()
    print("=" * 70)
    print("STEP 22 COMPLETE")
    print("=" * 70)

    print(
        f"SEAD runs processed: {len(feature_df)}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_df.columns) - len(metadata)}"
    )


if __name__ == "__main__":
    main()


