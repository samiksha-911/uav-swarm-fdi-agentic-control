import pandas as pd
import numpy as np
from pathlib import Path

INPUT = "step22_sead_run_features.csv"
OUTPUT = "step23_sead_feature_validation.csv"

EXPECTED_CLASSES = [
    "No Failure",
    "RC Failure",
    "GPS Failure",
    "Accelerometer Failure",
    "Gyro Failure",
    "Compass Failure",
    "Barometer Failure",
]

print("=" * 70)
print("STEP 23 — SEAD FEATURE TABLE VALIDATION")
print("=" * 70)
print()
print("SEAD ONLY")
print("NO FDI MERGE")
print("NO TEST MODIFICATION")
print("NO MODEL TRAINING")
print()

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
print("1. LOADING FEATURE TABLE")
print("-" * 70)

if not Path(INPUT).exists():
    raise FileNotFoundError(f"Missing input file: {INPUT}")

df = pd.read_csv(INPUT)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df)}")

# Step 22 uses "label", not "class"
if "label" not in df.columns:
    raise RuntimeError("Expected 'label' column not found.")

if "run_key" not in df.columns:
    raise RuntimeError("Expected 'run_key' column not found.")

print("Label column: label")
print("Run identifier: run_key")

# ----------------------------------------------------------------------
# 2. RUN UNIQUENESS
# ----------------------------------------------------------------------
print()
print("2. RUN-KEY VALIDATION")
print("-" * 70)

duplicate_runs = df["run_key"].duplicated().sum()

print(f"Unique run keys: {df['run_key'].nunique()}")
print(f"Duplicate run keys: {duplicate_runs}")

if duplicate_runs:
    raise RuntimeError("Duplicate run_key values detected.")

# ----------------------------------------------------------------------
# 3. CLASS DISTRIBUTION
# ----------------------------------------------------------------------
print()
print("3. CLASS DISTRIBUTION")
print("-" * 70)

class_counts = df["label"].value_counts()

for cls in EXPECTED_CLASSES:
    print(f"{cls:<35} {class_counts.get(cls, 0)}")

unexpected = sorted(set(class_counts.index) - set(EXPECTED_CLASSES))

if unexpected:
    print()
    print("WARNING — unexpected labels:")
    for cls in unexpected:
        print(f"  {cls}")

missing_classes = [
    cls for cls in EXPECTED_CLASSES
    if class_counts.get(cls, 0) == 0
]

if missing_classes:
    raise RuntimeError(
        "Expected SEAD classes missing: " + ", ".join(missing_classes)
    )

# Every expected class should contain 10 runs.
bad_counts = {
    cls: int(class_counts.get(cls, 0))
    for cls in EXPECTED_CLASSES
    if class_counts.get(cls, 0) != 10
}

if bad_counts:
    print()
    print("WARNING — class counts are not exactly 10:")
    for cls, count in bad_counts.items():
        print(f"  {cls}: {count}")

# ----------------------------------------------------------------------
# 4. FEATURE COLUMN IDENTIFICATION
# ----------------------------------------------------------------------
print()
print("4. FEATURE COLUMN IDENTIFICATION")
print("-" * 70)

metadata_columns = {
    "run_key",
    "label",
    "available_required_streams",
    "total_generated_features",
}

feature_columns = [
    c for c in df.columns
    if c not in metadata_columns
]

print(f"Metadata columns: {len(metadata_columns)}")
print(f"Feature columns: {len(feature_columns)}")

if not feature_columns:
    raise RuntimeError("No feature columns found.")

print("First 10 feature columns:")
for c in feature_columns[:10]:
    print(f"  {c}")

# ----------------------------------------------------------------------
# 5. NUMERIC VALIDATION
# ----------------------------------------------------------------------
print()
print("5. NUMERIC FEATURE VALIDATION")
print("-" * 70)

non_numeric = []

for c in feature_columns:
    if not pd.api.types.is_numeric_dtype(df[c]):
        non_numeric.append(c)

print(f"Non-numeric feature columns: {len(non_numeric)}")

if non_numeric:
    print("First non-numeric columns:")
    for c in non_numeric[:20]:
        print(f"  {c}")
    raise RuntimeError("Non-numeric feature columns detected.")

# ----------------------------------------------------------------------
# 6. MISSING VALUE CHECK
# ----------------------------------------------------------------------
print()
print("6. MISSING VALUE CHECK")
print("-" * 70)

missing_counts = df[feature_columns].isna().sum()

columns_with_missing = int((missing_counts > 0).sum())
columns_over_50 = int(
    (missing_counts > len(df) * 0.50).sum()
)

total_missing = int(missing_counts.sum())

print(f"Total missing feature values: {total_missing}")
print(f"Columns with any missing values: {columns_with_missing}")
print(f"Columns >50% missing: {columns_over_50}")

if columns_with_missing:
    print()
    print("Columns containing missing values:")
    for c in missing_counts[missing_counts > 0].index[:20]:
        print(f"  {c}: {missing_counts[c]}")

# ----------------------------------------------------------------------
# 7. INFINITE VALUE CHECK
# ----------------------------------------------------------------------
print()
print("7. INFINITE VALUE CHECK")
print("-" * 70)

numeric_values = df[feature_columns].to_numpy(dtype=float)

inf_mask = np.isinf(numeric_values)

infinite_values = int(inf_mask.sum())
columns_with_inf = int(inf_mask.any(axis=0).sum())

print(f"Total infinite feature values: {infinite_values}")
print(f"Columns containing infinity: {columns_with_inf}")

if infinite_values:
    raise RuntimeError("Infinite feature values detected.")

# ----------------------------------------------------------------------
# 8. CONSTANT FEATURE CHECK
# ----------------------------------------------------------------------
print()
print("8. CONSTANT FEATURE CHECK")
print("-" * 70)

nunique = df[feature_columns].nunique(dropna=False)

constant_columns = nunique[nunique <= 1].index.tolist()

print(f"Constant feature columns: {len(constant_columns)}")
print(f"Non-constant feature columns: {len(feature_columns) - len(constant_columns)}")

if constant_columns:
    print()
    print("First 20 constant features:")
    for c in constant_columns[:20]:
        print(f"  {c}")

# Constant features are not a fatal problem at this stage.
# They can be removed later during training preprocessing.

# ----------------------------------------------------------------------
# 9. STREAM AVAILABILITY
# ----------------------------------------------------------------------
print()
print("9. REQUIRED STREAM AVAILABILITY")
print("-" * 70)

if "available_required_streams" in df.columns:
    print(
        df["available_required_streams"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    incomplete = (df["available_required_streams"] < 11).sum()

    print(f"\nRuns with fewer than 11 required streams: {incomplete}")

    if incomplete:
        raise RuntimeError(
            "Some SEAD runs do not contain all required telemetry streams."
        )

# ----------------------------------------------------------------------
# 10. FEATURE COUNT CONSISTENCY
# ----------------------------------------------------------------------
print()
print("10. FEATURE COUNT CONSISTENCY")
print("-" * 70)

if "total_generated_features" in df.columns:
    generated_counts = df["total_generated_features"]

    print(
        f"Reported generated features: "
        f"min={generated_counts.min()}, "
        f"max={generated_counts.max()}, "
        f"unique={generated_counts.nunique()}"
    )

    mismatch = generated_counts != len(feature_columns)

    print(f"Runs with feature-count mismatch: {int(mismatch.sum())}")

    if mismatch.any():
        print("WARNING: reported feature count differs from actual table.")

# ----------------------------------------------------------------------
# 11. CLASS-WISE FEATURE SANITY
# ----------------------------------------------------------------------
print()
print("11. CLASS-WISE FEATURE SANITY")
print("-" * 70)

for cls in EXPECTED_CLASSES:
    subset = df[df["label"] == cls]

    if len(subset) == 0:
        continue

    nan_values = int(subset[feature_columns].isna().sum().sum())

    print(
        f"{cls:<35} "
        f"runs={len(subset):2d} "
        f"missing={nan_values}"
    )

# ----------------------------------------------------------------------
# 12. SAVE VALIDATION SUMMARY
# ----------------------------------------------------------------------
print()
print("12. SAVING VALIDATION SUMMARY")
print("-" * 70)

summary_rows = []

for cls in EXPECTED_CLASSES:
    subset = df[df["label"] == cls]

    summary_rows.append({
        "label": cls,
        "run_count": len(subset),
        "feature_count": len(feature_columns),
        "missing_values": int(
            subset[feature_columns].isna().sum().sum()
        ),
        "infinite_values": int(
            np.isinf(
                subset[feature_columns].to_numpy(dtype=float)
            ).sum()
        ),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT, index=False)

print(f"Saved: {OUTPUT}")

# ----------------------------------------------------------------------
# 13. FINAL RESULT
# ----------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 23 VALIDATION RESULT")
print("=" * 70)

print()
print(f"SEAD runs validated: {len(df)}")
print(f"SEAD classes: {df['label'].nunique()}")
print(f"Feature columns: {len(feature_columns)}")
print(f"Duplicate run keys: {duplicate_runs}")
print(f"Missing feature values: {total_missing}")
print(f"Infinite feature values: {infinite_values}")
print(f"Constant feature columns: {len(constant_columns)}")

print()
print("CLASS COUNTS:")
for cls in EXPECTED_CLASSES:
    print(f"  {cls:<35} {class_counts.get(cls, 0)}")

print()
print("No FDI data was merged.")
print("fdi_train.csv was not modified.")
print("fdi_test.csv was not modified.")
print("No classifier was trained.")
print("The 8 GB archive was not extracted.")

print()
print("=" * 70)
print("STEP 23 COMPLETE")
print("=" * 70)

