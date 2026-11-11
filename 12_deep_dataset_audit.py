import os
import numpy as np
import pandas as pd

print("=" * 70)
print("STEP 12 - DEEP DATASET AUDIT")
print("=" * 70)

BASE = "/home/samiksha/uav_fdi_data"

TRAIN_FILE = os.path.join(BASE, "fdi_train.csv")
TEST_FILE = os.path.join(BASE, "fdi_test.csv")

# ---------------------------------------------------------------------
# 1. LOAD LABELS + DATA
# ---------------------------------------------------------------------

print("\nLoading train/test data...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train shape: {train.shape}")
print(f"Test shape : {test.shape}")

if "failure_type" not in train.columns or "failure_type" not in test.columns:
    raise ValueError("failure_type column not found.")

# ---------------------------------------------------------------------
# 2. CLASS DISTRIBUTION
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("1. CLASS DISTRIBUTION")
print("=" * 70)

train_counts = train["failure_type"].value_counts().sort_index()
test_counts = test["failure_type"].value_counts().sort_index()

classes = sorted(set(train["failure_type"]) | set(test["failure_type"]))

distribution = pd.DataFrame({
    "train": train_counts,
    "test": test_counts
}).fillna(0).astype(int)

distribution["total"] = distribution["train"] + distribution["test"]

distribution["train_pct"] = (
    distribution["train"] / distribution["total"] * 100
)

distribution["test_pct"] = (
    distribution["test"] / distribution["total"] * 100
)

print(distribution)

# ---------------------------------------------------------------------
# 3. LABEL ORDER / DATA ORDER
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("2. LABEL ORDER / DATA ORDER")
print("=" * 70)

print("\nFirst 30 train labels:")
print(train["failure_type"].head(30).tolist())

print("\nLast 30 train labels:")
print(train["failure_type"].tail(30).tolist())

print("\nFirst 30 test labels:")
print(test["failure_type"].head(30).tolist())

print("\nLast 30 test labels:")
print(test["failure_type"].tail(30).tolist())

print("\nTrain consecutive label transitions:")

train_labels = train["failure_type"].to_numpy()

transitions = np.sum(
    train_labels[1:] != train_labels[:-1]
)

print("Number of label transitions:", transitions)

# ---------------------------------------------------------------------
# 4. RUN LENGTHS OF LABELS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("3. LABEL RUN ANALYSIS")
print("=" * 70)

def get_runs(labels):
    runs = []

    start = 0

    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append({
                "label": labels[start],
                "start": start,
                "end": i - 1,
                "length": i - start
            })
            start = i

    return pd.DataFrame(runs)


train_runs = get_runs(train_labels)
test_runs = get_runs(test["failure_type"].to_numpy())

print("\nTrain label runs:")
print(train_runs.head(50).to_string(index=False))

print("\nNumber of train runs:", len(train_runs))

print("\nLargest train runs:")
print(
    train_runs.sort_values("length", ascending=False)
    .head(20)
    .to_string(index=False)
)

# ---------------------------------------------------------------------
# 5. MISSING VALUES
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

train_missing = train.isna().sum()
test_missing = test.isna().sum()

print("\nTrain columns with missing values:")
print(train_missing[train_missing > 0].sort_values(ascending=False))

print("\nTest columns with missing values:")
print(test_missing[test_missing > 0].sort_values(ascending=False))

print(
    "\nTotal train missing values:",
    int(train_missing.sum())
)

print(
    "Total test missing values:",
    int(test_missing.sum())
)

# ---------------------------------------------------------------------
# 6. CONSTANT / NEAR-CONSTANT FEATURES
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("5. CONSTANT / NEAR-CONSTANT FEATURES")
print("=" * 70)

numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()

print("Numeric features:", len(numeric_cols))

constant_features = []

near_constant_features = []

for col in numeric_cols:

    nunique = train[col].nunique(dropna=False)

    if nunique <= 1:
        constant_features.append(col)

    value_counts = train[col].value_counts(normalize=True, dropna=False)

    if len(value_counts) > 0 and value_counts.iloc[0] >= 0.999:
        near_constant_features.append(
            (col, value_counts.iloc[0])
        )

print("\nConstant features:")
print(constant_features)

print("\nNear-constant features (>=99.9% same value):")

for col, pct in near_constant_features:
    print(f"{col:30s} {pct * 100:.4f}%")

# ---------------------------------------------------------------------
# 7. TRAIN/TEST FEATURE STATISTICS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("6. TRAIN / TEST FEATURE DISTRIBUTION")
print("=" * 70)

stats = []

for col in numeric_cols:

    tr = train[col]
    te = test[col]

    stats.append({
        "feature": col,

        "train_mean": tr.mean(),
        "test_mean": te.mean(),

        "train_std": tr.std(),
        "test_std": te.std(),

        "train_min": tr.min(),
        "test_min": te.min(),

        "train_max": tr.max(),
        "test_max": te.max(),

        "train_nan": tr.isna().sum(),
        "test_nan": te.isna().sum()
    })

stats_df = pd.DataFrame(stats)

stats_df["mean_abs_difference"] = (
    stats_df["train_mean"] - stats_df["test_mean"]
).abs()

stats_df["std_abs_difference"] = (
    stats_df["train_std"] - stats_df["test_std"]
).abs()

print("\nFeatures with largest train/test mean differences:")

print(
    stats_df
    .sort_values("mean_abs_difference", ascending=False)
    .head(30)
    .to_string(index=False)
)

stats_df.to_csv(
    os.path.join(BASE, "step12_train_test_feature_stats.csv"),
    index=False
)

# ---------------------------------------------------------------------
# 8. CLASS-WISE FEATURE MEANS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("7. CLASS-WISE FEATURE MEANS")
print("=" * 70)

class_means = train.groupby("failure_type")[numeric_cols].mean()

print("\nSelected class-wise feature means:")

important_check_features = [
    "IMU_AccX",
    "IMU_AccY",
    "IMU_AccZ",
    "IMU_GyrX",
    "IMU_GyrY",
    "IMU_GyrZ",
    "MAG_MagX",
    "MAG_MagY",
    "MAG_MagZ",
    "BARO_Alt",
    "BARO_Press",
    "BARO_Temp",
    "GPS_NSats",
    "GPS_Alt",
    "GPS_Spd",
    "GPS_VZ",
    "MAV_rxp",
    "MAV_txp",
    "XKF3_ErSc",
    "XKF4_GPS",
    "XKF4_errRP",
    "ATT_ErrRP",
    "ATT_ErrYaw",
    "CTUN_Alt",
    "CTUN_TAlt",
    "CTUN_DAlt",
    "VIBE_VibeX",
    "VIBE_VibeY",
    "VIBE_VibeZ",
    "IMU_T",
    "BAT_Res",
    "BAT_CurrTot",
    "BAT_EnrgTot",
    "BAT_RemPct"
]

available_check_features = [
    x for x in important_check_features
    if x in class_means.columns
]

print(
    class_means[available_check_features]
    .round(4)
    .to_string()
)

class_means.to_csv(
    os.path.join(BASE, "step12_class_feature_means.csv")
)

# ---------------------------------------------------------------------
# 9. CLASS SEPARABILITY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("8. CLASS SEPARABILITY")
print("=" * 70)

separability = []

for col in numeric_cols:

    grouped = train.groupby("failure_type")[col]

    means = grouped.mean()
    stds = grouped.std()

    if len(means) < 2:
        continue

    overall_std = train[col].std()

    if pd.isna(overall_std) or overall_std == 0:
        continue

    between_std = means.std()

    ratio = between_std / overall_std

    separability.append({
        "feature": col,
        "between_class_std": between_std,
        "overall_std": overall_std,
        "separability_ratio": ratio
    })

sep_df = pd.DataFrame(separability)

print("\nMost class-separating features:")

print(
    sep_df
    .sort_values("separability_ratio", ascending=False)
    .head(40)
    .to_string(index=False)
)

sep_df.to_csv(
    os.path.join(BASE, "step12_class_separability.csv"),
    index=False
)

# ---------------------------------------------------------------------
# 10. DUPLICATE ROW CHECK
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("9. DUPLICATE ROW CHECK")
print("=" * 70)

print("\nChecking duplicate rows...")

train_duplicates = train.duplicated().sum()
test_duplicates = test.duplicated().sum()

print("Train duplicate rows:", train_duplicates)
print("Test duplicate rows :", test_duplicates)

# ---------------------------------------------------------------------
# 11. CROSS TRAIN/TEST DUPLICATES
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("10. TRAIN/TEST OVERLAP CHECK")
print("=" * 70)

print(
    "\nChecking whether complete feature rows appear in both "
    "train and test..."
)

# Hash numeric feature rows instead of constructing huge string rows.
# This is much more memory efficient.

train_numeric = train[numeric_cols].copy()
test_numeric = test[numeric_cols].copy()

train_hash = pd.util.hash_pandas_object(
    train_numeric,
    index=False
)

test_hash = pd.util.hash_pandas_object(
    test_numeric,
    index=False
)

train_hash_set = set(train_hash.values)

overlap_count = sum(
    h in train_hash_set for h in test_hash.values
)

print(
    "Exact numeric feature rows appearing in both train/test:",
    overlap_count
)

# ---------------------------------------------------------------------
# 12. LABEL CONSISTENCY FOR DUPLICATES
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("11. DUPLICATE LABEL CONSISTENCY")
print("=" * 70)

all_data = pd.concat(
    [
        train[["failure_type"] + numeric_cols],
        test[["failure_type"] + numeric_cols]
    ],
    ignore_index=True
)

all_hash = pd.util.hash_pandas_object(
    all_data[numeric_cols],
    index=False
)

all_data["_hash"] = all_hash

duplicate_groups = (
    all_data
    .groupby("_hash")["failure_type"]
    .agg(["count", "nunique"])
)

conflicting_duplicates = duplicate_groups[
    (duplicate_groups["count"] > 1) &
    (duplicate_groups["nunique"] > 1)
]

print(
    "Repeated feature rows with conflicting labels:",
    len(conflicting_duplicates)
)

if len(conflicting_duplicates) > 0:
    print("\nWARNING: conflicting duplicate feature rows detected.")

# ---------------------------------------------------------------------
# 13. CLASS FEATURE VARIANCE
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("12. CLASS-WISE VARIANCE CHECK")
print("=" * 70)

class_variance = train.groupby("failure_type")[numeric_cols].var()

low_variance_results = []

for cls in class_variance.index:

    for col in numeric_cols:

        value = class_variance.loc[cls, col]

        if pd.notna(value) and value == 0:
            low_variance_results.append({
                "failure_type": cls,
                "feature": col
            })

print(
    "Class/feature combinations with exactly zero variance:",
    len(low_variance_results)
)

if low_variance_results:
    print(
        pd.DataFrame(low_variance_results)
        .head(100)
        .to_string(index=False)
    )

# ---------------------------------------------------------------------
# 14. FAILURE SIGNATURE FILES
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("13. FAILURE SIGNATURE FILE CHECK")
print("=" * 70)

signature_files = [
    "failure_signature_means.csv",
    "failure_signature_std.csv",
    "failure_signature_difference.csv"
]

for filename in signature_files:

    path = os.path.join(BASE, filename)

    if os.path.exists(path):

        df = pd.read_csv(path)

        print(
            f"{filename:35s} "
            f"shape={df.shape}"
        )

        print("Columns:", df.columns.tolist()[:10])

    else:

        print(f"{filename:35s} NOT FOUND")

# ---------------------------------------------------------------------
# 15. FINAL DIAGNOSIS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("14. AUTOMATIC DIAGNOSTIC SUMMARY")
print("=" * 70)

rc_test = int(test_counts.get("RC Failure", 0))

if rc_test == 0:
    print(
        "\n[CRITICAL] RC Failure has ZERO test samples."
    )
else:
    print(
        "\n[OK] RC Failure exists in the test set."
    )

if len(classes) != 7:
    print(
        "[WARNING] Expected 7 failure classes, found:",
        len(classes)
    )
else:
    print(
        "[OK] All 7 expected failure classes are present "
        "across train/test."
    )

if train_duplicates > 0 or test_duplicates > 0:
    print(
        "[WARNING] Duplicate rows exist inside train/test."
    )
else:
    print(
        "[OK] No exact duplicate rows inside individual splits."
    )

if overlap_count > 0:
    print(
        "[WARNING] Exact numeric feature overlap exists "
        "between train/test."
    )
else:
    print(
        "[OK] No exact numeric feature overlap detected."
    )

if len(conflicting_duplicates) > 0:
    print(
        "[CRITICAL] Some repeated feature rows have "
        "different labels."
    )
else:
    print(
        "[OK] No conflicting labels detected for duplicate "
        "feature rows."
    )

print("\n" + "=" * 70)
print("STEP 12 COMPLETE")
print("=" * 70)

print("\nGenerated files:")

for filename in [
    "step12_train_test_feature_stats.csv",
    "step12_class_feature_means.csv",
    "step12_class_separability.csv"
]:

    print(
        os.path.join(BASE, filename)
    )

print("\nDo NOT retrain the model yet.")
print("Use this audit to determine the correct split/model strategy.")
