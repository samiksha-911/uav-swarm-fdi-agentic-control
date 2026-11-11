import pandas as pd
import numpy as np

TRAIN = "/home/samiksha/uav_fdi_data/fdi_train_specified.csv"
TEST = "/home/samiksha/uav_fdi_data/fdi_test_specified.csv"

OUTPUT_MEANS = "/home/samiksha/uav_fdi_data/failure_signature_means.csv"
OUTPUT_STD = "/home/samiksha/uav_fdi_data/failure_signature_std.csv"
OUTPUT_DIFF = "/home/samiksha/uav_fdi_data/failure_signature_difference.csv"

print("=" * 70)
print("STEP 09 - FAILURE SIGNATURE ANALYSIS")
print("=" * 70)

print("\nLoading data...")
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

print("Train:", train.shape)
print("Test :", test.shape)

# =========================================================
# TARGET
# =========================================================

TARGET = "failure_type"

classes = sorted(train[TARGET].dropna().unique())

print("\n" + "=" * 70)
print("FAILURE CLASSES")
print("=" * 70)

for c in classes:
    print(f"{c:25s}: {len(train[train[TARGET] == c]):,}")

# =========================================================
# SENSOR / TELEMETRY FEATURES
# =========================================================

exclude = {
    "Status",
    "failure_type",
    "run",
    "sample_index",
    "relative_to_failure"
}

features = [
    c for c in train.columns
    if c not in exclude
    and pd.api.types.is_numeric_dtype(train[c])
]

print("\nNumber of numeric telemetry features:", len(features))

# =========================================================
# CLASS MEANS
# =========================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS MEANS")
print("=" * 70)

means = train.groupby(TARGET)[features].mean()

means.to_csv(OUTPUT_MEANS)

print("Saved:", OUTPUT_MEANS)

# =========================================================
# CLASS STANDARD DEVIATIONS
# =========================================================

print("\nCalculating class standard deviations...")

stds = train.groupby(TARGET)[features].std()

stds.to_csv(OUTPUT_STD)

print("Saved:", OUTPUT_STD)

# =========================================================
# GLOBAL MEAN
# =========================================================

global_mean = train[features].mean()
global_std = train[features].std()

# =========================================================
# STANDARDIZED CLASS DIFFERENCE
# =========================================================
# This estimates how strongly each feature differs from
# the overall dataset.
#
# Larger absolute values = stronger class-specific signal.
# =========================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS-SPECIFIC SIGNAL STRENGTH")
print("=" * 70)

all_results = []

for cls in classes:

    class_mean = means.loc[cls]

    # Avoid division by zero
    denom = global_std.replace(0, np.nan)

    z = (class_mean - global_mean) / denom

    temp = pd.DataFrame({
        "feature": features,
        "class": cls,
        "class_mean": class_mean.values,
        "global_mean": global_mean.values,
        "global_std": global_std.values,
        "standardized_difference": z.values,
        "absolute_difference": np.abs(z.values)
    })

    all_results.append(temp)

difference = pd.concat(all_results, ignore_index=True)

difference = difference.sort_values(
    ["class", "absolute_difference"],
    ascending=[True, False]
)

difference.to_csv(OUTPUT_DIFF, index=False)

print("Saved:", OUTPUT_DIFF)

# =========================================================
# TOP FEATURES FOR EACH FAILURE
# =========================================================

print("\n" + "=" * 70)
print("TOP 15 FEATURES FOR EACH CLASS")
print("=" * 70)

for cls in classes:

    print("\n" + "-" * 70)
    print(cls)
    print("-" * 70)

    subset = difference[
        difference["class"] == cls
    ].head(15)

    print(
        subset[
            [
                "feature",
                "class_mean",
                "global_mean",
                "standardized_difference"
            ]
        ].to_string(index=False)
    )

# =========================================================
# SPECIFIC INVESTIGATION
# =========================================================

print("\n" + "=" * 70)
print("TARGETED INVESTIGATION")
print("=" * 70)

target_features = [
    "MAV_rxp",
    "MAV_txp",
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
    "VIBE_VibeZ"
]

available = [
    f for f in target_features
    if f in train.columns
]

print("\nTargeted features available:", len(available))

target_means = train.groupby(TARGET)[available].mean()

print("\n")
print(target_means.to_string())

# =========================================================
# RANGE / VARIABILITY CHECK
# =========================================================

print("\n" + "=" * 70)
print("VARIABILITY CHECK FOR TARGETED FEATURES")
print("=" * 70)

for feature in available:

    print("\n" + "-" * 60)
    print("FEATURE:", feature)
    print("-" * 60)

    summary = train.groupby(TARGET)[feature].agg(
        ["mean", "std", "min", "max"]
    )

    print(summary.to_string())

# =========================================================
# MOST DISCRIMINATIVE FEATURES FOR THE 3 PROBLEM CLASSES
# =========================================================

problem_classes = [
    "RC Failure",
    "Compass Failure",
    "Barometer Failure"
]

print("\n" + "=" * 70)
print("PROBLEM CLASS ANALYSIS")
print("=" * 70)

for cls in problem_classes:

    if cls not in classes:
        print("\nClass not found:", cls)
        continue

    print("\n" + "-" * 70)
    print(cls)
    print("-" * 70)

    subset = difference[
        difference["class"] == cls
    ].head(30)

    print(
        subset[
            [
                "feature",
                "class_mean",
                "global_mean",
                "standardized_difference"
            ]
        ].to_string(index=False)
    )

# =========================================================
# TRAIN / TEST FEATURE CONSISTENCY
# =========================================================

print("\n" + "=" * 70)
print("TRAIN / TEST FEATURE CONSISTENCY")
print("=" * 70)

train_features = set(features)

test_features = set(
    c for c in test.columns
    if c not in exclude
    and pd.api.types.is_numeric_dtype(test[c])
)

print("Train numeric features:", len(train_features))
print("Test numeric features :", len(test_features))

print("Missing from test:", sorted(train_features - test_features))
print("Extra in test     :", sorted(test_features - train_features))

# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("STEP 09 ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput files:")
print(OUTPUT_MEANS)
print(OUTPUT_STD)
print(OUTPUT_DIFF)

print("\nNext step:")
print("Use these results to identify failure-specific features")
print("before building the improved FDI model.")
