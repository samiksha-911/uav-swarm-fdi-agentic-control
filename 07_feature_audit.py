import pandas as pd
import numpy as np

TRAIN = "fdi_train_specified.csv"
TEST = "fdi_test_specified.csv"

print("=" * 70)
print("STEP 07 - FEATURE / LEAKAGE AUDIT")
print("=" * 70)

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

print("\nTrain shape:", train.shape)
print("Test shape :", test.shape)

# ---------------------------------------------------------
# 1. Target and run information
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("1. TARGET / RUN INFORMATION")
print("=" * 70)

print("\nTraining classes:")
print(train["failure_type"].value_counts())

print("\nTesting classes:")
print(test["failure_type"].value_counts())

print("\nTraining runs:", train["run"].nunique())
print("Testing runs :", test["run"].nunique())

overlap = set(train["run"]) & set(test["run"])
print("Run overlap :", len(overlap))

# ---------------------------------------------------------
# 2. Suspicious columns
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("2. SUSPICIOUS / IDENTIFIER FEATURES")
print("=" * 70)

keywords = [
    "status",
    "failure",
    "run",
    "index",
    "time",
    "timestamp",
    "relative",
    "sample",
    "id"
]

for col in train.columns:
    if any(k in col.lower() for k in keywords):
        print(col)

# ---------------------------------------------------------
# 3. Explicit suspicious feature statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("3. SUSPICIOUS FEATURE STATISTICS")
print("=" * 70)

for col in ["sample_index", "relative_to_failure", "Status", "run"]:
    if col in train.columns:
        print("\n---", col, "---")
        print("Type   :", train[col].dtype)
        print("Unique :", train[col].nunique())

        if train[col].dtype != "object":
            print(train[col].describe())

# ---------------------------------------------------------
# 4. Missing values
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

missing = train.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values in training data.")
else:
    print(missing)

# ---------------------------------------------------------
# 5. Constant features
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("5. CONSTANT FEATURES")
print("=" * 70)

constant = []

for col in train.columns:
    if train[col].nunique(dropna=False) <= 1:
        constant.append(col)

if constant:
    for col in constant:
        print(col)
else:
    print("No constant features.")

# ---------------------------------------------------------
# 6. Near-constant numeric features
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("6. NEAR-CONSTANT FEATURES")
print("=" * 70)

numeric = train.select_dtypes(include=np.number).columns

found = False

for col in numeric:
    counts = train[col].value_counts(dropna=False)

    if len(counts) > 0:
        fraction = counts.iloc[0] / len(train)

        if fraction >= 0.995:
            print(f"{col:30s} {fraction:.5f}")
            found = True

if not found:
    print("No near-constant numeric features.")

# ---------------------------------------------------------
# 7. Class-wise statistics for important features
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("7. CLASS-WISE IMPORTANT FEATURE MEANS")
print("=" * 70)

features = [
    "sample_index",
    "relative_to_failure",
    "CTUN_TAlt",
    "MAV_rxp",
    "IMU_T",
    "CTUN_ThH",
    "BAT_Res",
    "MAV_txp",
    "MOTB_LiftMax",
    "IMU_AccZ",
    "GPS_NSats",
    "XKF3_ErSc",
    "XKF4_GPS",
    "IMU_AccX",
    "IMU_AccY",
    "IMU_AccZ",
    "IMU_GyrX",
    "IMU_GyrY",
    "IMU_GyrZ",
    "BARO_Alt",
    "BARO_Press"
]

available = [c for c in features if c in train.columns]

if available:
    means = train.groupby("failure_type")[available].mean()
    print(means.to_string())

    means.to_csv("class_feature_means.csv")
    print("\nSaved: class_feature_means.csv")

# ---------------------------------------------------------
# 8. Run/class consistency
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("8. RUN / FAILURE CLASS CONSISTENCY")
print("=" * 70)

run_classes = train.groupby("run")["failure_type"].nunique()

print("Runs with exactly one class :", (run_classes == 1).sum())
print("Runs with multiple classes  :", (run_classes > 1).sum())

# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 07 AUDIT COMPLETE")
print("=" * 70)
