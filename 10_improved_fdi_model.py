import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings("ignore")

print("=" * 70)
print("STEP 10 - IMPROVED FDI MODEL")
print("=" * 70)

# ================================================================
# PATHS
# ================================================================

BASE_DIR = "/home/samiksha/uav_fdi_data"

TRAIN_FILE = os.path.join(BASE_DIR, "fdi_train.csv")
TEST_FILE = os.path.join(BASE_DIR, "fdi_test.csv")

MEANS_FILE = os.path.join(
    BASE_DIR, "failure_signature_means.csv"
)

DIFF_FILE = os.path.join(
    BASE_DIR, "failure_signature_difference.csv"
)

OUTPUT_MODEL = os.path.join(
    BASE_DIR, "improved_fdi_predictions.csv"
)

OUTPUT_IMPORTANCE = os.path.join(
    BASE_DIR, "improved_fdi_feature_importance.csv"
)

OUTPUT_CM = os.path.join(
    BASE_DIR, "improved_fdi_confusion_matrix.csv"
)

# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading data...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("Train:", train.shape)
print("Test :", test.shape)

# ================================================================
# TARGET
# ================================================================

TARGET = "failure_type"

if TARGET not in train.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in training data."
    )

if TARGET not in test.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in test data."
    )

y_train = train[TARGET].astype(str)
y_test = test[TARGET].astype(str)

print("\nFailure classes:")
for cls, count in y_train.value_counts().sort_index().items():
    print(f"{cls:25s}: {count:,}")

# ================================================================
# NUMERIC FEATURES
# ================================================================

numeric_features = train.select_dtypes(
    include=[np.number]
).columns.tolist()

if TARGET in numeric_features:
    numeric_features.remove(TARGET)

print("\nNumeric telemetry features:", len(numeric_features))

# Make sure test contains exactly the same usable features
numeric_features = [
    f for f in numeric_features
    if f in test.columns
]

# ================================================================
# FAILURE-SPECIFIC FEATURES
# SELECTED FROM STEP 09
# ================================================================

signature_features = [
    # Accelerometer / Gyro
    "IMU_AccX",
    "IMU_AccY",
    "IMU_AccZ",
    "IMU_GyrX",
    "IMU_GyrY",
    "IMU_GyrZ",

    # Magnetometer / Compass
    "MAG_MagX",
    "MAG_MagY",
    "MAG_MagZ",

    # Barometer
    "BARO_Alt",
    "BARO_Press",
    "BARO_Temp",

    # GPS
    "GPS_NSats",
    "GPS_Alt",
    "GPS_Spd",
    "GPS_VZ",

    # MAVLink / RC
    "MAV_rxp",
    "MAV_txp",

    # Estimator
    "XKF3_ErSc",
    "XKF4_GPS",
    "XKF4_errRP",

    # Attitude errors
    "ATT_ErrRP",
    "ATT_ErrYaw",

    # Control / altitude
    "CTUN_Alt",
    "CTUN_TAlt",
    "CTUN_DAlt",

    # Vibration
    "VIBE_VibeX",
    "VIBE_VibeY",
    "VIBE_VibeZ",

    # Temperature
    "IMU_T",

    # Battery
    "BAT_Res",
    "BAT_CurrTot",
    "BAT_EnrgTot",
    "BAT_RemPct",
]

signature_features = [
    f for f in signature_features
    if f in numeric_features
]

print("\nFailure-signature features available:",
      len(signature_features))

print("\nSelected features:")
for f in signature_features:
    print("  ", f)

# ================================================================
# ADDITIONAL FEATURES FROM STEP 09
#
# Use the strongest features automatically.
# ================================================================

print("\nLoading Step 09 feature signatures...")

try:
    diff = pd.read_csv(DIFF_FILE)

    # Find feature column
    if "feature" in diff.columns:
        diff_features = diff["feature"].astype(str).tolist()
    else:
        diff_features = []

    # Add features that actually exist in the data
    extra_features = [
        f for f in diff_features
        if f in numeric_features
        and f not in signature_features
    ]

    # Limit additions so the model does not simply use every feature
    extra_features = extra_features[:25]

    selected_features = signature_features + extra_features

except Exception as e:
    print("Could not load difference file:", e)
    selected_features = signature_features

# Remove duplicates while preserving order
selected_features = list(dict.fromkeys(selected_features))

print("\nFINAL FEATURE COUNT:", len(selected_features))

# ================================================================
# PREPARE X
# ================================================================

X_train = train[selected_features].copy()
X_test = test[selected_features].copy()

# Replace infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill missing values using training medians
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("\nTraining matrix:", X_train.shape)
print("Testing matrix :", X_test.shape)

# ================================================================
# RANDOM FOREST
# ================================================================

print("\n" + "=" * 70)
print("TRAINING IMPROVED RANDOM FOREST")
print("=" * 70)

start = time.time()

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

train_time = time.time() - start

print(f"\nTraining completed in {train_time:.2f} seconds.")

# ================================================================
# PREDICTION
# ================================================================

print("\nGenerating predictions...")

start = time.time()

y_pred = model.predict(X_test)

prediction_time = time.time() - start

print(f"Prediction completed in {prediction_time:.2f} seconds.")

# ================================================================
# OVERALL METRICS
# ================================================================

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

print(f"Accuracy    : {accuracy:.6f}")
print(f"Macro F1    : {macro_f1:.6f}")
print(f"Weighted F1 : {weighted_f1:.6f}")

# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0
)

print(report)

# ================================================================
# CONFUSION MATRIX
# ================================================================

classes = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_df.to_csv(OUTPUT_CM)

print("Saved confusion matrix:")
print(OUTPUT_CM)

print("\nConfusion Matrix:")
print(cm_df)

# ================================================================
# PER-CLASS PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for cls in classes:

    mask = y_test == cls

    class_total = mask.sum()
    class_correct = (
        (y_pred[mask] == cls).sum()
    )

    class_accuracy = (
        class_correct / class_total
        if class_total > 0 else 0
    )

    print(
        f"{cls:25s} "
        f"Correct: {class_correct:7,d} / {class_total:7,d} "
        f"({class_accuracy * 100:7.3f}%)"
    )

# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({
    "feature": selected_features,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_IMPORTANCE,
    index=False
)

print("\nTop 30 features:")

print(
    importance_df.head(30).to_string(index=False)
)

print("\nSaved feature importance:")
print(OUTPUT_IMPORTANCE)

# ================================================================
# SAVE PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame({
    "actual_failure": y_test,
    "predicted_failure": y_pred
})

prediction_df.to_csv(
    OUTPUT_MODEL,
    index=False
)

print("\nSaved predictions:")
print(OUTPUT_MODEL)

# ================================================================
# ERROR ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

errors = prediction_df[
    prediction_df["actual_failure"]
    != prediction_df["predicted_failure"]
]

print(
    f"\nTotal test samples : {len(prediction_df):,}"
)

print(
    f"Incorrect           : {len(errors):,}"
)

print(
    f"Error rate          : "
    f"{len(errors) / len(prediction_df) * 100:.4f}%"
)

if len(errors) > 0:

    print("\nMost common misclassifications:")

    error_pairs = (
        errors
        .groupby([
            "actual_failure",
            "predicted_failure"
        ])
        .size()
        .sort_values(ascending=False)
        .head(20)
    )

    for (actual, predicted), count in error_pairs.items():

        print(
            f"{actual:25s} -> "
            f"{predicted:25s}: "
            f"{count:,}"
        )

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("STEP 10 ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nAccuracy    : {accuracy:.6f}")
print(f"Macro F1    : {macro_f1:.6f}")
print(f"Weighted F1 : {weighted_f1:.6f}")

print("\nOutput files:")
print(OUTPUT_MODEL)
print(OUTPUT_IMPORTANCE)
print(OUTPUT_CM)

print("\nNext step:")
print("Compare this model against the previous FDI model.")
