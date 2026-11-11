import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

TRAIN = "/home/samiksha/uav_fdi_data/fdi_train.csv"
TEST  = "/home/samiksha/uav_fdi_data/fdi_test.csv"

print("=" * 70)
print("FDI BASELINE CLASSIFIER")
print("=" * 70)

# ---------------------------------------------------------
# Features selected from previous robust-feature analysis
# ---------------------------------------------------------

FEATURES = [
    "XKF5_eVel",
    "PSCD_AD",
    "RATE_A",
    "XKF5_ePos",
    "XKF3_ErSc",
    "XKF4_OFE",
    "XKF4_OFN",
    "XKF3_IPE",
    "XKF2_AX",
    "XKF2_AY",
    "XKF2_AZ",
    "ATT_ErrYaw",
    "XKF5_eAng",
    "XKF4_errRP",
    "XKF4_SV",
    "XKF4_SP",
    "XKF3_IPN",
    "XKF3_IVE",
    "MAV_txp",
    "MAV_rxp",
    "BAT_RemPct",
    "BAT_Res",
    "BAT_VoltR",
    "MOTB_BatVolt",
    "MOTB_LiftMax",
    "IMU_T",
    "XKF1_GX",
    "XKF1_GY",
    "XKF1_GZ",
    "CTUN_ThH",
]

TARGET = "failure_type"

print("Loading training data...")
train = pd.read_csv(TRAIN, usecols=FEATURES + [TARGET, "run"])

print("Loading testing data...")
test = pd.read_csv(TEST, usecols=FEATURES + [TARGET, "run"])

print()
print("Train:", train.shape)
print("Test :", test.shape)
print("Features:", len(FEATURES))

# ---------------------------------------------------------
# Convert to numeric
# ---------------------------------------------------------

X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
X_test  = test[FEATURES].apply(pd.to_numeric, errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test  = X_test.replace([np.inf, -np.inf], np.nan)

# Fill missing values using TRAIN medians only
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test  = X_test.fillna(medians)

y_train = train[TARGET]
y_test = test[TARGET]

print()
print("Training classes:")
print(y_train.value_counts())

# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training complete.")

# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
balanced_acc = balanced_accuracy_score(y_test, pred)

print()
print("Accuracy          :", accuracy)
print("Balanced accuracy :", balanced_acc)

print()
print("Classification report:")
print(
    classification_report(
        y_test,
        pred,
        digits=4
    )
)

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    pred,
    labels=labels
)

print()
print("Confusion matrix:")
print("Labels:")
print(labels)
print(cm)

# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)

print(importance.to_string(index=False))

importance.to_csv(
    "/home/samiksha/uav_fdi_data/baseline_feature_importance.csv",
    index=False
)

# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

predictions = test[["run", TARGET]].copy()
predictions["prediction"] = pred

predictions.to_csv(
    "/home/samiksha/uav_fdi_data/baseline_predictions.csv",
    index=False
)

print()
print("Saved:")
print("/home/samiksha/uav_fdi_data/baseline_feature_importance.csv")
print("/home/samiksha/uav_fdi_data/baseline_predictions.csv")

print()
print("=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)
