import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

TRAIN_INPUT = "/home/samiksha/uav_fdi_data/fdi_train_specified.csv"
TEST_INPUT  = "/home/samiksha/uav_fdi_data/fdi_test_specified.csv"

IMPORTANCE_OUTPUT = "/home/samiksha/uav_fdi_data/baseline_specified_feature_importance.csv"
PREDICTIONS_OUTPUT = "/home/samiksha/uav_fdi_data/baseline_specified_predictions.csv"

print("=" * 70)
print("FDI BASELINE CLASSIFIER - SPECIFIED RUN SPLIT")
print("=" * 70)

# =========================================================
# LOAD DATA
# =========================================================

print("Loading training data...")
train = pd.read_csv(TRAIN_INPUT)

print("Loading testing data...")
test = pd.read_csv(TEST_INPUT)

print()
print("Train:", train.shape)
print("Test :", test.shape)

# =========================================================
# TARGET
# =========================================================

TARGET = "failure_type"

# Columns that must NOT be used as ML features
DROP_COLUMNS = [
    "failure_type",
    "Status",
    "run"
]

feature_columns = [
    c for c in train.columns
    if c not in DROP_COLUMNS
]

X_train = train[feature_columns]
y_train = train[TARGET]

X_test = test[feature_columns]
y_test = test[TARGET]

print("Features:", len(feature_columns))

# =========================================================
# CHECK FOR NON-NUMERIC FEATURES
# =========================================================

non_numeric = X_train.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric:
    print()
    print("ERROR: Non-numeric features detected:")
    for col in non_numeric:
        print(" ", col)
    raise SystemExit(1)

# =========================================================
# TRAINING DISTRIBUTION
# =========================================================

print()
print("=" * 70)
print("TRAINING CLASSES")
print("=" * 70)

print(y_train.value_counts())

# =========================================================
# TEST DISTRIBUTION
# =========================================================

print()
print("=" * 70)
print("TEST CLASSES")
print("=" * 70)

print(y_test.value_counts())

# =========================================================
# RANDOM FOREST
# =========================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

model.fit(X_train, y_train)

print("Training complete.")

# =========================================================
# PREDICTION
# =========================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

print()
print(f"Accuracy          : {accuracy:.6f}")
print(f"Balanced accuracy : {balanced_accuracy:.6f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

labels = sorted(
    set(y_test.unique()) | set(y_pred)
)

print()
print("Classification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print()
print("Confusion matrix:")
print("Labels:")
print(labels)
print(cm)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print()
print("=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(index=False)
)

# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

importance_df.to_csv(
    IMPORTANCE_OUTPUT,
    index=False
)

# =========================================================
# SAVE PREDICTIONS
# =========================================================

predictions = test[
    ["run", "failure_type", "Status"]
].copy()

predictions["predicted_failure_type"] = y_pred

predictions.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)

# =========================================================
# SAVE COMPLETE
# =========================================================

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(IMPORTANCE_OUTPUT)
print(PREDICTIONS_OUTPUT)

print()
print("=" * 70)
print("SPECIFIED BASELINE COMPLETE")
print("=" * 70)
