import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

TRAIN = "/home/samiksha/uav_fdi_data/fdi_train_specified.csv"
TEST = "/home/samiksha/uav_fdi_data/fdi_test_specified.csv"

IMPORTANCE_OUT = "/home/samiksha/uav_fdi_data/clean_baseline_feature_importance.csv"
PREDICTIONS_OUT = "/home/samiksha/uav_fdi_data/clean_baseline_predictions.csv"

print("=" * 70)
print("FDI CLEAN BASELINE - SPECIFIED RUN SPLIT")
print("=" * 70)

print("Loading training data...")
train = pd.read_csv(TRAIN)

print("Loading testing data...")
test = pd.read_csv(TEST)

print("\nTrain:", train.shape)
print("Test :", test.shape)

TARGET = "failure_type"

y_train = train[TARGET]
y_test = test[TARGET]
DROP_COLUMNS = [
    "Status",
    "failure_type",
    "run",
    "sample_index",
    "relative_to_failure",
    "MAG_OfsX",
    "MAG_OfsY",
    "MAG_OfsZ",
    "MOTB_FailFlags",
    "XKF5_offset",
    "XKF5_Herr",
]

print("\n" + "=" * 70)
print("REMOVING TARGET / ID / LEAKAGE-PRONE FEATURES")
print("=" * 70)

for col in DROP_COLUMNS:
    print(col)

missing_columns = [c for c in DROP_COLUMNS if c not in train.columns]

if missing_columns:
    print("\nERROR: Missing expected columns:")
    for col in missing_columns:
        print(col)
    raise SystemExit(1)

X_train = train.drop(columns=DROP_COLUMNS)
X_test = test.drop(columns=DROP_COLUMNS)

print("\nClean train:", X_train.shape)
print("Clean test :", X_test.shape)

if list(X_train.columns) != list(X_test.columns):
    print("\nERROR: Train/test feature columns do not match.")
    raise SystemExit(1)

print("Train/test feature columns match: YES")

non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric:
    print("\nERROR: Non-numeric features found:")
    for col in non_numeric:
        print(col)
    raise SystemExit(1)

print("\nNumber of clean features:", X_train.shape[1])
print("\n" + "=" * 70)
print("MISSING VALUE HANDLING")
print("=" * 70)

print("Train missing values before:", X_train.isna().sum().sum())
print("Test missing values before :", X_test.isna().sum().sum())

train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print("Train missing values after :", X_train.isna().sum().sum())
print("Test missing values after  :", X_test.isna().sum().sum())

if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
    print("\nERROR: Missing values remain.")
    raise SystemExit(1)

print("\n" + "=" * 70)
print("TRAINING CLASSES")
print("=" * 70)
print(y_train.value_counts())

print("\n" + "=" * 70)
print("TEST CLASSES")
print("=" * 70)
print(y_test.value_counts())

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("CLEAN BASELINE TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy          : {accuracy:.6f}")
print(f"Balanced accuracy : {balanced_accuracy:.6f}")

labels = sorted(y_test.unique())

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion matrix:")
print("Labels:")
print(labels)
print(cm)
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)

print(importance.to_string(index=False))

importance.to_csv(IMPORTANCE_OUT, index=False)

predictions = pd.DataFrame({
    "true_failure_type": y_test.values,
    "predicted_failure_type": y_pred
})

predictions.to_csv(PREDICTIONS_OUT, index=False)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(IMPORTANCE_OUT)
print(PREDICTIONS_OUT)

print("\n" + "=" * 70)
print("CLEAN BASELINE COMPLETE")
print("=" * 70)
