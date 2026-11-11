import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# PATH
# ============================================================

DATASET = "/home/samiksha/uav_fdi_data/step48_day2_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET)

print("=" * 70)
print("STEP 49 — PX4-NATIVE ML-FDI")
print("=" * 70)

print()
print("Dataset shape:", df.shape)


# ============================================================
# METADATA
# ============================================================

META_COLUMNS = [
    "run_id",
    "window_number",
    "label",
]

X = df.drop(columns=META_COLUMNS)
y = df["label"]
groups = df["run_id"]


print()
print("ML feature count:", X.shape[1])
print("Samples/windows:", len(df))

print()
print("Classes:")
print(y.value_counts())

print()
print("Independent runs:")
print(groups.nunique())

print()
print(groups.value_counts())


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [
    c for c in X.columns
    if X[c].nunique(dropna=False) <= 1
]

print()
print("Constant features:", len(constant_features))

X = X.drop(columns=constant_features)

print("Remaining features:", X.shape[1])


# ============================================================
# LEAVE-ONE-RUN-OUT
# ============================================================

logo = LeaveOneGroupOut()

all_true = []
all_pred = []

fold_results = []


print()
print("=" * 70)
print("LEAVE-ONE-RUN-OUT EVALUATION")
print("=" * 70)


for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    start=1
):

    train_runs = groups.iloc[train_idx].unique()
    test_run = groups.iloc[test_idx].unique()

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predict held-out recording
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    all_true.extend(y_test.tolist())
    all_pred.extend(y_pred.tolist())

    fold_accuracy = accuracy_score(y_test, y_pred)

    fold_results.append({
        "fold": fold,
        "test_run": test_run[0],
        "test_windows": len(test_idx),
        "accuracy": fold_accuracy,
        "true_labels": ",".join(sorted(y_test.unique())),
    })

    print()
    print(
        f"Fold {fold}: "
        f"test_run={test_run[0]}, "
        f"windows={len(test_idx)}, "
        f"accuracy={fold_accuracy:.4f}"
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

all_true = np.array(all_true)
all_pred = np.array(all_pred)

accuracy = accuracy_score(all_true, all_pred)

balanced_accuracy = balanced_accuracy_score(
    all_true,
    all_pred
)


print()
print("=" * 70)
print("STEP 49 RESULTS")
print("=" * 70)

print()
print(f"Overall accuracy:          {accuracy:.4f}")
print(f"Overall balanced accuracy: {balanced_accuracy:.4f}")

print()
print("Classification report:")
print(
    classification_report(
        all_true,
        all_pred,
        zero_division=0
    )
)

print()
print("Confusion matrix:")

classes = sorted(y.unique())

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=classes
)

print()
print("Labels:", classes)
print(cm)


# ============================================================
# FOLD SUMMARY
# ============================================================

results_df = pd.DataFrame(fold_results)

print()
print("Fold results:")
print(results_df.to_string(index=False))


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

oof = df[
    [
        "run_id",
        "window_number",
        "label",
    ]
].copy()

oof["prediction"] = all_pred

OOF_FILE = "/home/samiksha/uav_fdi_data/step49_px4_native_oof_predictions.csv"

oof.to_csv(
    OOF_FILE,
    index=False
)

print()
print("Saved OOF predictions:")
print(OOF_FILE)


# ============================================================
# FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING FINAL PX4-NATIVE MODEL")
print("=" * 70)

final_model = ExtraTreesClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    max_features="sqrt",
)

final_model.fit(X, y)


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

MODEL_FILE = "/home/samiksha/uav_fdi_data/px4_native_fdi_model.joblib"

bundle = {
    "model": final_model,
    "features": list(X.columns),
    "label_column": "label",
    "run_column": "run_id",
    "window_size": 50,
    "feature_method": [
        "mean",
        "std",
        "min",
        "max",
        "range",
    ],
    "classes": list(final_model.classes_),
    "n_features": X.shape[1],
    "training_runs": sorted(groups.unique()),
    "description": (
        "PX4-native telemetry FDI model. "
        "Features are computed from 50-sample windows "
        "using mean/std/min/max/range."
    ),
}

joblib.dump(
    bundle,
    MODEL_FILE
)

print()
print("Saved model:")
print(MODEL_FILE)

print()
print("=" * 70)
print("STEP 49 COMPLETE")
print("=" * 70)
