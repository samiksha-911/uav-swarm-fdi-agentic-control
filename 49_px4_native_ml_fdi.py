#!/usr/bin/env python3

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/home/samiksha/uav_fdi_data"

DATASET_FILE = os.path.join(
    BASE,
    "day2_px4_native_window_dataset.csv"
)

MODEL_FILE = os.path.join(
    BASE,
    "px4_native_fdi_model.joblib"
)

OOF_FILE = os.path.join(
    BASE,
    "step49_px4_native_oof_predictions.csv"
)

RANDOM_STATE = 42

N_ESTIMATORS_EVAL = 300
N_ESTIMATORS_FINAL = 500

# ------------------------------------------------------------
# IMPORTANT:
# Prevent one very long recording from dominating evaluation.
# Each independent run contributes at most this many windows.
# ------------------------------------------------------------

MAX_WINDOWS_PER_RUN = 10


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 49 — PX4-NATIVE ML-FDI")
print("=" * 70)

df = pd.read_csv(DATASET_FILE)

print()
print(f"Original dataset shape: {df.shape}")

required_columns = [
    "run_id",
    "window_number",
    "label"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# RUN-LEVEL WINDOW BALANCING
# ============================================================

print()
print("=" * 70)
print("RUN-LEVEL WINDOW BALANCING")
print("=" * 70)

print()
print("Original windows per run:")

print(
    df.groupby("run_id")
      .size()
      .sort_values(ascending=False)
      .to_string()
)


balanced_parts = []

for run_id, run_df in df.groupby(
    "run_id",
    sort=True
):

    run_df = run_df.copy()

    n_windows = len(run_df)

    if n_windows > MAX_WINDOWS_PER_RUN:

        # Deterministic sampling.
        run_df = run_df.sample(
            n=MAX_WINDOWS_PER_RUN,
            random_state=RANDOM_STATE
        )

        # Restore chronological order.
        run_df = run_df.sort_values(
            "window_number"
        )

        print(
            f"{run_id}: "
            f"{n_windows} -> "
            f"{MAX_WINDOWS_PER_RUN} windows"
        )

    else:

        print(
            f"{run_id}: "
            f"{n_windows} -> "
            f"{n_windows} windows"
        )

    balanced_parts.append(
        run_df
    )


df = pd.concat(
    balanced_parts,
    ignore_index=True
)

df = df.sort_values(
    ["run_id", "window_number"]
).reset_index(
    drop=True
)

print()
print(f"Balanced dataset shape: {df.shape}")


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("Balanced class distribution:")

print(
    df["label"]
      .value_counts()
      .to_string()
)

print()
print("Balanced windows per run:")

print(
    df.groupby(
        ["run_id", "label"]
    ).size()
     .reset_index(name="windows")
     .to_string(index=False)
)


# ============================================================
# BUILD ML MATRIX
# ============================================================

metadata_columns = [
    "run_id",
    "window_number",
    "label"
]

feature_columns = [
    c for c in df.columns
    if c not in metadata_columns
]

X = df[feature_columns].copy()

y = df["label"].copy()

groups = df["run_id"].copy()


print()
print("=" * 70)
print("FEATURE PREPARATION")
print("=" * 70)

print()
print(f"ML feature count: {len(feature_columns)}")
print(f"Samples/windows:  {len(df)}")
print(f"Independent runs: {groups.nunique()}")


# ============================================================
# CLEAN NUMERIC FEATURES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

missing_count = int(
    X.isna().sum().sum()
)

if missing_count > 0:

    print()
    print(
        f"Missing feature values before filling: "
        f"{missing_count}"
    )

    medians = X.median()

    X = X.fillna(
        medians
    )

else:

    medians = X.median()

    print()
    print("Missing feature values: 0")


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [
    c for c in X.columns
    if X[c].nunique(dropna=False) <= 1
]

X = X.drop(
    columns=constant_features
)

remaining_features = list(
    X.columns
)

print(
    f"Constant features: "
    f"{len(constant_features)}"
)

print(
    f"Remaining features: "
    f"{len(remaining_features)}"
)


# ============================================================
# LEAVE-ONE-RUN-OUT EVALUATION
# ============================================================

print()
print("=" * 70)
print("LEAVE-ONE-RUN-OUT EVALUATION")
print("=" * 70)

logo = LeaveOneGroupOut()

oof_predictions = np.empty(
    len(df),
    dtype=object
)

fold_results = []

for fold, (
    train_idx,
    test_idx
) in enumerate(
    logo.split(
        X,
        y,
        groups
    ),
    start=1
):

    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    test_run = groups.iloc[
        test_idx
    ].iloc[0]

    model = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS_EVAL,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    oof_predictions[
        test_idx
    ] = predictions

    fold_accuracy = accuracy_score(
        y_test,
        predictions
    )

    fold_results.append(
        {
            "fold": fold,
            "test_run": test_run,
            "test_windows": len(test_idx),
            "accuracy": fold_accuracy,
            "true_labels": y_test.iloc[0]
        }
    )

    print(
        f"Fold {fold}: "
        f"test_run={test_run}, "
        f"windows={len(test_idx)}, "
        f"accuracy={fold_accuracy:.4f}"
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_accuracy = accuracy_score(
    y,
    oof_predictions
)

balanced_accuracy = balanced_accuracy_score(
    y,
    oof_predictions
)

print()
print("=" * 70)
print("STEP 49 RESULTS")
print("=" * 70)

print()
print(
    f"Overall accuracy:          "
    f"{overall_accuracy:.4f}"
)

print(
    f"Overall balanced accuracy: "
    f"{balanced_accuracy:.4f}"
)

print()
print("Classification report:")

print(
    classification_report(
        y,
        oof_predictions,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y,
    oof_predictions,
    labels=labels
)

print()
print("Confusion matrix:")
print()
print(
    f"Labels: {labels}"
)
print(cm)


# ============================================================
# FOLD RESULTS
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

print()
print("Fold results:")

print(
    fold_results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

oof_output = df[
    [
        "run_id",
        "window_number",
        "label"
    ]
].copy()

oof_output[
    "prediction"
] = oof_predictions

oof_output.to_csv(
    OOF_FILE,
    index=False
)

print()
print(
    "Saved OOF predictions:"
)
print(
    OOF_FILE
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING FINAL PX4-NATIVE MODEL")
print("=" * 70)

final_model = ExtraTreesClassifier(
    n_estimators=N_ESTIMATORS_FINAL,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    max_features="sqrt",
    n_jobs=-1
)

final_model.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL BUNDLE
# ============================================================

model_bundle = {
    "model": final_model,
    "features": remaining_features,
    "medians": medians[remaining_features],
    "label_column": "label",
    "run_column": "run_id",
    "window_column": "window_number",
    "n_features": len(remaining_features),
    "model_name": "PX4-native ExtraTrees FDI",
    "n_estimators": N_ESTIMATORS_FINAL,
    "random_state": RANDOM_STATE,
    "classes": list(final_model.classes_),
    "max_windows_per_run": MAX_WINDOWS_PER_RUN,
    "description": (
        "PX4-native window-level FDI model using "
        "50-sample telemetry windows and statistical "
        "features. Training/evaluation dataset is "
        "balanced at the independent-run window level "
        "to prevent long recordings from dominating."
    )
}

joblib.dump(
    model_bundle,
    MODEL_FILE
)

print()
print(
    "Saved model:"
)
print(
    MODEL_FILE
)

print()
print("=" * 70)
print("STEP 49 COMPLETE")
print("=" * 70)
