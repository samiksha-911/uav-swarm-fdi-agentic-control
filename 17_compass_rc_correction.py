#!/usr/bin/env python3

"""
STEP 17 — TARGETED COMPASS vs RC CORRECTION

Uses:
    fdi_train.csv
    fdi_test.csv
    step15_run_features_train.csv
    step15_run_features_test.csv
    step15_test_predictions.csv

The Compass-vs-RC threshold is selected ONLY from training-run
out-of-fold predictions.

Test labels are NOT used for threshold selection.
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIG
# ======================================================================

TRAIN_FILE = "fdi_train.csv"
TEST_FILE = "fdi_test.csv"

TRAIN_FEATURE_FILE = "step15_run_features_train.csv"
TEST_FEATURE_FILE = "step15_run_features_test.csv"
STEP15_PRED_FILE = "step15_test_predictions.csv"

OUTPUT_OOF = "step17_compass_rc_oof.csv"
OUTPUT_THRESHOLDS = "step17_compass_rc_thresholds.csv"
OUTPUT_RESULTS = "step17_compass_rc_corrected_results.csv"
OUTPUT_SUMMARY = "step17_compass_rc_summary.csv"

RUN_COL = "run"
TARGET_COL = "failure_type"

COMPASS = "Compass Failure"
RC = "RC Failure"

RANDOM_STATE = 42


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("STEP 17 — TARGETED COMPASS vs RC CORRECTION")
print("=" * 70)


# ======================================================================
# HELPERS
# ======================================================================

def check_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}"
        )


def build_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])


def get_compass_probability(model, X):

    probabilities = model.predict_proba(X)

    final_estimator = model.named_steps["model"]

    classes = final_estimator.classes_

    compass_index = list(classes).index(COMPASS)

    return probabilities[:, compass_index]


# ======================================================================
# 1. LOAD ORIGINAL DATA
# ======================================================================

print("\nLoading files...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
pred = pd.read_csv(STEP15_PRED_FILE)

check_columns(
    train,
    [RUN_COL, TARGET_COL],
    "Training dataset"
)

check_columns(
    test,
    [RUN_COL, TARGET_COL],
    "Test dataset"
)

check_columns(
    pred,
    [RUN_COL, "predicted_failure_type"],
    "Step 15 prediction file"
)

print(f"Training runs : {train[RUN_COL].nunique()}")
print(f"Test runs     : {test[RUN_COL].nunique()}")
print(f"Prediction rows: {len(pred)}")

print(
    f"\nTraining target column : {TARGET_COL}"
)

print(
    f"Test true column       : {TARGET_COL}"
)

print(
    "Prediction column      : predicted_failure_type"
)


# ======================================================================
# 2. LOAD STEP 15 RUN FEATURES
# ======================================================================

print("\nLoading Step 15 run-level features...")

if not os.path.exists(TRAIN_FEATURE_FILE):
    raise FileNotFoundError(
        TRAIN_FEATURE_FILE
    )

if not os.path.exists(TEST_FEATURE_FILE):
    raise FileNotFoundError(
        TEST_FEATURE_FILE
    )

train_features = pd.read_csv(
    TRAIN_FEATURE_FILE
)

test_features = pd.read_csv(
    TEST_FEATURE_FILE
)

check_columns(
    train_features,
    [RUN_COL],
    "Step 15 training feature file"
)

check_columns(
    test_features,
    [RUN_COL],
    "Step 15 test feature file"
)

print(
    f"Training feature rows: {len(train_features)}"
)

print(
    f"Test feature rows    : {len(test_features)}"
)

print(
    f"Training feature runs: "
    f"{train_features[RUN_COL].nunique()}"
)

print(
    f"Test feature runs    : "
    f"{test_features[RUN_COL].nunique()}"
)


# ======================================================================
# 3. CREATE RUN-LEVEL TRUE LABEL TABLE
# ======================================================================

print("\nCreating run-level labels from original datasets...")

train_labels = (
    train
    .groupby(RUN_COL)[TARGET_COL]
    .agg(
        lambda x: (
            x.dropna().iloc[0]
            if len(x.dropna()) > 0
            else np.nan
        )
    )
    .reset_index()
)

train_labels = train_labels.rename(
    columns={
        TARGET_COL: "true_failure_type"
    }
)

test_labels = (
    test
    .groupby(RUN_COL)[TARGET_COL]
    .agg(
        lambda x: (
            x.dropna().iloc[0]
            if len(x.dropna()) > 0
            else np.nan
        )
    )
    .reset_index()
)

test_labels = test_labels.rename(
    columns={
        TARGET_COL: "true_failure_type"
    }
)

print(
    f"Training run labels: {len(train_labels)}"
)

print(
    f"Test run labels    : {len(test_labels)}"
)


# ======================================================================
# 4. MERGE LABELS ONTO FEATURE MATRICES
# ======================================================================

print("\n" + "=" * 70)
print("1. ALIGNING RUN-LEVEL FEATURES AND LABELS")
print("=" * 70)

# Remove accidental target columns if they already exist in Step 15 files.

for col in [
    TARGET_COL,
    "true_failure_type",
    "predicted_failure_type",
]:

    if col in train_features.columns:
        train_features = train_features.drop(
            columns=[col]
        )

    if col in test_features.columns:
        test_features = test_features.drop(
            columns=[col]
        )


# Merge training labels
train_features = train_features.merge(
    train_labels,
    on=RUN_COL,
    how="inner",
    validate="one_to_one"
)

# Merge test labels
test_features = test_features.merge(
    test_labels,
    on=RUN_COL,
    how="inner",
    validate="one_to_one"
)

print(
    f"Aligned training runs: "
    f"{len(train_features)}"
)

print(
    f"Aligned test runs    : "
    f"{len(test_features)}"
)

if len(train_features) != train_labels[RUN_COL].nunique():
    raise ValueError(
        "Training feature/label run alignment failed."
    )

if len(test_features) != test_labels[RUN_COL].nunique():
    raise ValueError(
        "Test feature/label run alignment failed."
    )


# ======================================================================
# 5. FEATURE MATRIX
# ======================================================================

print("\n" + "=" * 70)
print("2. FEATURE MATRIX")
print("=" * 70)

excluded = {
    RUN_COL,
    "true_failure_type",
    TARGET_COL,
    "prediction",
    "predicted_failure_type",
    "correct",
}

shared_features = [
    c
    for c in train_features.columns
    if c in test_features.columns
    and c not in excluded
    and pd.api.types.is_numeric_dtype(
        train_features[c]
    )
    and pd.api.types.is_numeric_dtype(
        test_features[c]
    )
]

print(
    f"Numeric shared features: "
    f"{len(shared_features)}"
)

if len(shared_features) == 0:
    raise ValueError(
        "No numeric shared features found."
    )


# ======================================================================
# 6. TARGETED COMPASS vs RC DATA
# ======================================================================

print("\n" + "=" * 70)
print("3. BUILDING TARGETED COMPASS vs RC DATA")
print("=" * 70)

target_train = train_features[
    train_features["true_failure_type"].isin(
        [COMPASS, RC]
    )
].copy()

print("\nTarget training distribution:")

print(
    target_train[
        "true_failure_type"
    ].value_counts()
)

n_compass = int(
    (
        target_train["true_failure_type"]
        == COMPASS
    ).sum()
)

n_rc = int(
    (
        target_train["true_failure_type"]
        == RC
    ).sum()
)

print(
    f"\nCompass training runs: {n_compass}"
)

print(
    f"RC training runs     : {n_rc}"
)

if n_compass < 2 or n_rc < 2:
    raise ValueError(
        "Not enough Compass/RC runs for validation."
    )


# ======================================================================
# 7. REMOVE CONSTANT FEATURES
# ======================================================================

print("\nRemoving constant features...")

X_target = target_train[
    shared_features
].copy()

y_target = target_train[
    "true_failure_type"
].copy()

constant_features = []

for col in X_target.columns:

    values = X_target[col].dropna()

    if len(values) == 0:
        constant_features.append(col)

    elif values.nunique() <= 1:
        constant_features.append(col)

usable_features = [
    c
    for c in shared_features
    if c not in constant_features
]

print(
    f"Original features : "
    f"{len(shared_features)}"
)

print(
    f"Removed constants : "
    f"{len(constant_features)}"
)

print(
    f"Usable features   : "
    f"{len(usable_features)}"
)

X_target = target_train[
    usable_features
].copy()


# ======================================================================
# 8. TRAINING-ONLY OOF
# ======================================================================

print("\n" + "=" * 70)
print("4. TRAINING-ONLY OOF VALIDATION")
print("=" * 70)

print(
    "\nImportant:"
)

print(
    "The test set is NOT used to choose the threshold."
)

min_class_count = int(
    y_target.value_counts().min()
)

n_splits = min(
    4,
    min_class_count
)

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_model = build_model()

oof_prediction = cross_val_predict(
    oof_model,
    X_target,
    y_target,
    cv=cv,
    method="predict",
    n_jobs=-1
)

oof_probability_matrix = cross_val_predict(
    oof_model,
    X_target,
    y_target,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

# Fit only on training data to establish class ordering.
class_reference_model = build_model()

class_reference_model.fit(
    X_target,
    y_target
)

classes = (
    class_reference_model
    .named_steps["model"]
    .classes_
)

print("\nModel classes:")
print(classes)

compass_index = list(classes).index(
    COMPASS
)

oof_compass_probability = (
    oof_probability_matrix[:, compass_index]
)

oof_accuracy = accuracy_score(
    y_target,
    oof_prediction
)

oof_balanced = balanced_accuracy_score(
    y_target,
    oof_prediction
)

print(
    "\nTargeted OOF accuracy:"
)

print(
    f"{oof_accuracy:.4f}"
)

print(
    "\nTargeted OOF balanced accuracy:"
)

print(
    f"{oof_balanced:.4f}"
)

print(
    "\nOOF classification report:"
)

print(
    classification_report(
        y_target,
        oof_prediction,
        labels=[
            COMPASS,
            RC
        ],
        zero_division=0
    )
)


# ======================================================================
# 9. SAVE OOF
# ======================================================================

oof_df = pd.DataFrame({
    RUN_COL:
        target_train[RUN_COL].values,

    "true_failure_type":
        y_target.values,

    "oof_prediction":
        oof_prediction,

    "compass_probability":
        oof_compass_probability,
})

oof_df.to_csv(
    OUTPUT_OOF,
    index=False
)

print(
    f"Saved: {OUTPUT_OOF}"
)


# ======================================================================
# 10. THRESHOLD SEARCH
# ======================================================================

print("\n" + "=" * 70)
print("5. TRAINING-ONLY THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = np.round(
    np.arange(
        0.05,
        1.00,
        0.05
    ),
    2
)

threshold_results = []

for threshold in thresholds:

    threshold_prediction = np.where(
        oof_compass_probability >= threshold,
        COMPASS,
        RC
    )

    accuracy = accuracy_score(
        y_target,
        threshold_prediction
    )

    balanced_accuracy = balanced_accuracy_score(
        y_target,
        threshold_prediction
    )

    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
    })

threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(
        index=False,
        formatters={
            "accuracy":
                "{:.4f}".format,

            "balanced_accuracy":
                "{:.4f}".format,
        }
    )
)

threshold_df.to_csv(
    OUTPUT_THRESHOLDS,
    index=False
)

print(
    f"\nSaved: {OUTPUT_THRESHOLDS}"
)


# ======================================================================
# 11. BEST TRAINING THRESHOLD
# ======================================================================

best_row = (
    threshold_df
    .sort_values(
        by=[
            "balanced_accuracy",
            "accuracy"
        ],
        ascending=False
    )
    .iloc[0]
)

best_threshold = float(
    best_row["threshold"]
)

best_threshold_accuracy = float(
    best_row["accuracy"]
)

best_threshold_balanced = float(
    best_row["balanced_accuracy"]
)

print(
    "\nBest training-only threshold:"
)

print(
    f"Threshold          : "
    f"{best_threshold:.2f}"
)

print(
    f"OOF accuracy       : "
    f"{best_threshold_accuracy:.4f}"
)

print(
    f"OOF balanced acc   : "
    f"{best_threshold_balanced:.4f}"
)


# ======================================================================
# 12. TRAIN FINAL TARGETED MODEL
# ======================================================================

print("\n" + "=" * 70)
print("6. TRAINING FINAL TARGETED MODEL")
print("=" * 70)

final_model = build_model()

final_model.fit(
    X_target,
    y_target
)

print(
    "Targeted model training complete."
)


# ======================================================================
# 13. LOAD STEP 15 BASELINE
# ======================================================================

print("\n" + "=" * 70)
print("7. STEP 15 BASELINE")
print("=" * 70)

baseline = pred[
    [
        RUN_COL,
        "predicted_failure_type"
    ]
].drop_duplicates(
    RUN_COL
)

baseline = baseline.rename(
    columns={
        "predicted_failure_type":
            "baseline_prediction"
    }
)

aligned_test = test_labels.merge(
    baseline,
    on=RUN_COL,
    how="inner",
    validate="one_to_one"
)

if len(aligned_test) != len(test_labels):

    missing = set(
        test_labels[RUN_COL]
    ) - set(
        baseline[RUN_COL]
    )

    raise ValueError(
        f"Missing Step 15 predictions for: "
        f"{sorted(missing)}"
    )

baseline_accuracy = accuracy_score(
    aligned_test["true_failure_type"],
    aligned_test["baseline_prediction"]
)

baseline_balanced = balanced_accuracy_score(
    aligned_test["true_failure_type"],
    aligned_test["baseline_prediction"]
)

print(
    f"Step 15 baseline accuracy: "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Step 15 baseline balanced accuracy: "
    f"{baseline_balanced:.4f}"
)


# ======================================================================
# 14. TARGET ONLY COMPASS/RC BASELINE PREDICTIONS
# ======================================================================

test_target = aligned_test[
    aligned_test["baseline_prediction"].isin(
        [COMPASS, RC]
    )
].copy()

print(
    "\nTest runs requiring Compass/RC consideration: "
    f"{len(test_target)}"
)

if len(test_target) > 0:

    candidate_features = (
        test_features[
            test_features[RUN_COL].isin(
                set(test_target[RUN_COL])
            )
        ]
        .copy()
    )

    candidate_features = candidate_features[
        [RUN_COL] + usable_features
    ]

    test_target = test_target.merge(
        candidate_features,
        on=RUN_COL,
        how="left",
        validate="one_to_one"
    )

    X_test_target = test_target[
        usable_features
    ]

    targeted_prediction = final_model.predict(
        X_test_target
    )

    targeted_probability = (
        get_compass_probability(
            final_model,
            X_test_target
        )
    )

    threshold_prediction = np.where(
        targeted_probability >= best_threshold,
        COMPASS,
        RC
    )

    test_target[
        "targeted_prediction"
    ] = targeted_prediction

    test_target[
        "compass_probability"
    ] = targeted_probability

    test_target[
        "threshold_prediction"
    ] = threshold_prediction

    print("\n" + "=" * 70)
    print("8. TARGETED TEST PREDICTIONS")
    print("=" * 70)

    print(
        test_target[
            [
                RUN_COL,
                "true_failure_type",
                "baseline_prediction",
                "targeted_prediction",
                "threshold_prediction",
                "compass_probability",
            ]
        ].to_string(index=False)
    )


# ======================================================================
# 15. APPLY CORRECTION
# ======================================================================

print("\n" + "=" * 70)
print("9. APPLYING TRAINING-ONLY CORRECTION")
print("=" * 70)

corrected = aligned_test.copy()

# Start with Step 15 baseline.
corrected[
    "corrected_prediction"
] = corrected[
    "baseline_prediction"
]

if len(test_target) > 0:

    correction_map = test_target[
        [
            RUN_COL,
            "threshold_prediction",
            "compass_probability",
        ]
    ].copy()

    correction_map = correction_map.rename(
        columns={
            "threshold_prediction":
                "correction_prediction"
        }
    )

    corrected = corrected.merge(
        correction_map,
        on=RUN_COL,
        how="left",
        validate="one_to_one"
    )

    has_correction = (
        corrected[
            "correction_prediction"
        ].notna()
    )

    corrected.loc[
        has_correction,
        "corrected_prediction"
    ] = corrected.loc[
        has_correction,
        "correction_prediction"
    ]

    corrected = corrected.drop(
        columns=[
            "correction_prediction"
        ]
    )

else:

    corrected[
        "compass_probability"
    ] = np.nan


# ======================================================================
# 16. PERFORMANCE
# ======================================================================

corrected[
    "baseline_correct"
] = (
    corrected["baseline_prediction"]
    == corrected["true_failure_type"]
)

corrected[
    "corrected_correct"
] = (
    corrected["corrected_prediction"]
    == corrected["true_failure_type"]
)

corrected_accuracy = accuracy_score(
    corrected["true_failure_type"],
    corrected["corrected_prediction"]
)

corrected_balanced = balanced_accuracy_score(
    corrected["true_failure_type"],
    corrected["corrected_prediction"]
)

print("\n" + "=" * 70)
print("10. BASELINE vs CORRECTED PERFORMANCE")
print("=" * 70)

print(
    "\nBaseline:"
)

print(
    f"Accuracy          : "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Balanced accuracy : "
    f"{baseline_balanced:.4f}"
)

print(
    "\nCorrected:"
)

print(
    f"Accuracy          : "
    f"{corrected_accuracy:.4f}"
)

print(
    f"Balanced accuracy : "
    f"{corrected_balanced:.4f}"
)

print(
    "\nAccuracy change   : "
    f"{corrected_accuracy - baseline_accuracy:+.4f}"
)

print(
    "Balanced change   : "
    f"{corrected_balanced - baseline_balanced:+.4f}"
)


# ======================================================================
# 17. RUN-BY-RUN
# ======================================================================

print("\n" + "=" * 70)
print("11. RUN-BY-RUN RESULTS")
print("=" * 70)

print(
    corrected[
        [
            RUN_COL,
            "true_failure_type",
            "baseline_prediction",
            "corrected_prediction",
            "baseline_correct",
            "corrected_correct",
        ]
    ].to_string(index=False)
)


# ======================================================================
# 18. CHANGED PREDICTIONS
# ======================================================================

changed = corrected[
    corrected["baseline_prediction"]
    != corrected["corrected_prediction"]
].copy()

print("\n" + "=" * 70)
print("12. CHANGED PREDICTIONS")
print("=" * 70)

print(
    f"Changed runs: {len(changed)}"
)

if len(changed) > 0:

    print(
        changed[
            [
                RUN_COL,
                "true_failure_type",
                "baseline_prediction",
                "corrected_prediction",
                "compass_probability",
            ]
        ].to_string(index=False)
    )

else:

    print(
        "No predictions changed."
    )


# ======================================================================
# 19. CORRECTION EFFECT
# ======================================================================

improved = changed[
    changed["corrected_correct"]
    & ~changed["baseline_correct"]
]

worsened = changed[
    ~changed["corrected_correct"]
    & changed["baseline_correct"]
]

print("\n" + "=" * 70)
print("13. CORRECTION EFFECT")
print("=" * 70)

print(
    f"Improved runs : {len(improved)}"
)

print(
    f"Worsened runs : {len(worsened)}"
)


# ======================================================================
# 20. SAVE RESULTS
# ======================================================================

corrected.to_csv(
    OUTPUT_RESULTS,
    index=False
)

summary = pd.DataFrame([
    {
        "baseline_accuracy":
            baseline_accuracy,

        "baseline_balanced_accuracy":
            baseline_balanced,

        "corrected_accuracy":
            corrected_accuracy,

        "corrected_balanced_accuracy":
            corrected_balanced,

        "accuracy_change":
            corrected_accuracy
            - baseline_accuracy,

        "balanced_accuracy_change":
            corrected_balanced
            - baseline_balanced,

        "training_only_threshold":
            best_threshold,

        "targeted_oof_accuracy":
            oof_accuracy,

        "targeted_oof_balanced_accuracy":
            oof_balanced,

        "threshold_oof_accuracy":
            best_threshold_accuracy,

        "threshold_oof_balanced_accuracy":
            best_threshold_balanced,

        "test_runs":
            len(corrected),

        "changed_runs":
            len(changed),

        "improved_runs":
            len(improved),

        "worsened_runs":
            len(worsened),
    }
])

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ======================================================================
# FINAL
# ======================================================================

print("\n" + "=" * 70)
print("14. SAVED OUTPUTS")
print("=" * 70)

print(
    f"  {OUTPUT_OOF}"
)

print(
    f"  {OUTPUT_THRESHOLDS}"
)

print(
    f"  {OUTPUT_RESULTS}"
)

print(
    f"  {OUTPUT_SUMMARY}"
)

print("\n" + "=" * 70)
print("STEP 17 COMPLETE")
print("=" * 70)

print(
    f"\nBaseline accuracy   : "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Corrected accuracy  : "
    f"{corrected_accuracy:.4f}"
)

print(
    f"Accuracy change     : "
    f"{corrected_accuracy - baseline_accuracy:+.4f}"
)

print(
    f"\nBaseline balanced   : "
    f"{baseline_balanced:.4f}"
)

print(
    f"Corrected balanced  : "
    f"{corrected_balanced:.4f}"
)

print(
    f"Balanced change     : "
    f"{corrected_balanced - baseline_balanced:+.4f}"
)

print(
    f"\nChanged runs        : "
    f"{len(changed)}"
)

print(
    f"Improved runs       : "
    f"{len(improved)}"
)

print(
    f"Worsened runs       : "
    f"{len(worsened)}"
)

print("=" * 70)
