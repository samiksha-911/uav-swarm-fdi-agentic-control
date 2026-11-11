#!/usr/bin/env python3

"""
======================================================================
STEP 15 — RUN-LEVEL FEATURE ENGINEERING + CLASSIFICATION BASELINE
======================================================================

Goal:
    Stop treating 10,000 samples from one flight/run as independent
    training examples.

    Instead:
        10,000 samples/run
                |
                v
        statistical run-level features
                |
                v
        1 feature vector/run
                |
                v
        classifier
                |
                v
        failure type

Training:
    56 runs

Testing:
    14 runs

Outputs:
    step15_run_features_train.csv
    step15_run_features_test.csv
    step15_cv_results.csv
    step15_test_predictions.csv
    step15_results.csv
    step15_feature_importance.csv
======================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURATION
# =====================================================================

TRAIN_FILE = "fdi_train.csv"
TEST_FILE = "fdi_test.csv"

RUN_COLUMN = "run"
TARGET_COLUMN = "failure_type"

OUTPUT_TRAIN_FEATURES = "step15_run_features_train.csv"
OUTPUT_TEST_FEATURES = "step15_run_features_test.csv"
OUTPUT_CV = "step15_cv_results.csv"
OUTPUT_PREDICTIONS = "step15_test_predictions.csv"
OUTPUT_RESULTS = "step15_results.csv"
OUTPUT_IMPORTANCE = "step15_feature_importance.csv"


# =====================================================================
# LOAD DATA
# =====================================================================

print("=" * 70)
print("STEP 15 — RUN-LEVEL FEATURE ENGINEERING + CLASSIFICATION")
print("=" * 70)

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train rows: {len(train):,}")
print(f"Test rows : {len(test):,}")

print(f"Train runs: {train[RUN_COLUMN].nunique()}")
print(f"Test runs : {test[RUN_COLUMN].nunique()}")


# =====================================================================
# BASIC VALIDATION
# =====================================================================

required_train = [RUN_COLUMN, TARGET_COLUMN]
required_test = [RUN_COLUMN]

missing_train = [c for c in required_train if c not in train.columns]
missing_test = [c for c in required_test if c not in test.columns]

if missing_train:
    raise ValueError(f"Missing train columns: {missing_train}")

if missing_test:
    raise ValueError(f"Missing test columns: {missing_test}")


print("\n" + "-" * 70)
print("DATASET STRUCTURE")
print("-" * 70)

train_runs = train.groupby(RUN_COLUMN)[TARGET_COLUMN].nunique()

print("\nTraining runs with exactly one class:")
print((train_runs == 1).sum())

if (train_runs != 1).any():
    print("\nWARNING: Some training runs have multiple/zero classes:")
    print(train_runs[train_runs != 1])

test_true_available = TARGET_COLUMN in test.columns

if test_true_available:
    test_runs = test.groupby(RUN_COLUMN)[TARGET_COLUMN].nunique()

    print("\nTest runs with exactly one class:")
    print((test_runs == 1).sum())

    if (test_runs != 1).any():
        print("\nWARNING: Some test runs have multiple/zero classes:")
        print(test_runs[test_runs != 1])


# =====================================================================
# IDENTIFY NUMERIC TELEMETRY FEATURES
# =====================================================================

print("\n" + "-" * 70)
print("FEATURE SELECTION")
print("-" * 70)

numeric_columns = train.select_dtypes(include=[np.number]).columns.tolist()

# Metadata / leakage candidates.
metadata_columns = {
    "metadata",
    "sample_index",
    "relative_to_failure",
    "failure",
    "label",
}

# Remove target if numeric for any reason.
metadata_columns.add(TARGET_COLUMN)

telemetry_columns = [
    c for c in numeric_columns
    if c not in metadata_columns
]

# Also remove obvious IDs.
id_like = []

for c in telemetry_columns:
    lower = c.lower()

    if lower in {
        "id",
        "index",
        "row_id",
        "run_id",
    }:
        id_like.append(c)

telemetry_columns = [
    c for c in telemetry_columns
    if c not in id_like
]

print(f"Numeric columns found: {len(numeric_columns)}")
print(f"Telemetry columns used: {len(telemetry_columns)}")

print("\nFirst 30 telemetry features:")
print(telemetry_columns[:30])


if len(telemetry_columns) == 0:
    raise ValueError("No numeric telemetry features available.")


# =====================================================================
# RUN-LEVEL FEATURE EXTRACTION
# =====================================================================

def safe_quantile(x, q):
    """
    Robust quantile calculation.
    """
    try:
        return float(np.nanquantile(x, q))
    except Exception:
        return np.nan


def calculate_run_features(df, include_target=True):
    """
    Convert all samples belonging to each run into one row.

    Features per telemetry signal:

        mean
        std
        min
        max
        median
        q05
        q25
        q75
        q95
        range
        first
        last
        absolute change
        mean absolute first difference
        std first difference

    This gives the classifier both:
        - level information
        - variability information
        - temporal information
    """

    output_rows = []

    grouped = df.groupby(RUN_COLUMN, sort=False)

    total_runs = len(grouped)

    print(f"\nExtracting features from {total_runs} runs...")

    for run_number, group in grouped:

        row = {
            RUN_COLUMN: run_number,
        }

        # -------------------------------------------------------------
        # Target
        # -------------------------------------------------------------

        if include_target and TARGET_COLUMN in group.columns:

            labels = group[TARGET_COLUMN].dropna()

            if len(labels) == 0:
                row[TARGET_COLUMN] = np.nan

            else:
                counts = labels.value_counts()

                # We expect exactly one class/run.
                row[TARGET_COLUMN] = counts.index[0]

        # -------------------------------------------------------------
        # Telemetry features
        # -------------------------------------------------------------

        for feature in telemetry_columns:

            x = pd.to_numeric(
                group[feature],
                errors="coerce"
            ).to_numpy(dtype=float)

            valid = x[np.isfinite(x)]

            if len(valid) == 0:

                row[f"{feature}__mean"] = np.nan
                row[f"{feature}__std"] = np.nan
                row[f"{feature}__min"] = np.nan
                row[f"{feature}__max"] = np.nan
                row[f"{feature}__median"] = np.nan
                row[f"{feature}__q05"] = np.nan
                row[f"{feature}__q25"] = np.nan
                row[f"{feature}__q75"] = np.nan
                row[f"{feature}__q95"] = np.nan
                row[f"{feature}__range"] = np.nan
                row[f"{feature}__first"] = np.nan
                row[f"{feature}__last"] = np.nan
                row[f"{feature}__abs_change"] = np.nan
                row[f"{feature}__diff_mean_abs"] = np.nan
                row[f"{feature}__diff_std"] = np.nan

                continue

            row[f"{feature}__mean"] = float(np.mean(valid))

            if len(valid) > 1:
                row[f"{feature}__std"] = float(np.std(valid))
            else:
                row[f"{feature}__std"] = 0.0

            row[f"{feature}__min"] = float(np.min(valid))
            row[f"{feature}__max"] = float(np.max(valid))
            row[f"{feature}__median"] = float(np.median(valid))

            row[f"{feature}__q05"] = safe_quantile(valid, 0.05)
            row[f"{feature}__q25"] = safe_quantile(valid, 0.25)
            row[f"{feature}__q75"] = safe_quantile(valid, 0.75)
            row[f"{feature}__q95"] = safe_quantile(valid, 0.95)

            row[f"{feature}__range"] = (
                float(np.max(valid) - np.min(valid))
            )

            # ---------------------------------------------------------
            # Temporal features
            # ---------------------------------------------------------

            row[f"{feature}__first"] = float(valid[0])
            row[f"{feature}__last"] = float(valid[-1])

            row[f"{feature}__abs_change"] = float(
                abs(valid[-1] - valid[0])
            )

            if len(valid) > 1:

                diffs = np.diff(valid)

                row[f"{feature}__diff_mean_abs"] = float(
                    np.mean(np.abs(diffs))
                )

                row[f"{feature}__diff_std"] = float(
                    np.std(diffs)
                )

            else:

                row[f"{feature}__diff_mean_abs"] = 0.0
                row[f"{feature}__diff_std"] = 0.0

        output_rows.append(row)

    result = pd.DataFrame(output_rows)

    return result


# =====================================================================
# EXTRACT RUN FEATURES
# =====================================================================

print("\n" + "-" * 70)
print("RUN-LEVEL FEATURE ENGINEERING")
print("-" * 70)

train_features = calculate_run_features(
    train,
    include_target=True
)

test_features = calculate_run_features(
    test,
    include_target=test_true_available
)

print("\nFeature matrices:")
print(f"Train: {train_features.shape}")
print(f"Test : {test_features.shape}")


# =====================================================================
# SAVE RAW RUN FEATURES
# =====================================================================

train_features.to_csv(
    OUTPUT_TRAIN_FEATURES,
    index=False
)

test_features.to_csv(
    OUTPUT_TEST_FEATURES,
    index=False
)

print("\nSaved:")
print(f"  {OUTPUT_TRAIN_FEATURES}")
print(f"  {OUTPUT_TEST_FEATURES}")


# =====================================================================
# PREPARE X / y
# =====================================================================

feature_columns = [
    c for c in train_features.columns
    if c not in {
        RUN_COLUMN,
        TARGET_COLUMN,
    }
]

X = train_features[feature_columns].copy()

y = train_features[TARGET_COLUMN].copy()

X_test = test_features[feature_columns].copy()

print("\n" + "-" * 70)
print("MODEL MATRIX")
print("-" * 70)

print(f"Training runs: {len(X)}")
print(f"Test runs: {len(X_test)}")
print(f"Features before filtering: {X.shape[1]}")


# =====================================================================
# REMOVE CONSTANT FEATURES
# =====================================================================

print("\nRemoving constant features...")

nunique = X.nunique(dropna=False)

constant_features = nunique[
    nunique <= 1
].index.tolist()

if constant_features:
    X = X.drop(columns=constant_features)
    X_test = X_test.drop(columns=constant_features)

print(f"Constant features removed: {len(constant_features)}")
print(f"Features remaining: {X.shape[1]}")


# =====================================================================
# REMOVE EXTREME MISSINGNESS FEATURES
# =====================================================================

missing_fraction = X.isna().mean()

high_missing = missing_fraction[
    missing_fraction > 0.80
].index.tolist()

if high_missing:

    print(
        f"Removing {len(high_missing)} features with >80% missingness."
    )

    X = X.drop(columns=high_missing)
    X_test = X_test.drop(columns=high_missing)

print(f"Final feature count: {X.shape[1]}")


# =====================================================================
# CLASS DISTRIBUTION
# =====================================================================

print("\n" + "-" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("-" * 70)

print(y.value_counts())


# =====================================================================
# MODELS
# =====================================================================

models = {

    "LogisticRegression": Pipeline([
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
                class_weight="balanced"
            )
        ),
    ]),

    "RandomForest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),
    ]),

    "ExtraTrees": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),
    ]),

    "HistGradientBoosting": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42
            )
        ),
    ]),
}


# =====================================================================
# CROSS VALIDATION
# =====================================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)

# IMPORTANT:
# We are already operating at run level.
# Therefore each row is an independent run.
#
# StratifiedKFold is appropriate here because there are only 56 runs.

min_class_count = y.value_counts().min()

n_splits = min(
    5,
    int(min_class_count)
)

if n_splits < 2:
    raise ValueError(
        "Not enough runs per class for cross-validation."
    )

print(f"\nUsing {n_splits}-fold stratified CV.")


cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)


cv_results = []

for name, model in models.items():

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    accuracy_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    balanced_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=1
    )

    print(
        "Accuracy:",
        np.round(accuracy_scores, 4)
    )

    print(
        f"Mean accuracy: "
        f"{accuracy_scores.mean():.4f}"
    )

    print(
        f"Std accuracy: "
        f"{accuracy_scores.std():.4f}"
    )

    print(
        f"Mean balanced accuracy: "
        f"{balanced_scores.mean():.4f}"
    )

    cv_results.append({
        "model": name,
        "accuracy_mean": accuracy_scores.mean(),
        "accuracy_std": accuracy_scores.std(),
        "balanced_accuracy_mean": balanced_scores.mean(),
    })


cv_results_df = pd.DataFrame(cv_results)

cv_results_df = cv_results_df.sort_values(
    "accuracy_mean",
    ascending=False
)

cv_results_df.to_csv(
    OUTPUT_CV,
    index=False
)

print("\n" + "=" * 70)
print("CV MODEL COMPARISON")
print("=" * 70)

print(
    cv_results_df.to_string(index=False)
)


# =====================================================================
# SELECT BEST MODEL
# =====================================================================

best_model_name = cv_results_df.iloc[0]["model"]

best_model = models[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(best_model_name)


# =====================================================================
# TRAIN BEST MODEL ON ALL 56 RUNS
# =====================================================================

print("\nTraining best model on all training runs...")

best_model.fit(
    X,
    y
)

print("Training complete.")


# =====================================================================
# PREDICT TEST RUNS
# =====================================================================

test_predictions = best_model.predict(X_test)

test_predictions = pd.Series(
    test_predictions,
    name="predicted_failure_type"
)


# =====================================================================
# TEST PREDICTION OUTPUT
# =====================================================================

prediction_output = pd.DataFrame({
    RUN_COLUMN: test_features[RUN_COLUMN],
    "predicted_failure_type": test_predictions,
})

if test_true_available:
    prediction_output[TARGET_COLUMN] = (
        test_features[TARGET_COLUMN]
    )

    prediction_output["correct"] = (
        prediction_output[TARGET_COLUMN]
        ==
        prediction_output["predicted_failure_type"]
    )


prediction_output.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# =====================================================================
# TEST RESULTS
# =====================================================================

print("\n" + "=" * 70)
print("TEST RUN PREDICTIONS")
print("=" * 70)

print(
    prediction_output.to_string(index=False)
)

print("\nPrediction distribution:")

print(
    prediction_output[
        "predicted_failure_type"
    ].value_counts()
)


# =====================================================================
# EVALUATE TEST IF TRUE LABELS EXIST
# =====================================================================

if test_true_available:

    y_test = prediction_output[TARGET_COLUMN]
    y_pred = prediction_output["predicted_failure_type"]

    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    test_balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("FINAL TEST PERFORMANCE")
    print("=" * 70)

    print(
        f"\nRun-level accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Run-level balanced accuracy: "
        f"{test_balanced_accuracy:.4f}"
    )

    print(
        f"Correct runs: "
        f"{int((y_test == y_pred).sum())}/{len(y_test)}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    labels = sorted(
        pd.unique(
            pd.concat(
                [
                    y_test,
                    y_pred
                ]
            )
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print("\nConfusion matrix:")

    print(cm_df)


    # ---------------------------------------------------------------
    # PER-CLASS RESULTS
    # ---------------------------------------------------------------

    results_rows = []

    for cls in sorted(y_test.unique()):

        mask = y_test == cls

        total = int(mask.sum())
        correct = int(
            (y_pred[mask] == cls).sum()
        )

        results_rows.append({
            "failure_type": cls,
            "total_test_runs": total,
            "correct": correct,
            "accuracy": (
                correct / total
                if total > 0
                else np.nan
            )
        })

    results_df = pd.DataFrame(
        results_rows
    )

    print("\nPer-class test performance:")
    print(
        results_df.to_string(index=False)
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False
    )


else:

    print(
        "\nNo true failure_type in test file."
    )

    prediction_output[
        [RUN_COLUMN, "predicted_failure_type"]
    ].to_csv(
        OUTPUT_RESULTS,
        index=False
    )


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance_df = None

try:

    model_step = best_model.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):

        importances = model_step.feature_importances_

        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        })

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        )

    elif hasattr(model_step, "coef_"):

        coefficients = np.mean(
            np.abs(model_step.coef_),
            axis=0
        )

        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": coefficients
        })

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        )

except Exception as e:

    print(
        f"Could not calculate feature importance: {e}"
    )


if importance_df is not None:

    importance_df.to_csv(
        OUTPUT_IMPORTANCE,
        index=False
    )

    print("\nTop 40 features:")

    print(
        importance_df.head(40).to_string(
            index=False
        )
    )

else:

    print(
        "\nFeature importance unavailable for this model."
    )


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("STEP 15 COMPLETE")
print("=" * 70)

print(f"\nTraining runs : {len(train_features)}")
print(f"Test runs     : {len(test_features)}")
print(f"Features used : {X.shape[1]}")
print(f"Best model    : {best_model_name}")

if len(cv_results_df) > 0:

    best_cv_accuracy = cv_results_df.iloc[0][
        "accuracy_mean"
    ]

    print(
        f"CV accuracy   : "
        f"{best_cv_accuracy:.4f}"
    )

if test_true_available:

    print(
        f"Test accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Correct runs  : "
        f"{int((y_test == y_pred).sum())}/{len(y_test)}"
    )

print("\nSaved outputs:")

print(f"  {OUTPUT_TRAIN_FEATURES}")
print(f"  {OUTPUT_TEST_FEATURES}")
print(f"  {OUTPUT_CV}")
print(f"  {OUTPUT_PREDICTIONS}")
print(f"  {OUTPUT_RESULTS}")
print(f"  {OUTPUT_IMPORTANCE}")

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    """
This experiment changes the representation from:

    560,000 individual samples
        ->
    56 independent training runs

and from:

    140,000 test samples
        ->
    14 independent test runs

The result to focus on is:

    CV accuracy
    Test run-level accuracy
    Per-class accuracy
    Confusion matrix
    Top feature importance

Do NOT modify the model based on one test prediction.
First compare the CV result against the previous 28.57% run-level result.
"""
)

print("=" * 70)
