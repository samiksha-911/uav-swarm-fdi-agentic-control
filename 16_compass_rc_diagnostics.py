#!/usr/bin/env python3

"""
======================================================================
STEP 16 — TARGETED COMPASS vs RC DIAGNOSTICS
======================================================================

Purpose:
    Investigate why the Step 15 run-level ExtraTrees model predicts
    two Compass Failure test runs as RC Failure.

This step does NOT modify the final model.

It investigates:
    1. Training Compass vs RC separation
    2. Feature-level differences
    3. Test Compass similarity to RC / Compass
    4. Targeted Compass-vs-RC classification
    5. Feature importance
    6. Whether a small number of features dominate the confusion

Expected input files:
    step15_run_features_train.csv
    step15_run_features_test.csv
    step15_test_predictions.csv

Expected output files:
    step16_compass_rc_feature_stats.csv
    step16_compass_rc_test_similarity.csv
    step16_compass_rc_cv.csv
    step16_compass_rc_feature_importance.csv
    step16_compass_rc_summary.csv
======================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

TRAIN_FILE = "step15_run_features_train.csv"
TEST_FILE = "step15_run_features_test.csv"
PRED_FILE = "step15_test_predictions.csv"

TOP_N = 50

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_numeric(df):
    """
    Convert feature columns to numeric where possible.
    """
    return df.apply(pd.to_numeric, errors="coerce")


def clean_feature_name(name):
    return str(name).strip()


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print_header("STEP 16 — TARGETED COMPASS vs RC DIAGNOSTICS")

print("\nLoading files...")

for f in [TRAIN_FILE, TEST_FILE, PRED_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"Required file not found: {f}\n"
            f"Make sure Step 15 has completed successfully."
        )

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
pred = pd.read_csv(PRED_FILE)

print(f"Training runs: {len(train)}")
print(f"Test runs    : {len(test)}")
print(f"Prediction rows: {len(pred)}")

print("\nTraining classes:")
print(train["failure_type"].value_counts())


# ---------------------------------------------------------------------
# IDENTIFY COLUMNS
# ---------------------------------------------------------------------

required_train = ["run", "failure_type"]
required_test = ["run", "failure_type"]

for c in required_train:
    if c not in train.columns:
        raise ValueError(f"Training file missing required column: {c}")

for c in required_test:
    if c not in test.columns:
        raise ValueError(f"Test file missing required column: {c}")

# Metadata columns that should NOT be treated as telemetry features
metadata_cols = {
    "run",
    "failure_type",
    "predicted_failure_type",
    "prediction",
    "correct",
}

feature_cols = [
    c for c in train.columns
    if c not in metadata_cols
]

# Keep only columns shared between train/test
feature_cols = [
    c for c in feature_cols
    if c in test.columns
]

if len(feature_cols) == 0:
    raise ValueError("No shared feature columns found.")

print_header("1. FEATURE MATRIX")

print(f"Shared feature columns: {len(feature_cols)}")

# ---------------------------------------------------------------------
# EXTRACT COMPASS / RC
# ---------------------------------------------------------------------

compass_train = train[
    train["failure_type"] == "Compass Failure"
].copy()

rc_train = train[
    train["failure_type"] == "RC Failure"
].copy()

compass_test = test[
    test["failure_type"] == "Compass Failure"
].copy()

print("\nTraining Compass runs:")
print(len(compass_train))

print("\nTraining RC runs:")
print(len(rc_train))

print("\nTest Compass runs:")
print(len(compass_test))

if len(compass_train) == 0:
    raise ValueError("No Compass Failure training runs found.")

if len(rc_train) == 0:
    raise ValueError("No RC Failure training runs found.")

if len(compass_test) == 0:
    print("\nWARNING: No Compass Failure test runs found.")


# ---------------------------------------------------------------------
# 2. COMPASS vs RC FEATURE STATISTICS
# ---------------------------------------------------------------------

print_header("2. COMPASS vs RC FEATURE DIFFERENCES")

X_compass = safe_numeric(compass_train[feature_cols])
X_rc = safe_numeric(rc_train[feature_cols])

compass_mean = X_compass.mean()
rc_mean = X_rc.mean()

compass_std = X_compass.std().replace(0, np.nan)
rc_std = X_rc.std().replace(0, np.nan)

# Pooled standard deviation
n_c = len(X_compass)
n_r = len(X_rc)

pooled_std = np.sqrt(
    (
        (n_c - 1) * (compass_std ** 2)
        + (n_r - 1) * (rc_std ** 2)
    )
    / max(n_c + n_r - 2, 1)
)

effect_size = (
    (compass_mean - rc_mean)
    / pooled_std.replace(0, np.nan)
)

effect_abs = effect_size.abs()

feature_stats = pd.DataFrame({
    "feature": feature_cols,
    "compass_mean": compass_mean.values,
    "rc_mean": rc_mean.values,
    "difference": (compass_mean - rc_mean).values,
    "abs_difference": (compass_mean - rc_mean).abs().values,
    "compass_std": compass_std.values,
    "rc_std": rc_std.values,
    "effect_size": effect_size.values,
    "abs_effect_size": effect_abs.values,
})

feature_stats = feature_stats.sort_values(
    "abs_effect_size",
    ascending=False
).reset_index(drop=True)

print("\nTop Compass-vs-RC separating features:")

print(
    feature_stats[
        [
            "feature",
            "compass_mean",
            "rc_mean",
            "effect_size",
        ]
    ].head(TOP_N).to_string(index=False)
)

feature_stats.to_csv(
    "step16_compass_rc_feature_stats.csv",
    index=False
)

print("\nSaved:")
print("  step16_compass_rc_feature_stats.csv")


# ---------------------------------------------------------------------
# 3. TEST COMPASS RUN SIMILARITY
# ---------------------------------------------------------------------

print_header("3. TEST COMPASS RUN SIMILARITY")

if len(compass_test) > 0:

    X_test_compass = safe_numeric(
        compass_test[feature_cols]
    )

    # Use Compass/RC training means and scales.
    # Standardize each feature according to the training
    # Compass + RC population.

    combined = pd.concat(
        [X_compass, X_rc],
        axis=0,
        ignore_index=True
    )

    feature_center = combined.mean()
    feature_scale = combined.std().replace(0, np.nan)

    C = (
        X_compass
        .sub(feature_center, axis=1)
        .div(feature_scale, axis=1)
    )

    R = (
        X_rc
        .sub(feature_center, axis=1)
        .div(feature_scale, axis=1)
    )

    T = (
        X_test_compass
        .sub(feature_center, axis=1)
        .div(feature_scale, axis=1)
    )

    compass_center = C.mean(axis=0)
    rc_center = R.mean(axis=0)

    rows = []

    for idx in range(len(T)):

        test_row = T.iloc[idx]

        distance_compass = np.sqrt(
            np.nanmean(
                (test_row.values - compass_center.values) ** 2
            )
        )

        distance_rc = np.sqrt(
            np.nanmean(
                (test_row.values - rc_center.values) ** 2
            )
        )

        ratio = (
            distance_compass
            / max(distance_rc, 1e-12)
        )

        rows.append({
            "run": compass_test.iloc[idx]["run"],
            "true_failure_type": compass_test.iloc[idx]["failure_type"],
            "distance_to_compass": distance_compass,
            "distance_to_rc": distance_rc,
            "compass_to_rc_distance_ratio": ratio,
            "closer_to": (
                "Compass Failure"
                if distance_compass < distance_rc
                else "RC Failure"
            ),
        })

    similarity = pd.DataFrame(rows)

    print(
        similarity.to_string(index=False)
    )

    similarity.to_csv(
        "step16_compass_rc_test_similarity.csv",
        index=False
    )

    print("\nSaved:")
    print("  step16_compass_rc_test_similarity.csv")

else:
    similarity = pd.DataFrame()


# ---------------------------------------------------------------------
# 4. TARGETED CLASSIFICATION DATASET
# ---------------------------------------------------------------------

print_header("4. TARGETED COMPASS vs RC CLASSIFICATION")

target_train = train[
    train["failure_type"].isin(
        ["Compass Failure", "RC Failure"]
    )
].copy()

X_target = safe_numeric(
    target_train[feature_cols]
)

y_target = target_train["failure_type"]

print("\nTarget training distribution:")
print(y_target.value_counts())

# Remove features that contain no useful numeric values
valid_features = [
    c for c in X_target.columns
    if X_target[c].notna().sum() > 0
]

X_target = X_target[valid_features]

print(f"\nValid numeric features: {len(valid_features)}")


# ---------------------------------------------------------------------
# 5. TARGETED CROSS-VALIDATION
# ---------------------------------------------------------------------

print_header("5. COMPASS vs RC CROSS-VALIDATION")

# With only 6 Compass + 10 RC runs, use 3-fold CV.
#
# Important:
# The purpose is diagnostic, not to claim a highly reliable
# generalization estimate.

min_class_count = y_target.value_counts().min()

n_splits = min(3, int(min_class_count))

if n_splits < 2:
    raise ValueError(
        "Not enough Compass/RC runs for cross-validation."
    )

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE
)

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
            "classifier",
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
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
                max_features="sqrt",
            )
        ),
    ]),

    "ExtraTrees": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
                max_features="sqrt",
            )
        ),
    ]),
}

cv_results = []

for name, model in models.items():

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    scores = cross_val_score(
        model,
        X_target,
        y_target,
        cv=cv,
        scoring="accuracy"
    )

    balanced_scores = cross_val_score(
        model,
        X_target,
        y_target,
        cv=cv,
        scoring="balanced_accuracy"
    )

    print(
        "Accuracy:",
        np.round(scores, 4)
    )

    print(
        "Mean accuracy:",
        round(scores.mean(), 4)
    )

    print(
        "Balanced accuracy:",
        round(balanced_scores.mean(), 4)
    )

    cv_results.append({
        "model": name,
        "accuracy_mean": scores.mean(),
        "accuracy_std": scores.std(),
        "balanced_accuracy_mean": balanced_scores.mean(),
    })


cv_results_df = pd.DataFrame(cv_results).sort_values(
    "accuracy_mean",
    ascending=False
)

print_header("6. TARGETED MODEL COMPARISON")

print(
    cv_results_df.to_string(index=False)
)

cv_results_df.to_csv(
    "step16_compass_rc_cv.csv",
    index=False
)

print("\nSaved:")
print("  step16_compass_rc_cv.csv")


# ---------------------------------------------------------------------
# 7. EXTRA TREES FEATURE IMPORTANCE
# ---------------------------------------------------------------------

print_header("7. COMPASS vs RC FEATURE IMPORTANCE")

imputer = SimpleImputer(strategy="median")

X_imp = imputer.fit_transform(X_target)

et = ExtraTreesClassifier(
    n_estimators=1000,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
    max_features="sqrt",
)

et.fit(X_imp, y_target)

importance = pd.DataFrame({
    "feature": valid_features,
    "importance": et.feature_importances_,
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

print("\nTop 50 Compass-vs-RC features:")

print(
    importance.head(50).to_string(index=False)
)

importance.to_csv(
    "step16_compass_rc_feature_importance.csv",
    index=False
)

print("\nSaved:")
print("  step16_compass_rc_feature_importance.csv")


# ---------------------------------------------------------------------
# 8. OVERLAP BETWEEN EFFECT-SIZE FEATURES AND MODEL FEATURES
# ---------------------------------------------------------------------

print_header("8. FEATURE AGREEMENT CHECK")

effect_top = set(
    feature_stats.head(TOP_N)["feature"]
)

importance_top = set(
    importance.head(TOP_N)["feature"]
)

overlap = sorted(
    effect_top.intersection(importance_top)
)

print(
    f"Top-{TOP_N} effect-size features: "
    f"{len(effect_top)}"
)

print(
    f"Top-{TOP_N} model-importance features: "
    f"{len(importance_top)}"
)

print(
    f"Overlap: {len(overlap)}"
)

print("\nFeatures appearing in both lists:")

for f in overlap:
    print("  ", f)


# ---------------------------------------------------------------------
# 9. STEP 15 PREDICTIONS FOR COMPASS TEST RUNS
# ---------------------------------------------------------------------

print_header("9. STEP 15 COMPASS TEST ERRORS")

if "predicted_failure_type" in pred.columns:

    pred_compass = pred[
        pred["run"].isin(
            compass_test["run"]
        )
    ].copy()

    if len(pred_compass) > 0:

        cols = [
            c for c in [
                "run",
                "true_failure_type",
                "predicted_failure_type",
            ]
            if c in pred_compass.columns
        ]

        print(
            pred_compass[cols].to_string(index=False)
        )

    else:
        print(
            "No Compass test runs found in prediction file."
        )

else:
    print(
        "Prediction file does not contain "
        "'predicted_failure_type'."
    )


# ---------------------------------------------------------------------
# 10. SUMMARY
# ---------------------------------------------------------------------

print_header("10. DIAGNOSTIC SUMMARY")

best_model = cv_results_df.iloc[0]

print(
    f"Training Compass runs : {len(compass_train)}"
)

print(
    f"Training RC runs      : {len(rc_train)}"
)

print(
    f"Test Compass runs     : {len(compass_test)}"
)

print(
    f"Best targeted model   : {best_model['model']}"
)

print(
    f"Targeted CV accuracy  : "
    f"{best_model['accuracy_mean']:.4f}"
)

print(
    f"Targeted balanced acc : "
    f"{best_model['balanced_accuracy_mean']:.4f}"
)

print(
    f"Top-feature overlap   : "
    f"{len(overlap)}/{TOP_N}"
)


# ---------------------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------------------

print_header("11. INTERPRETATION")

if len(compass_test) > 0 and not similarity.empty:

    rc_like = (
        similarity["closer_to"] == "RC Failure"
    ).sum()

    compass_like = (
        similarity["closer_to"] == "Compass Failure"
    ).sum()

    print(
        f"Test Compass runs closer to Compass centroid: "
        f"{compass_like}/{len(similarity)}"
    )

    print(
        f"Test Compass runs closer to RC centroid: "
        f"{rc_like}/{len(similarity)}"
    )

    if rc_like > 0:
        print(
            "\nIMPORTANT:"
        )
        print(
            "At least one test Compass run is "
            "numerically closer to the RC training signature."
        )
        print(
            "This suggests genuine feature overlap may "
            "be contributing to the Compass → RC errors."
        )

    else:
        print(
            "\nThe Compass test runs are all closer to "
            "the Compass training centroid."
        )
        print(
            "The Compass → RC errors may therefore be "
            "caused by nonlinear feature interactions, "
            "feature weighting, or limited training examples."
        )


if best_model["accuracy_mean"] >= 0.90:

    print(
        "\nTargeted Compass-vs-RC separation is strong."
    )
    print(
        "The problem is likely not a complete lack of "
        "Compass/RC signal."
    )

elif best_model["accuracy_mean"] >= 0.75:

    print(
        "\nTargeted Compass-vs-RC separation is moderate."
    )
    print(
        "There is useful signal, but Compass and RC "
        "signatures overlap."
    )

else:

    print(
        "\nTargeted Compass-vs-RC separation is weak."
    )
    print(
        "The available run-level features may not cleanly "
        "separate Compass from RC."
    )


# ---------------------------------------------------------------------
# SAVE SUMMARY
# ---------------------------------------------------------------------

summary_rows = [{
    "training_compass_runs": len(compass_train),
    "training_rc_runs": len(rc_train),
    "test_compass_runs": len(compass_test),
    "best_model": best_model["model"],
    "targeted_cv_accuracy": best_model["accuracy_mean"],
    "targeted_cv_balanced_accuracy":
        best_model["balanced_accuracy_mean"],
    "top_feature_overlap": len(overlap),
    "top_feature_overlap_fraction":
        len(overlap) / TOP_N,
}]

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    "step16_compass_rc_summary.csv",
    index=False
)


# ---------------------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------------------

print_header("STEP 16 COMPLETE")

print("Saved outputs:")
print("  step16_compass_rc_feature_stats.csv")
print("  step16_compass_rc_test_similarity.csv")
print("  step16_compass_rc_cv.csv")
print("  step16_compass_rc_feature_importance.csv")
print("  step16_compass_rc_summary.csv")

print("\nNext action:")
print(
    "Paste the COMPLETE terminal output here."
)

print(
    "\nDo NOT change Step 15 yet."
)

print(
    "We will use Step 16 to decide whether the next improvement "
    "should target features, Compass/RC discrimination, or "
    "the training representation."
)

print("=" * 70)
