#!/usr/bin/env python3

"""
======================================================================
STEP 18 — TEMPORAL / WINDOWED RUN-LEVEL FEATURE ENGINEERING
======================================================================

Purpose
-------
Test whether temporal failure signatures are being lost by whole-run
aggregation.

Representation:

    sample-level telemetry
            ->
    independent runs
            ->
    global + temporal-window statistics
            ->
    run-level classifier

IMPORTANT
---------
Step 15 is NOT modified.

This experiment uses:
    fdi_train.csv
    fdi_test.csv

The test labels are NEVER used during feature selection, CV, or model
selection.

Temporal windows are constructed from:
    relative_to_failure

when available.

Expected structure:
    - 10,000 samples/run
    - relative_to_failure approximately -5000 ... +4999

The model is trained on runs, not individual samples.
======================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_selection import SelectKBest, f_classif


# ======================================================================
# CONFIGURATION
# ======================================================================

TRAIN_FILE = "fdi_train.csv"
TEST_FILE = "fdi_test.csv"

RUN_COL = "run"
TARGET_COL = "failure_type"
RELATIVE_COL = "relative_to_failure"
SAMPLE_INDEX_COL = "sample_index"

# Minimum number of samples required for a run
MIN_RUN_SAMPLES = 100

# Temporal windows.
#
# The data seen in previous steps indicates:
#
#     relative_to_failure:
#         -5000 ... +4999
#
# Therefore these windows are meaningful:
#
# PRE-FAILURE:
#     -5000 to -2501
#     -2500 to -1001
#     -1000 to -1
#
# AROUND FAILURE:
#     0 to 999
#     1000 to 2499
#     2500 to 4999
#
# We also include the complete run.
WINDOWS = {
    "global": None,

    "pre_early": (-5000, -2501),
    "pre_mid": (-2500, -1001),
    "pre_late": (-1000, -1),

    "post_early": (0, 999),
    "post_mid": (1000, 2499),
    "post_late": (2500, 4999),
}

# Statistics used per telemetry signal.
#
# Keep this reasonably compact because we only have 56 training runs.
STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "q25",
    "q75",
    "q95",
    "first",
    "last",
    "abs_change",
    "diff_std",
    "diff_mean_abs",
]

# We do NOT want thousands of noisy features to dominate a dataset
# containing only 56 training examples.
#
# First perform univariate selection, then train ExtraTrees.
TOP_K_FEATURES = 500

RANDOM_STATE = 42


# ======================================================================
# PRINT HELPERS
# ======================================================================

def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def section(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ======================================================================
# DATA LOADING
# ======================================================================

def load_dataset(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nCould not find {description} dataset:\n"
            f"  {path}\n\n"
            f"Current directory:\n"
            + "\n".join(
                f"  {x}" for x in sorted(os.listdir("."))
            )
        )

    print(f"Loading {description}: {path}")

    df = pd.read_csv(path)

    print(f"{description} rows: {len(df):,}")
    print(f"{description} columns: {len(df.columns):,}")

    return df


# ======================================================================
# BASIC VALIDATION
# ======================================================================

def validate_dataset(df, name):
    required = [RUN_COL, TARGET_COL]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} dataset is missing required columns: {missing}\n"
            f"Available columns include:\n"
            f"{list(df.columns[:50])}"
        )

    if df[RUN_COL].isna().any():
        raise ValueError(f"{name}: null values found in '{RUN_COL}'")

    if df[TARGET_COL].isna().any():
        raise ValueError(f"{name}: null values found in '{TARGET_COL}'")


# ======================================================================
# NUMERIC TELEMETRY COLUMN SELECTION
# ======================================================================

def get_numeric_telemetry_columns(train_df, test_df):
    common = sorted(
        set(train_df.columns).intersection(test_df.columns)
    )

    excluded = {
        RUN_COL,
        TARGET_COL,
        RELATIVE_COL,
        SAMPLE_INDEX_COL,
        "metadata",
        "failure",
        "label",
        "prediction",
    }

    telemetry = []

    for col in common:
        if col in excluded:
            continue

        train_numeric = pd.to_numeric(
            train_df[col], errors="coerce"
        )

        test_numeric = pd.to_numeric(
            test_df[col], errors="coerce"
        )

        # Require at least some numeric information in both datasets.
        if train_numeric.notna().sum() > 0 and test_numeric.notna().sum() > 0:
            telemetry.append(col)

    return telemetry


# ======================================================================
# SAFE STATISTICS
# ======================================================================

def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return 0.0
    except Exception:
        return 0.0


def compute_series_statistics(series):
    """
    Return compact temporal statistics for one telemetry signal.
    """

    values = pd.to_numeric(series, errors="coerce").dropna()

    result = {}

    if len(values) == 0:
        for stat in STATISTICS:
            result[stat] = 0.0
        return result

    arr = values.to_numpy(dtype=float)

    result["mean"] = safe_float(np.mean(arr))

    if len(arr) > 1:
        result["std"] = safe_float(np.std(arr, ddof=1))
    else:
        result["std"] = 0.0

    result["min"] = safe_float(np.min(arr))
    result["max"] = safe_float(np.max(arr))
    result["range"] = safe_float(np.max(arr) - np.min(arr))

    result["q25"] = safe_float(np.quantile(arr, 0.25))
    result["q75"] = safe_float(np.quantile(arr, 0.75))
    result["q95"] = safe_float(np.quantile(arr, 0.95))

    result["first"] = safe_float(arr[0])
    result["last"] = safe_float(arr[-1])

    if len(arr) > 1:
        diff = np.diff(arr)

        result["abs_change"] = safe_float(
            np.sum(np.abs(diff))
        )

        result["diff_std"] = safe_float(
            np.std(diff, ddof=1)
        )

        result["diff_mean_abs"] = safe_float(
            np.mean(np.abs(diff))
        )
    else:
        result["abs_change"] = 0.0
        result["diff_std"] = 0.0
        result["diff_mean_abs"] = 0.0

    return result


# ======================================================================
# RUN FEATURE EXTRACTION
# ======================================================================

def extract_run_features(
    df,
    telemetry_columns,
    dataset_name
):
    """
    Create exactly one feature row per run.

    Each telemetry feature gets statistics from:

        global
        pre_early
        pre_mid
        pre_late
        post_early
        post_mid
        post_late

    We additionally create temporal-difference features between selected
    windows.
    """

    rows = []

    grouped = df.groupby(RUN_COL, sort=False)

    total_runs = grouped.ngroups

    print(
        f"\nExtracting temporal features from "
        f"{total_runs} {dataset_name} runs..."
    )

    has_relative = RELATIVE_COL in df.columns

    for run_number, (run_id, run_df) in enumerate(grouped, start=1):

        run_features = {
            RUN_COL: run_id
        }

        # --------------------------------------------------------------
        # LABEL
        # --------------------------------------------------------------

        labels = run_df[TARGET_COL].dropna().astype(str).unique()

        if len(labels) == 1:
            run_features[TARGET_COL] = labels[0]
        elif len(labels) > 1:
            # This dataset should contain exactly one failure class/run.
            # Use dominant class rather than crashing.
            dominant = (
                run_df[TARGET_COL]
                .astype(str)
                .value_counts()
                .idxmax()
            )
            run_features[TARGET_COL] = dominant
        else:
            run_features[TARGET_COL] = "Unknown"

        # --------------------------------------------------------------
        # SORT RUN
        # --------------------------------------------------------------

        local_df = run_df.copy()

        if (
            has_relative
            and local_df[RELATIVE_COL].notna().sum() > 0
        ):
            local_df = local_df.sort_values(
                RELATIVE_COL
            )

        elif SAMPLE_INDEX_COL in local_df.columns:
            local_df = local_df.sort_values(
                SAMPLE_INDEX_COL
            )

        # --------------------------------------------------------------
        # TEMPORAL WINDOWS
        # --------------------------------------------------------------

        window_data = {}

        # Global window
        window_data["global"] = local_df

        if has_relative:
            relative = pd.to_numeric(
                local_df[RELATIVE_COL],
                errors="coerce"
            )

            for window_name, bounds in WINDOWS.items():

                if window_name == "global":
                    continue

                low, high = bounds

                mask = (
                    (relative >= low)
                    & (relative <= high)
                )

                window_data[window_name] = local_df.loc[mask]

        else:
            # If relative_to_failure is unavailable, create positional
            # windows based on the ordered samples.
            n = len(local_df)

            positional_windows = {
                "pre_early": (0, int(n * 0.25)),
                "pre_mid": (int(n * 0.25), int(n * 0.40)),
                "pre_late": (int(n * 0.40), int(n * 0.50)),
                "post_early": (int(n * 0.50), int(n * 0.60)),
                "post_mid": (int(n * 0.60), int(n * 0.75)),
                "post_late": (int(n * 0.75), n),
            }

            for window_name, (start, end) in positional_windows.items():
                window_data[window_name] = local_df.iloc[start:end]

        # --------------------------------------------------------------
        # EXTRACT STATISTICS
        # --------------------------------------------------------------

        for window_name, window_df in window_data.items():

            for feature in telemetry_columns:

                if feature not in window_df.columns:
                    continue

                stats = compute_series_statistics(
                    window_df[feature]
                )

                for stat_name, value in stats.items():

                    feature_name = (
                        f"{feature}__{window_name}__{stat_name}"
                    )

                    run_features[feature_name] = value

        # --------------------------------------------------------------
        # TEMPORAL DIFFERENCE FEATURES
        #
        # These directly test whether the signal changes around the
        # failure point rather than merely having a different global mean.
        # --------------------------------------------------------------

        comparison_pairs = [
            ("pre_late", "post_early"),
            ("pre_mid", "post_mid"),
            ("pre_early", "post_late"),
        ]

        for left_name, right_name in comparison_pairs:

            left_df = window_data.get(
                left_name,
                pd.DataFrame()
            )

            right_df = window_data.get(
                right_name,
                pd.DataFrame()
            )

            if len(left_df) == 0 or len(right_df) == 0:
                continue

            for feature in telemetry_columns:

                if (
                    feature not in left_df.columns
                    or feature not in right_df.columns
                ):
                    continue

                left_values = pd.to_numeric(
                    left_df[feature],
                    errors="coerce"
                ).dropna()

                right_values = pd.to_numeric(
                    right_df[feature],
                    errors="coerce"
                ).dropna()

                if len(left_values) == 0 or len(right_values) == 0:
                    continue

                left_mean = float(left_values.mean())
                right_mean = float(right_values.mean())

                left_std = float(left_values.std()) if len(left_values) > 1 else 0.0
                right_std = float(right_values.std()) if len(right_values) > 1 else 0.0

                base = f"{feature}__{left_name}_vs_{right_name}"

                run_features[f"{base}__mean_delta"] = (
                    right_mean - left_mean
                )

                run_features[f"{base}__abs_mean_delta"] = (
                    abs(right_mean - left_mean)
                )

                run_features[f"{base}__std_delta"] = (
                    right_std - left_std
                )

        rows.append(run_features)

        if run_number % 10 == 0 or run_number == total_runs:
            print(
                f"  processed {run_number}/{total_runs} runs"
            )

    result = pd.DataFrame(rows)

    # Replace infinite values.
    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return result


# ======================================================================
# FEATURE MATRIX PREPARATION
# ======================================================================

def prepare_feature_matrices(
    train_features,
    test_features
):

    feature_columns = sorted(
        set(train_features.columns)
        .intersection(test_features.columns)
    )

    feature_columns = [
        c for c in feature_columns
        if c not in {RUN_COL, TARGET_COL}
    ]

    train_matrix = train_features[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    test_matrix = test_features[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove features which are completely missing in training.
    valid = (
        train_matrix.notna().sum(axis=0) > 0
    )

    train_matrix = train_matrix.loc[:, valid]
    test_matrix = test_matrix.loc[:, valid]

    feature_columns = train_matrix.columns.tolist()

    # Fill using TRAIN medians only.
    train_medians = train_matrix.median()

    train_matrix = train_matrix.fillna(
        train_medians
    )

    test_matrix = test_matrix.fillna(
        train_medians
    )

    # Remove constants based ONLY on training data.
    nunique = train_matrix.nunique()

    variable = nunique > 1

    train_matrix = train_matrix.loc[:, variable]
    test_matrix = test_matrix.loc[:, variable]

    feature_columns = train_matrix.columns.tolist()

    return (
        train_matrix,
        test_matrix,
        feature_columns
    )


# ======================================================================
# FEATURE SELECTION
# ======================================================================

def select_features_training_only(
    X_train,
    y_train,
    X_test,
    top_k
):
    """
    Select features using ONLY training data.

    Important:
    test data never participates in feature selection.
    """

    n_features = X_train.shape[1]

    if n_features <= top_k:
        return (
            X_train.copy(),
            X_test.copy(),
            X_train.columns.tolist(),
            None
        )

    # Convert labels to integer for ANOVA selector.
    y_codes = pd.Series(y_train).astype("category").cat.codes

    k = min(top_k, n_features)

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train_selected = selector.fit_transform(
        X_train,
        y_codes
    )

    X_test_selected = selector.transform(
        X_test
    )

    selected_columns = (
        X_train.columns[
            selector.get_support()
        ].tolist()
    )

    X_train_selected = pd.DataFrame(
        X_train_selected,
        columns=selected_columns,
        index=X_train.index
    )

    X_test_selected = pd.DataFrame(
        X_test_selected,
        columns=selected_columns,
        index=X_test.index
    )

    return (
        X_train_selected,
        X_test_selected,
        selected_columns,
        selector
    )


# ======================================================================
# MODEL EVALUATION
# ======================================================================

def evaluate_cv(
    name,
    model,
    X,
    y,
    cv
):
    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    accuracy_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    balanced_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy"
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

    return {
        "model": name,
        "accuracy_mean": accuracy_scores.mean(),
        "accuracy_std": accuracy_scores.std(),
        "balanced_accuracy_mean": balanced_scores.mean(),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "STEP 18 — TEMPORAL / WINDOWED RUN-LEVEL FEATURE ENGINEERING"
    )

    print(
        """
Purpose:
    Test whether temporal failure signatures are being lost by
    whole-run aggregation.

Step 15 remains unchanged.

Training:
    fdi_train.csv

Testing:
    fdi_test.csv

Representation:
    samples
       ->
    temporal windows around failure
       ->
    one row per run
       ->
    classifier
    """
    )

    # ==================================================================
    # 1. LOAD
    # ==================================================================

    section("1. LOADING DATASETS")

    train_df = load_dataset(
        TRAIN_FILE,
        "Training"
    )

    test_df = load_dataset(
        TEST_FILE,
        "Test"
    )

    validate_dataset(
        train_df,
        "Training"
    )

    validate_dataset(
        test_df,
        "Test"
    )

    train_runs = train_df[RUN_COL].nunique()
    test_runs = test_df[RUN_COL].nunique()

    print(f"\nTraining runs: {train_runs}")
    print(f"Test runs    : {test_runs}")

    # ==================================================================
    # 2. DATASET STRUCTURE
    # ==================================================================

    section("2. DATASET STRUCTURE")

    train_run_sizes = train_df.groupby(RUN_COL).size()
    test_run_sizes = test_df.groupby(RUN_COL).size()

    print(
        "Training samples/run:"
    )
    print(
        train_run_sizes.describe()
    )

    print(
        "\nTest samples/run:"
    )
    print(
        test_run_sizes.describe()
    )

    train_class_counts = (
        train_df.groupby(RUN_COL)[TARGET_COL]
        .first()
        .value_counts()
    )

    test_class_counts = (
        test_df.groupby(RUN_COL)[TARGET_COL]
        .first()
        .value_counts()
    )

    print(
        "\nTraining run classes:"
    )
    print(train_class_counts)

    print(
        "\nTest run classes:"
    )
    print(test_class_counts)

    if RELATIVE_COL in train_df.columns:
        rel = pd.to_numeric(
            train_df[RELATIVE_COL],
            errors="coerce"
        )

        print(
            "\nrelative_to_failure available: YES"
        )

        print(
            f"Range: "
            f"{rel.min()} to {rel.max()}"
        )

    else:
        print(
            "\nrelative_to_failure available: NO"
        )
        print(
            "Using positional windows instead."
        )

    # ==================================================================
    # 3. FEATURE SELECTION
    # ==================================================================

    section("3. TELEMETRY FEATURE SELECTION")

    telemetry_columns = get_numeric_telemetry_columns(
        train_df,
        test_df
    )

    print(
        f"Numeric telemetry columns: "
        f"{len(telemetry_columns)}"
    )

    print(
        "\nFirst 30 telemetry features:"
    )

    print(
        telemetry_columns[:30]
    )

    # ==================================================================
    # 4. TEMPORAL FEATURE ENGINEERING
    # ==================================================================

    section(
        "4. TEMPORAL RUN-LEVEL FEATURE ENGINEERING"
    )

    print(
        "\nTemporal windows:"
    )

    for name, bounds in WINDOWS.items():
        if bounds is None:
            print(
                f"  {name}: complete run"
            )
        else:
            print(
                f"  {name}: "
                f"{bounds[0]} to {bounds[1]}"
            )

    train_features = extract_run_features(
        train_df,
        telemetry_columns,
        "training"
    )

    test_features = extract_run_features(
        test_df,
        telemetry_columns,
        "test"
    )

    print(
        "\nFeature matrices before filtering:"
    )

    print(
        f"Train: {train_features.shape}"
    )

    print(
        f"Test : {test_features.shape}"
    )

    # Save raw temporal features.
    train_features.to_csv(
        "step18_temporal_features_train.csv",
        index=False
    )

    test_features.to_csv(
        "step18_temporal_features_test.csv",
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "  step18_temporal_features_train.csv"
    )

    print(
        "  step18_temporal_features_test.csv"
    )

    # ==================================================================
    # 5. PREPARE MATRICES
    # ==================================================================

    section("5. MODEL MATRIX")

    X_train, X_test, feature_columns = (
        prepare_feature_matrices(
            train_features,
            test_features
        )
    )

    y_train = (
        train_features[TARGET_COL]
        .astype(str)
        .values
    )

    y_test = (
        test_features[TARGET_COL]
        .astype(str)
        .values
    )

    print(
        f"Features after cleaning: "
        f"{len(feature_columns)}"
    )

    print(
        f"Training matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Test matrix: "
        f"{X_test.shape}"
    )

    # ==================================================================
    # 6. TRAINING-ONLY FEATURE SELECTION
    # ==================================================================

    section(
        "6. TRAINING-ONLY FEATURE SELECTION"
    )

    print(
        f"Features before selection: "
        f"{X_train.shape[1]}"
    )

    print(
        f"TOP_K_FEATURES: "
        f"{TOP_K_FEATURES}"
    )

    (
        X_train_selected,
        X_test_selected,
        selected_features,
        selector
    ) = select_features_training_only(
        X_train,
        y_train,
        X_test,
        TOP_K_FEATURES
    )

    print(
        f"Features after selection: "
        f"{len(selected_features)}"
    )

    pd.DataFrame({
        "feature": selected_features
    }).to_csv(
        "step18_selected_features.csv",
        index=False
    )

    # ==================================================================
    # 7. CROSS-VALIDATION
    # ==================================================================

    section("7. CROSS-VALIDATION")

    class_counts = (
        pd.Series(y_train)
        .value_counts()
    )

    min_class_count = int(
        class_counts.min()
    )

    n_splits = min(
        5,
        min_class_count
    )

    if n_splits < 2:
        raise ValueError(
            "Not enough training runs per class "
            "for stratified cross-validation."
        )

    print(
        f"Using {n_splits}-fold stratified CV."
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------------
    # MODELS
    # --------------------------------------------------------------

    models = {

        "LogisticRegression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    C=0.1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    cv_results = []

    for name, model in models.items():

        result = evaluate_cv(
            name,
            model,
            X_train_selected,
            y_train,
            cv
        )

        cv_results.append(
            result
        )

    cv_df = pd.DataFrame(
        cv_results
    ).sort_values(
        "balanced_accuracy_mean",
        ascending=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CV MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        cv_df.to_string(
            index=False
        )
    )

    cv_df.to_csv(
        "step18_cv_results.csv",
        index=False
    )

    # ==================================================================
    # 8. BEST MODEL
    # ==================================================================

    section("8. BEST TEMPORAL MODEL")

    best_name = cv_df.iloc[0]["model"]

    print(
        f"Selected model: {best_name}"
    )

    best_model = models[
        best_name
    ]

    # ==================================================================
    # 9. TRAIN FINAL MODEL
    # ==================================================================

    section(
        "9. TRAINING FINAL TEMPORAL MODEL"
    )

    best_model.fit(
        X_train_selected,
        y_train
    )

    print(
        "Training complete."
    )

    # ==================================================================
    # 10. TEST PREDICTIONS
    # ==================================================================

    section(
        "10. TEST RUN PREDICTIONS"
    )

    predictions = best_model.predict(
        X_test_selected
    )

    test_results = pd.DataFrame({
        RUN_COL:
            test_features[RUN_COL].values,

        TARGET_COL:
            y_test,

        "predicted_failure_type":
            predictions,
    })

    test_results["correct"] = (
        test_results[TARGET_COL]
        ==
        test_results[
            "predicted_failure_type"
        ]
    )

    print(
        test_results.to_string(
            index=False
        )
    )

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        pd.Series(
            predictions,
            name="predicted_failure_type"
        ).value_counts()
    )

    # ==================================================================
    # 11. PERFORMANCE
    # ==================================================================

    section(
        "11. FINAL TEST PERFORMANCE"
    )

    print(
        f"Run-level accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Run-level balanced accuracy: "
        f"{test_balanced_accuracy:.4f}"
    )

    print(
        f"Correct runs: "
        f"{int(test_results['correct'].sum())}/"
        f"{len(test_results)}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    labels = sorted(
        set(y_test)
        .union(set(predictions))
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        cm_df
    )

    cm_df.to_csv(
        "step18_confusion_matrix.csv"
    )

    # ==================================================================
    # 12. PER-CLASS PERFORMANCE
    # ==================================================================

    section(
        "12. PER-CLASS TEST PERFORMANCE"
    )

    per_class = []

    for cls in sorted(
        pd.Series(y_test).unique()
    ):

        mask = (
            y_test == cls
        )

        total = int(
            mask.sum()
        )

        correct = int(
            (
                predictions[mask]
                == cls
            ).sum()
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        per_class.append({
            "failure_type": cls,
            "total_test_runs": total,
            "correct": correct,
            "accuracy": accuracy
        })

    per_class_df = pd.DataFrame(
        per_class
    )

    print(
        per_class_df.to_string(
            index=False
        )
    )

    per_class_df.to_csv(
        "step18_per_class_results.csv",
        index=False
    )

    # ==================================================================
    # 13. RUN-BY-RUN COMPARISON
    # ==================================================================

    section(
        "13. RUN-BY-RUN RESULTS"
    )

    # Try to load Step 15 predictions so we can directly compare.
    step15_file = (
        "step15_test_predictions.csv"
    )

    if os.path.exists(step15_file):

        step15 = pd.read_csv(
            step15_file
        )

        print(
            "\nStep 15 predictions found."
        )

        # Identify prediction column robustly.
        prediction_candidates = [
            "predicted_failure_type",
            "prediction",
            "baseline_prediction"
        ]

        step15_pred_col = None

        for col in prediction_candidates:
            if col in step15.columns:
                step15_pred_col = col
                break

        if step15_pred_col is not None:

            comparison = test_results[
                [
                    RUN_COL,
                    TARGET_COL,
                    "predicted_failure_type"
                ]
            ].copy()

            comparison = comparison.merge(
                step15[
                    [
                        RUN_COL,
                        step15_pred_col
                    ]
                ],
                on=RUN_COL,
                how="left"
            )

            comparison = comparison.rename(
                columns={
                    step15_pred_col:
                        "step15_prediction"
                }
            )

            comparison[
                "step15_correct"
            ] = (
                comparison[TARGET_COL]
                ==
                comparison[
                    "step15_prediction"
                ]
            )

            comparison[
                "step18_correct"
            ] = (
                comparison[TARGET_COL]
                ==
                comparison[
                    "predicted_failure_type"
                ]
            )

            comparison[
                "changed"
            ] = (
                comparison[
                    "step15_prediction"
                ]
                !=
                comparison[
                    "predicted_failure_type"
                ]
            )

            print(
                comparison.to_string(
                    index=False
                )
            )

            changed = int(
                comparison["changed"].sum()
            )

            improved = int(
                (
                    (~comparison["step15_correct"])
                    &
                    (comparison["step18_correct"])
                ).sum()
            )

            worsened = int(
                (
                    (comparison["step15_correct"])
                    &
                    (~comparison["step18_correct"])
                ).sum()
            )

            print(
                f"\nChanged predictions: {changed}"
            )

            print(
                f"Improved runs: {improved}"
            )

            print(
                f"Worsened runs: {worsened}"
            )

            comparison.to_csv(
                "step18_vs_step15.csv",
                index=False
            )

        else:
            print(
                "Could not identify Step 15 "
                "prediction column."
            )

    else:

        print(
            "Step 15 prediction file not found."
        )

    # ==================================================================
    # 14. FEATURE IMPORTANCE
    # ==================================================================

    section(
        "14. FEATURE IMPORTANCE"
    )

    importance_df = None

    # ExtraTrees and RandomForest expose feature_importances_ directly.
    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importances = (
            best_model.feature_importances_
        )

        importance_df = pd.DataFrame({
            "feature":
                X_train_selected.columns,

            "importance":
                importances
        })

    # LogisticRegression is inside a pipeline.
    elif isinstance(
        best_model,
        Pipeline
    ):

        final_estimator = (
            best_model.steps[-1][1]
        )

        if hasattr(
            final_estimator,
            "coef_"
        ):

            coefficients = np.abs(
                final_estimator.coef_
            )

            if coefficients.ndim == 2:
                coefficients = (
                    coefficients.mean(axis=0)
                )

            importance_df = pd.DataFrame({
                "feature":
                    X_train_selected.columns,

                "importance":
                    coefficients
            })

    if importance_df is not None:

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        print(
            "\nTop 50 temporal features:"
        )

        print(
            importance_df.head(50).to_string(
                index=False
            )
        )

        importance_df.to_csv(
            "step18_feature_importance.csv",
            index=False
        )

    else:

        print(
            "Feature importance unavailable."
        )

    # ==================================================================
    # 15. TEMPORAL SIGNATURE SUMMARY
    # ==================================================================

    section(
        "15. TEMPORAL SIGNATURE SUMMARY"
    )

    # Count feature groups.
    window_counts = {}

    for window_name in WINDOWS.keys():

        prefix = "__" + window_name + "__"

        count = sum(
            prefix in feature
            for feature in selected_features
        )

        window_counts[
            window_name
        ] = count

    print(
        "Selected feature count by temporal window:"
    )

    for window_name, count in window_counts.items():

        print(
            f"  {window_name:12s}: {count}"
        )

    temporal_summary = pd.DataFrame([
        {
            "window": k,
            "selected_feature_count": v
        }
        for k, v in window_counts.items()
    ])

    temporal_summary.to_csv(
        "step18_temporal_window_summary.csv",
        index=False
    )

    # ==================================================================
    # 16. SAVE RESULTS
    # ==================================================================

    section(
        "16. SAVING RESULTS"
    )

    test_results.to_csv(
        "step18_test_predictions.csv",
        index=False
    )

    pd.DataFrame({
        "run": test_features[RUN_COL],
        "true_failure_type": y_test,
        "predicted_failure_type": predictions,
        "correct": test_results["correct"]
    }).to_csv(
        "step18_results.csv",
        index=False
    )

    summary = pd.DataFrame([{
        "train_runs":
            train_runs,

        "test_runs":
            test_runs,

        "raw_features":
            len(feature_columns),

        "selected_features":
            len(selected_features),

        "best_model":
            best_name,

        "cv_accuracy":
            cv_df.iloc[0]["accuracy_mean"],

        "cv_balanced_accuracy":
            cv_df.iloc[0]["balanced_accuracy_mean"],

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced_accuracy,

        "correct_test_runs":
            int(test_results["correct"].sum()),

        "total_test_runs":
            len(test_results)
    }])

    summary.to_csv(
        "step18_summary.csv",
        index=False
    )

    # ==================================================================
    # FINAL
    # ==================================================================

    header(
        "STEP 18 COMPLETE"
    )

    print(
        f"""
Training runs       : {train_runs}
Test runs           : {test_runs}

Raw features        : {len(feature_columns)}
Selected features   : {len(selected_features)}

Best model          : {best_name}

CV accuracy         : {cv_df.iloc[0]["accuracy_mean"]:.4f}
CV balanced acc     : {cv_df.iloc[0]["balanced_accuracy_mean"]:.4f}

Test accuracy       : {test_accuracy:.4f}
Test balanced acc   : {test_balanced_accuracy:.4f}

Correct test runs   : {int(test_results["correct"].sum())}/{len(test_results)}

Saved outputs:
  step18_temporal_features_train.csv
  step18_temporal_features_test.csv
  step18_selected_features.csv
  step18_cv_results.csv
  step18_confusion_matrix.csv
  step18_per_class_results.csv
  step18_test_predictions.csv
  step18_results.csv
  step18_feature_importance.csv
  step18_temporal_window_summary.csv
  step18_summary.csv
"""
    )

    print(
        "=" * 70
    )

    print(
        "NEXT DECISION"
    )

    print(
        "=" * 70
    )

    print(
        """
The important comparison is NOT just whether Step 18's CV score
looks good.

Compare:

    Step 15 test accuracy
        vs
    Step 18 test accuracy

and especially inspect:

    Compass Failure
        recall

If temporal features improve Compass without damaging the other
classes, this becomes the new direction.

If Step 18 does NOT improve the 12/14 result, we should stop adding
more feature engineering and investigate the actual train/test
distribution shift or the construction of the Compass runs.
"""
    )


if __name__ == "__main__":
    main()
