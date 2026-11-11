#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold


# ================================================================
# STEP 33A
# STRICT LEAKAGE-SAFE GENERALIZATION / OVERFITTING CHECK
# ================================================================

TRAIN_FILE = "step31_sead_fdi_train_run_level.csv"
TEST_FILE = "step31_sead_fdi_test_run_level.csv"

LABEL_COL = "fdi_failure_type"
RUN_COL = "run_key"

TOP_K = 750

OUTPUT_SUMMARY = "step33A_generalization_summary.csv"
OUTPUT_FOLDS = "step33A_fold_results.csv"
OUTPUT_OOF = "step33A_oof_predictions.csv"
OUTPUT_CONFUSION = "step33A_oof_confusion_matrix.csv"


# ================================================================
# ETA-SQUARED
# Same method used in Step 32
# ================================================================

def eta_squared(X, y):

    y_codes = (
        pd.Series(y)
        .astype("category")
        .cat.codes
        .to_numpy()
    )

    X = np.asarray(X, dtype=float)

    global_mean = np.nanmean(X, axis=0)

    ss_total = np.nansum(
        (X - global_mean) ** 2,
        axis=0
    )

    ss_between = np.zeros(
        X.shape[1],
        dtype=float
    )

    for cls in np.unique(y_codes):

        mask = y_codes == cls
        n = np.sum(mask)

        if n == 0:
            continue

        cls_mean = np.nanmean(
            X[mask],
            axis=0
        )

        ss_between += (
            n * (cls_mean - global_mean) ** 2
        )

    eta = np.divide(
        ss_between,
        ss_total,
        out=np.zeros_like(ss_between),
        where=ss_total > 0,
    )

    eta[~np.isfinite(eta)] = 0.0

    eta = np.clip(
        eta,
        0.0,
        1.0
    )

    return eta


# ================================================================
# PREPARE ONE CV FOLD
# ================================================================

def prepare_fold(
    X_train_raw,
    y_train,
    X_val_raw,
    top_k=750
):

    # ------------------------------------------------------------
    # 1. Remove inf
    # ------------------------------------------------------------

    X_train = X_train_raw.replace(
        [np.inf, -np.inf],
        np.nan
    ).copy()

    X_val = X_val_raw.replace(
        [np.inf, -np.inf],
        np.nan
    ).copy()

    # ------------------------------------------------------------
    # 2. Remove constant features
    #    FIT USING FOLD-TRAIN ONLY
    # ------------------------------------------------------------

    nonconstant_features = []

    for c in X_train.columns:

        if X_train[c].nunique(dropna=False) > 1:
            nonconstant_features.append(c)

    X_train = X_train[
        nonconstant_features
    ]

    X_val = X_val[
        nonconstant_features
    ]

    # ------------------------------------------------------------
    # 3. Median imputation
    #    MEDIANS FIT USING FOLD-TRAIN ONLY
    # ------------------------------------------------------------

    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    # ------------------------------------------------------------
    # 4. Remove unusable columns
    #    Again decided from fold-training data only
    # ------------------------------------------------------------

    usable_features = []

    for c in X_train.columns:

        if X_train[c].isna().all():
            continue

        if X_train[c].nunique(dropna=False) <= 1:
            continue

        usable_features.append(c)

    X_train = X_train[
        usable_features
    ]

    X_val = X_val[
        usable_features
    ]

    # ------------------------------------------------------------
    # 5. ETA-SQUARED RANKING
    #    USING FOLD-TRAIN ONLY
    # ------------------------------------------------------------

    eta = eta_squared(
        X_train.to_numpy(dtype=float),
        y_train
    )

    ranking = pd.DataFrame({
        "feature": usable_features,
        "eta_squared": eta
    })

    ranking = ranking.sort_values(
        "eta_squared",
        ascending=False
    ).reset_index(drop=True)

    selected_features = (
        ranking
        .head(min(top_k, len(ranking)))
        ["feature"]
        .tolist()
    )

    # ------------------------------------------------------------
    # 6. Return selected matrices
    # ------------------------------------------------------------

    return (
        X_train[selected_features],
        X_val[selected_features],
        selected_features
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 75)
    print("STEP 33A — STRICT LEAKAGE-SAFE GENERALIZATION CHECK")
    print("=" * 75)
    print()

    # ------------------------------------------------------------
    # 1. LOAD ORIGINAL STEP 31 RUN-LEVEL DATA
    # ------------------------------------------------------------

    print("1. LOADING STEP 31 DATA")
    print("-" * 75)

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print("Training shape:", train.shape)
    print("Test shape:    ", test.shape)

    # ------------------------------------------------------------
    # 2. TRAIN/TEST RUN CHECK
    # ------------------------------------------------------------

    print()
    print("2. RUN-LEVEL TRAIN / TEST CHECK")
    print("-" * 75)

    train_runs = set(train[RUN_COL])
    test_runs = set(test[RUN_COL])

    overlap = train_runs & test_runs

    print("Training runs:", len(train_runs))
    print("Test runs:    ", len(test_runs))
    print("Run overlap:  ", len(overlap))

    if overlap:
        raise RuntimeError(
            "STOP: TRAIN/TEST RUN OVERLAP DETECTED."
        )

    # ------------------------------------------------------------
    # 3. IDENTIFY NUMERIC FEATURES
    # ------------------------------------------------------------

    print()
    print("3. IDENTIFYING NUMERIC FEATURES")
    print("-" * 75)

    metadata = {
        RUN_COL,
        LABEL_COL,
        "label",
        "available_required_streams",
        "total_generated_features",
    }

    candidate_features = [
        c for c in train.columns
        if c not in metadata
        and c in test.columns
    ]

    numeric_features = [
        c for c in candidate_features
        if pd.api.types.is_numeric_dtype(train[c])
    ]

    print("Numeric features:", len(numeric_features))

    X = train[numeric_features].copy()
    y = train[LABEL_COL].astype(str)

    # ------------------------------------------------------------
    # 4. STRICT CV
    # ------------------------------------------------------------

    print()
    print("4. STRICT LEAKAGE-SAFE 5-FOLD CV")
    print("-" * 75)

    print()
    print("IMPORTANT:")
    print("For EVERY fold:")
    print("  - Constant-feature removal uses fold-training only")
    print("  - Median imputation uses fold-training only")
    print("  - Eta-squared ranking uses fold-training only")
    print("  - Top 750 selection uses fold-training only")
    print("  - Validation fold remains untouched")
    print()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_rows = []

    all_true = []
    all_pred = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print("=" * 75)
        print(f"FOLD {fold}")
        print("=" * 75)

        X_fold_train_raw = X.iloc[train_idx].copy()
        X_fold_val_raw = X.iloc[val_idx].copy()

        y_fold_train = y.iloc[train_idx].copy()
        y_fold_val = y.iloc[val_idx].copy()

        print(
            "Fold train runs:",
            len(X_fold_train_raw)
        )

        print(
            "Fold validation runs:",
            len(X_fold_val_raw)
        )

        # --------------------------------------------------------
        # Fold-specific feature preparation
        # --------------------------------------------------------

        (
            X_fold_train,
            X_fold_val,
            selected_features
        ) = prepare_fold(
            X_fold_train_raw,
            y_fold_train,
            X_fold_val_raw,
            top_k=TOP_K
        )

        print(
            "Selected features:",
            len(selected_features)
        )

        # --------------------------------------------------------
        # Exact Step 33 ExtraTrees parameters
        # --------------------------------------------------------

        model = ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        # --------------------------------------------------------
        # FIT
        # --------------------------------------------------------

        model.fit(
            X_fold_train,
            y_fold_train
        )

        # --------------------------------------------------------
        # Training predictions
        # --------------------------------------------------------

        train_pred = model.predict(
            X_fold_train
        )

        train_accuracy = accuracy_score(
            y_fold_train,
            train_pred
        )

        train_macro_f1 = f1_score(
            y_fold_train,
            train_pred,
            average="macro"
        )

        # --------------------------------------------------------
        # Validation predictions
        # --------------------------------------------------------

        val_pred = model.predict(
            X_fold_val
        )

        val_accuracy = accuracy_score(
            y_fold_val,
            val_pred
        )

        val_macro_f1 = f1_score(
            y_fold_val,
            val_pred,
            average="macro"
        )

        print()
        print(
            f"Train Accuracy : {train_accuracy:.4f}"
        )

        print(
            f"Train Macro-F1 : {train_macro_f1:.4f}"
        )

        print(
            f"Val Accuracy   : {val_accuracy:.4f}"
        )

        print(
            f"Val Macro-F1   : {val_macro_f1:.4f}"
        )

        print(
            f"Accuracy Gap   : "
            f"{train_accuracy - val_accuracy:.4f}"
        )

        print(
            f"Macro-F1 Gap   : "
            f"{train_macro_f1 - val_macro_f1:.4f}"
        )

        # --------------------------------------------------------
        # Save fold results
        # --------------------------------------------------------

        fold_rows.append({

            "fold":
                fold,

            "train_runs":
                len(X_fold_train),

            "validation_runs":
                len(X_fold_val),

            "selected_features":
                len(selected_features),

            "train_accuracy":
                train_accuracy,

            "validation_accuracy":
                val_accuracy,

            "train_macro_f1":
                train_macro_f1,

            "validation_macro_f1":
                val_macro_f1,

            "accuracy_gap":
                train_accuracy - val_accuracy,

            "macro_f1_gap":
                train_macro_f1 - val_macro_f1,
        })

        all_true.extend(
            y_fold_val.tolist()
        )

        all_pred.extend(
            val_pred.tolist()
        )

    # ============================================================
    # 5. OVERALL OOF RESULTS
    # ============================================================

    fold_df = pd.DataFrame(
        fold_rows
    )

    oof_accuracy = accuracy_score(
        all_true,
        all_pred
    )

    oof_macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro"
    )

    mean_val_accuracy = (
        fold_df["validation_accuracy"].mean()
    )

    std_val_accuracy = (
        fold_df["validation_accuracy"].std(ddof=0)
    )

    mean_val_f1 = (
        fold_df["validation_macro_f1"].mean()
    )

    std_val_f1 = (
        fold_df["validation_macro_f1"].std(ddof=0)
    )

    mean_train_accuracy = (
        fold_df["train_accuracy"].mean()
    )

    mean_train_f1 = (
        fold_df["train_macro_f1"].mean()
    )

    mean_accuracy_gap = (
        fold_df["accuracy_gap"].mean()
    )

    mean_f1_gap = (
        fold_df["macro_f1_gap"].mean()
    )

    # ============================================================
    # 6. STRICT CV SUMMARY
    # ============================================================

    print()
    print("=" * 75)
    print("STRICT LEAKAGE-SAFE CV SUMMARY")
    print("=" * 75)

    print(
        f"Mean Training Accuracy : "
        f"{mean_train_accuracy:.4f}"
    )

    print(
        f"Mean Validation Accuracy: "
        f"{mean_val_accuracy:.4f} "
        f"± {std_val_accuracy:.4f}"
    )

    print(
        f"OOF Accuracy            : "
        f"{oof_accuracy:.4f}"
    )

    print()

    print(
        f"Mean Training Macro-F1 : "
        f"{mean_train_f1:.4f}"
    )

    print(
        f"Mean Validation Macro-F1: "
        f"{mean_val_f1:.4f} "
        f"± {std_val_f1:.4f}"
    )

    print(
        f"OOF Macro-F1            : "
        f"{oof_macro_f1:.4f}"
    )

    print()

    print(
        f"Mean Accuracy Gap       : "
        f"{mean_accuracy_gap:.4f}"
    )

    print(
        f"Mean Macro-F1 Gap       : "
        f"{mean_f1_gap:.4f}"
    )

    # ============================================================
    # 7. OOF CONFUSION MATRIX
    # ============================================================

    classes = sorted(
        y.unique()
    )

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{c}"
            for c in classes
        ],
        columns=[
            f"Pred_{c}"
            for c in classes
        ]
    )

    print()
    print("=" * 75)
    print("OOF CONFUSION MATRIX")
    print("=" * 75)

    print(
        cm_df.to_string()
    )

    # ============================================================
    # 8. OOF CLASSIFICATION REPORT
    # ============================================================

    print()
    print("=" * 75)
    print("OOF CLASSIFICATION REPORT")
    print("=" * 75)

    print(
        classification_report(
            all_true,
            all_pred,
            zero_division=0
        )
    )

    # ============================================================
    # 9. COMPARE AGAINST STEP 33 HELD-OUT TEST
    # ============================================================

    print()
    print("=" * 75)
    print("STEP 33 vs STEP 33A")
    print("=" * 75)

    step33_cv_accuracy = 1.0000
    step33_cv_macro_f1 = 1.0000

    step33_test_accuracy = 0.9285714285714286
    step33_test_macro_f1 = 0.8367346938775511

    print()
    print("STEP 33")
    print(
        f"CV Accuracy     : "
        f"{step33_cv_accuracy:.4f}"
    )

    print(
        f"CV Macro-F1     : "
        f"{step33_cv_macro_f1:.4f}"
    )

    print(
        f"Test Accuracy   : "
        f"{step33_test_accuracy:.4f}"
    )

    print(
        f"Test Macro-F1   : "
        f"{step33_test_macro_f1:.4f}"
    )

    print()
    print("STEP 33A — STRICT CV")

    print(
        f"CV Accuracy     : "
        f"{oof_accuracy:.4f}"
    )

    print(
        f"CV Macro-F1     : "
        f"{oof_macro_f1:.4f}"
    )

    print()

    print(
        "Step 33 original CV → strict CV accuracy change:",
        f"{step33_cv_accuracy - oof_accuracy:.4f}"
    )

    print(
        "Step 33 original CV → strict CV Macro-F1 change:",
        f"{step33_cv_macro_f1 - oof_macro_f1:.4f}"
    )

    # ============================================================
    # 10. SIMPLE DIAGNOSTIC
    # ============================================================

    print()
    print("=" * 75)
    print("GENERALIZATION DIAGNOSIS")
    print("=" * 75)

    if (
        oof_accuracy >= 0.90
        and mean_accuracy_gap <= 0.10
    ):

        print(
            "DIAGNOSIS: Good generalization evidence."
        )

        print(
            "The model retains strong validation performance "
            "after fold-wise feature selection."
        )

    elif (
        oof_accuracy >= 0.80
        and mean_accuracy_gap <= 0.20
    ):

        print(
            "DIAGNOSIS: Moderate generalization."
        )

        print(
            "Some overfitting may exist, "
            "but the model still learns useful patterns."
        )

    else:

        print(
            "DIAGNOSIS: Generalization concern."
        )

        print(
            "Further investigation of overfitting, "
            "feature representation, and dataset coverage is needed."
        )

    print()

    if mean_train_accuracy > 0.98:
        print(
            "Training capacity check: "
            "ExtraTrees fits the training folds very strongly."
        )

    if (
        mean_train_accuracy - mean_val_accuracy
        > 0.15
    ):

        print(
            "WARNING: Large train/validation gap."
        )

        print(
            "This is evidence of overfitting."
        )

    else:

        print(
            "No large train/validation gap detected."
        )

    if (
        oof_accuracy < 0.70
        and mean_train_accuracy < 0.80
    ):

        print(
            "Possible underfitting detected."
        )

    else:

        print(
            "No evidence of classical underfitting."
        )

    # ============================================================
    # 11. SAVE RESULTS
    # ============================================================

    fold_df.to_csv(
        OUTPUT_FOLDS,
        index=False
    )

    oof_df = pd.DataFrame({
        RUN_COL: train.iloc[
            cv.split(X, y).__next__()[1]
        ][RUN_COL]
        if False else
        np.arange(len(all_true)),
        "true_label": all_true,
        "predicted_label": all_pred,
    })

    oof_df.to_csv(
        OUTPUT_OOF,
        index=False
    )

    cm_df.to_csv(
        OUTPUT_CONFUSION
    )

    summary = pd.DataFrame([{

        "strict_cv_mean_validation_accuracy":
            mean_val_accuracy,

        "strict_cv_accuracy_std":
            std_val_accuracy,

        "strict_cv_oof_accuracy":
            oof_accuracy,

        "strict_cv_mean_validation_macro_f1":
            mean_val_f1,

        "strict_cv_macro_f1_std":
            std_val_f1,

        "strict_cv_oof_macro_f1":
            oof_macro_f1,

        "mean_training_accuracy":
            mean_train_accuracy,

        "mean_training_macro_f1":
            mean_train_f1,

        "mean_accuracy_gap":
            mean_accuracy_gap,

        "mean_macro_f1_gap":
            mean_f1_gap,

        "step33_independent_test_accuracy":
            step33_test_accuracy,

        "step33_independent_test_macro_f1":
            step33_test_macro_f1,
    }])

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print()
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_FOLDS
    )

    print(
        OUTPUT_OOF
    )

    print(
        OUTPUT_CONFUSION
    )

    print()
    print("=" * 75)
    print("STEP 33A COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
