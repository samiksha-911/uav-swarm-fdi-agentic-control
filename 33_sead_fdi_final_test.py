#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ================================================================
# STEP 33 — SEAD + FDI FINAL HELD-OUT TEST
# ================================================================

TRAIN_FILE = "step32_sead_fdi_train_reduced.csv"
TEST_FILE = "step32_sead_fdi_test_reduced.csv"

LABEL_COL = "fdi_failure_type"
RUN_COL = "run_key"

PREDICTIONS_FILE = "step33_sead_fdi_test_predictions.csv"
RESULTS_FILE = "step33_sead_fdi_final_results.csv"
CONFUSION_FILE = "step33_sead_fdi_confusion_matrix.csv"


def main():

    print("=" * 70)
    print("STEP 33 — SEAD + FDI FINAL HELD-OUT TEST")
    print("=" * 70)
    print()
    print("FINAL RUN-LEVEL EVALUATION")
    print("56 FDI TRAIN RUNS")
    print("14 FDI TEST RUNS")
    print("TEST SET USED ONLY ONCE FOR FINAL EVALUATION")
    print("NO ORIGINAL DATA MODIFICATION")
    print("NO YOLO")
    print()

    # ------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------
    print("1. LOADING REDUCED DATA")
    print("-" * 70)

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print("Train shape:", train.shape)
    print("Test shape: ", test.shape)

    if LABEL_COL not in train.columns:
        raise ValueError(f"Missing {LABEL_COL} in train")

    if LABEL_COL not in test.columns:
        raise ValueError(f"Missing {LABEL_COL} in test")

    # ------------------------------------------------------------
    # 2. RUN-LEVEL LEAKAGE CHECK
    # ------------------------------------------------------------
    print()
    print("2. TRAIN / TEST RUN LEAKAGE CHECK")
    print("-" * 70)

    train_runs = set(train[RUN_COL])
    test_runs = set(test[RUN_COL])

    overlap = train_runs & test_runs

    print("Train runs:", len(train_runs))
    print("Test runs: ", len(test_runs))
    print("Overlap:   ", len(overlap))

    if overlap:
        raise RuntimeError(
            "STOP: train/test run overlap detected."
        )

    # ------------------------------------------------------------
    # 3. LABEL DISTRIBUTION
    # ------------------------------------------------------------
    print()
    print("3. TRAIN LABELS")
    print("-" * 70)

    print(train[LABEL_COL].value_counts().sort_index())

    print()
    print("TEST LABELS")
    print("-" * 70)

    print(test[LABEL_COL].value_counts().sort_index())

    # ------------------------------------------------------------
    # 4. FEATURES
    # ------------------------------------------------------------
    metadata = {
        RUN_COL,
        LABEL_COL,
        "label",
        "available_required_streams",
        "total_generated_features",
    }

    features = [
        c for c in train.columns
        if c not in metadata and c in test.columns
    ]

    numeric_features = [
        c for c in features
        if pd.api.types.is_numeric_dtype(train[c])
    ]

    print()
    print("4. FEATURE SET")
    print("-" * 70)
    print("Features:", len(numeric_features))

    X_train = train[numeric_features].copy()
    X_test = test[numeric_features].copy()

    y_train = train[LABEL_COL].astype(str)
    y_test = test[LABEL_COL].astype(str)

    # ------------------------------------------------------------
    # 5. SANITIZE USING TRAIN ONLY
    # ------------------------------------------------------------
    print()
    print("5. TRAIN-ONLY PREPROCESSING")
    print("-" * 70)

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # ------------------------------------------------------------
    # 6. CV MODEL SELECTION
    # ------------------------------------------------------------
    print()
    print("6. TRAINING-RUN CROSS-VALIDATION")
    print("-" * 70)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
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
                "model",
                LogisticRegression(
                    max_iter=5000,
                    C=1.0,
                    solver="lbfgs"
                )
            ),
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
    }

    cv_rows = []

    for name, model in models.items():

        print()
        print(name)

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=[
                "accuracy",
                "f1_macro"
            ],
            n_jobs=1,
            error_score="raise"
        )

        acc = scores["test_accuracy"]
        f1 = scores["test_f1_macro"]

        acc_mean = acc.mean()
        acc_std = acc.std()

        f1_mean = f1.mean()
        f1_std = f1.std()

        print(
            f"  Accuracy : "
            f"{acc_mean:.4f} ± {acc_std:.4f}"
        )

        print(
            f"  Macro-F1 : "
            f"{f1_mean:.4f} ± {f1_std:.4f}"
        )

        cv_rows.append({
            "model": name,
            "cv_accuracy_mean": acc_mean,
            "cv_accuracy_std": acc_std,
            "cv_macro_f1_mean": f1_mean,
            "cv_macro_f1_std": f1_std,
        })

    cv_results = pd.DataFrame(cv_rows)

    # Select by macro-F1, then accuracy.
    cv_results = cv_results.sort_values(
        [
            "cv_macro_f1_mean",
            "cv_accuracy_mean"
        ],
        ascending=False
    ).reset_index(drop=True)

    selected_model_name = cv_results.iloc[0]["model"]

    print()
    print("Selected model:", selected_model_name)

    print()
    print(cv_results.to_string(index=False))

    # ------------------------------------------------------------
    # 7. FIT FINAL MODEL ON ALL 56 TRAIN RUNS
    # ------------------------------------------------------------
    print()
    print("7. FINAL MODEL FIT")
    print("-" * 70)

    final_model = models[selected_model_name]

    final_model.fit(
        X_train,
        y_train
    )

    print(
        "Final model trained on",
        len(train),
        "training runs."
    )

    # ------------------------------------------------------------
    # 8. FINAL TEST PREDICTION
    # ------------------------------------------------------------
    print()
    print("8. FINAL EVALUATION ON 14 HELD-OUT TEST RUNS")
    print("-" * 70)

    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print(f"Test accuracy : {accuracy:.4f}")
    print(f"Test macro-F1 : {macro_f1:.4f}")
    print(f"Test weighted-F1: {weighted_f1:.4f}")

    # ------------------------------------------------------------
    # 9. PER-CLASS REPORT
    # ------------------------------------------------------------
    print()
    print("9. PER-CLASS TEST METRICS")
    print("-" * 70)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    print(
        report_df.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ------------------------------------------------------------
    # 10. CONFUSION MATRIX
    # ------------------------------------------------------------
    print()
    print("10. TEST CONFUSION MATRIX")
    print("-" * 70)

    class_order = sorted(
        set(y_test) | set(y_pred)
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_order
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_order,
        columns=class_order
    )

    print(cm_df.to_string())

    # ------------------------------------------------------------
    # 11. TEST PREDICTIONS
    # ------------------------------------------------------------
    print()
    print("11. SAVING TEST PREDICTIONS")
    print("-" * 70)

    predictions = pd.DataFrame({
        RUN_COL: test[RUN_COL],
        "true_label": y_test,
        "predicted_label": y_pred,
        "correct": y_test.to_numpy() == y_pred,
    })

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    cm_df.to_csv(
        CONFUSION_FILE
    )

    # ------------------------------------------------------------
    # 12. FINAL RESULT TABLE
    # ------------------------------------------------------------
    result = pd.DataFrame([{
        "selected_model": selected_model_name,
        "train_runs": len(train),
        "test_runs": len(test),
        "features": len(numeric_features),
        "cv_accuracy_mean":
            cv_results.iloc[0]["cv_accuracy_mean"],
        "cv_accuracy_std":
            cv_results.iloc[0]["cv_accuracy_std"],
        "cv_macro_f1_mean":
            cv_results.iloc[0]["cv_macro_f1_mean"],
        "cv_macro_f1_std":
            cv_results.iloc[0]["cv_macro_f1_std"],
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
    }])

    result.to_csv(
        RESULTS_FILE,
        index=False
    )

    print("Saved:", PREDICTIONS_FILE)
    print("Saved:", CONFUSION_FILE)
    print("Saved:", RESULTS_FILE)

    # ------------------------------------------------------------
    # 13. FINAL SUMMARY
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 33 COMPLETE")
    print("=" * 70)

    print()
    print("FINAL MODEL:", selected_model_name)
    print()
    print(f"Training runs: {len(train)}")
    print(f"Test runs:     {len(test)}")
    print(f"Features:      {len(numeric_features)}")

    print()
    print("TRAINING CV")
    print(
        f"Accuracy : "
        f"{cv_results.iloc[0]['cv_accuracy_mean']:.4f} "
        f"± "
        f"{cv_results.iloc[0]['cv_accuracy_std']:.4f}"
    )

    print(
        f"Macro-F1 : "
        f"{cv_results.iloc[0]['cv_macro_f1_mean']:.4f} "
        f"± "
        f"{cv_results.iloc[0]['cv_macro_f1_std']:.4f}"
    )

    print()
    print("HELD-OUT FDI TEST")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")

    print()
    print("The 14 FDI test runs were kept completely")
    print("separate from feature/model selection.")
    print()
    print("Original files were NOT modified.")
    print("No sample-level train/test mixing occurred.")
    print("No YOLO was used.")

    print("=" * 70)


if __name__ == "__main__":
    main()
