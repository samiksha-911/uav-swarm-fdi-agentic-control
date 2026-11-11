import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score


INPUT = "step27_sead_nonconstant_features.csv"
OUTPUT = "step28_sead_baseline_model_results.csv"

RANDOM_STATE = 42
N_SPLITS = 5


def main():

    print("=" * 70)
    print("STEP 28 — SEAD BASELINE MODEL EVALUATION")
    print("=" * 70)
    print()
    print("SEAD ONLY")
    print("NO FDI MERGE")
    print("NO TEST MODIFICATION")
    print("NO MODEL TRAINING ON FDI")
    print()

    # ------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------
    print("1. LOADING REDUCED SEAD TABLE")
    print("-" * 70)

    df = pd.read_csv(INPUT)

    if "label" not in df.columns:
        raise RuntimeError("Expected 'label' column not found.")

    if "run_key" not in df.columns:
        raise RuntimeError("Expected 'run_key' column not found.")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ------------------------------------------------------------
    # 2. IDENTIFY FEATURES
    # ------------------------------------------------------------
    metadata = {
        "run_key",
        "label",
        "available_required_streams",
        "total_generated_features",
    }

    feature_cols = [c for c in df.columns if c not in metadata]

    X = df[feature_cols].copy()
    y = df["label"].copy()

    print()
    print("2. FEATURE SET")
    print("-" * 70)
    print(f"Feature columns: {len(feature_cols)}")

    # ------------------------------------------------------------
    # 3. VALIDATION
    # ------------------------------------------------------------
    if X.shape[1] == 0:
        raise RuntimeError("No feature columns found.")

    if X.isna().any().any():
        print("Missing values detected; fold-local imputation will be used.")
    else:
        print("Missing feature values: 0")

    if np.isinf(X.select_dtypes(include=[np.number])).any().any():
        raise RuntimeError("Infinite feature values detected.")

    print()
    print("3. CLASS DISTRIBUTION")
    print("-" * 70)
    print(y.value_counts().sort_index().to_string())

    # ------------------------------------------------------------
    # 4. CROSS VALIDATION
    # ------------------------------------------------------------
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "macro_f1": make_scorer(f1_score, average="macro"),
    }

    # ------------------------------------------------------------
    # 5. MODELS
    # ------------------------------------------------------------
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    multi_class="auto",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]),
    }

    # ------------------------------------------------------------
    # 6. EVALUATION
    # ------------------------------------------------------------
    print()
    print("4. STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("-" * 70)

    results = []

    for name, model in models.items():

        print(f"\n{name}")

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
        )

        acc = scores["test_accuracy"]
        f1 = scores["test_macro_f1"]

        for fold in range(N_SPLITS):
            print(
                f"  Fold {fold + 1}: "
                f"accuracy={acc[fold]:.4f}, "
                f"macro-F1={f1[fold]:.4f}"
            )

            results.append({
                "model": name,
                "fold": fold + 1,
                "accuracy": acc[fold],
                "macro_f1": f1[fold],
            })

        print(
            f"  Mean accuracy: {acc.mean():.4f} ± {acc.std():.4f}"
        )
        print(
            f"  Mean macro-F1: {f1.mean():.4f} ± {f1.std():.4f}"
        )

    result_df = pd.DataFrame(results)

    # ------------------------------------------------------------
    # 7. SUMMARY
    # ------------------------------------------------------------
    print()
    print("5. MODEL SUMMARY")
    print("-" * 70)

    summary = (
        result_df
        .groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    # ------------------------------------------------------------
    # 8. SAVE
    # ------------------------------------------------------------
    result_df.to_csv(OUTPUT, index=False)

    print()
    print("6. SAVING RESULTS")
    print("-" * 70)
    print(f"Saved: {OUTPUT}")

    print()
    print("=" * 70)
    print("STEP 28 COMPLETE")
    print("=" * 70)
    print()
    print("SEAD baseline classification evaluated using")
    print("stratified 5-fold run-level cross-validation.")
    print()
    print("No FDI data was merged.")
    print("fdi_train.csv was not modified.")
    print("fdi_test.csv was not modified.")
    print("Original SEAD feature tables were not modified.")


if __name__ == "__main__":
    main()

