import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score


INPUT_FEATURES = "step27_sead_ranked_features.csv"
INPUT_TABLE = "step27_sead_nonconstant_features.csv"
OUTPUT = "step29_sead_feature_count_experiment.csv"

RANDOM_STATE = 42
N_SPLITS = 5

TOP_K_VALUES = [25, 50, 100, 200, 300, 500, 750, 1000, 1154]


def main():

    print("=" * 70)
    print("STEP 29 — SEAD FEATURE COUNT EXPERIMENT")
    print("=" * 70)
    print()
    print("SEAD ONLY")
    print("NO FDI MERGE")
    print("NO TEST MODIFICATION")
    print("NO MODEL TRAINING ON FDI")
    print()

    print("1. LOADING")
    print("-" * 70)

    df = pd.read_csv(INPUT_TABLE)
    ranked = pd.read_csv(INPUT_FEATURES)

    if "label" not in df.columns:
        raise RuntimeError("label column missing.")

    if "feature" not in ranked.columns:
        raise RuntimeError("feature column missing from ranking file.")

    ranked_features = ranked["feature"].tolist()

    available = [c for c in ranked_features if c in df.columns]

    if len(available) != len(ranked_features):
        print(
            f"Warning: {len(ranked_features) - len(available)} "
            "ranked features were not found in table."
        )

    y = df["label"]

    print(f"Rows: {len(df)}")
    print(f"Ranked features available: {len(available)}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "macro_f1": make_scorer(f1_score, average="macro"),
    }

    results = []

    print()
    print("2. RANDOM FOREST FEATURE COUNT EXPERIMENT")
    print("-" * 70)

    for k in TOP_K_VALUES:

        k = min(k, len(available))
        features = available[:k]
        X = df[features]

        model = Pipeline([
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
        ])

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
        )

        acc_mean = scores["test_accuracy"].mean()
        acc_std = scores["test_accuracy"].std()

        f1_mean = scores["test_macro_f1"].mean()
        f1_std = scores["test_macro_f1"].std()

        print(
            f"Top-{k:4d}: "
            f"accuracy={acc_mean:.4f} ± {acc_std:.4f} | "
            f"macro-F1={f1_mean:.4f} ± {f1_std:.4f}"
        )

        results.append({
            "top_k": k,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
        })

    result_df = pd.DataFrame(results)

    print()
    print("3. RESULTS")
    print("-" * 70)
    print(result_df.to_string(index=False))

    best = result_df.loc[
        result_df["macro_f1_mean"].idxmax()
    ]

    print()
    print("4. BEST FEATURE COUNT")
    print("-" * 70)
    print(
        f"Best top-K by macro-F1: {int(best['top_k'])}"
    )
    print(
        f"Accuracy: {best['accuracy_mean']:.4f} "
        f"± {best['accuracy_std']:.4f}"
    )
    print(
        f"Macro-F1: {best['macro_f1_mean']:.4f} "
        f"± {best['macro_f1_std']:.4f}"
    )

    result_df.to_csv(OUTPUT, index=False)

    print()
    print("5. SAVING")
    print("-" * 70)
    print(f"Saved: {OUTPUT}")

    print()
    print("=" * 70)
    print("STEP 29 COMPLETE")
    print("=" * 70)
    print()
    print("Original SEAD tables were NOT modified.")
    print("No FDI data was merged.")
    print("fdi_train.csv was NOT modified.")
    print("fdi_test.csv was NOT modified.")


if __name__ == "__main__":
    main()

