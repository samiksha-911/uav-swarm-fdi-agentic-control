import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


INPUT = "step27_sead_nonconstant_features.csv"
OUTPUT = "step30_sead_final_cv_results.csv"

TOP_K = 750
RANDOM_STATE = 42
N_SPLITS = 5


def main():

    print("=" * 70)
    print("STEP 30 — SEAD FINAL LEAKAGE-SAFE CROSS-VALIDATION")
    print("=" * 70)
    print()
    print("SEAD ONLY")
    print("NO FDI MERGE")
    print("NO TEST MODIFICATION")
    print("NO FDI MODEL TRAINING")
    print()

    df = pd.read_csv(INPUT)

    metadata = {
        "run_key",
        "label",
        "available_required_streams",
        "total_generated_features",
    }

    feature_cols = [c for c in df.columns if c not in metadata]

    X = df[feature_cols]
    y = df["label"]

    print("1. DATA")
    print("-" * 70)
    print(f"Runs: {len(df)}")
    print(f"Available features: {len(feature_cols)}")
    print(f"Classes: {y.nunique()}")
    print()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []
    all_true = []
    all_pred = []

    print("2. LEAKAGE-SAFE 5-FOLD CV")
    print("-" * 70)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),

            # Feature selection is FIT ONLY on the training fold.
            ("selector", SelectKBest(
                score_func=f_classif,
                k=min(TOP_K, X_train.shape[1])
            )),

            ("model", RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ])

        pipeline.fit(X_train, y_train)

        pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        print(
            f"Fold {fold}: "
            f"accuracy={acc:.4f} | "
            f"macro-F1={f1:.4f}"
        )

        fold_results.append({
            "fold": fold,
            "accuracy": acc,
            "macro_f1": f1,
            "train_runs": len(train_idx),
            "test_runs": len(test_idx),
            "features_selected": TOP_K
        })

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

    results = pd.DataFrame(fold_results)

    print()
    print("3. FINAL CV SUMMARY")
    print("-" * 70)

    print(
        f"Accuracy:  "
        f"{results['accuracy'].mean():.4f} ± "
        f"{results['accuracy'].std():.4f}"
    )

    print(
        f"Macro-F1:   "
        f"{results['macro_f1'].mean():.4f} ± "
        f"{results['macro_f1'].std():.4f}"
    )

    print()
    print("4. OUT-OF-FOLD CONFUSION MATRIX")
    print("-" * 70)

    labels = sorted(y.unique())

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print(cm_df.to_string())

    print()
    print("5. OUT-OF-FOLD METRICS")
    print("-" * 70)

    oof_acc = accuracy_score(all_true, all_pred)
    oof_f1 = f1_score(
        all_true,
        all_pred,
        average="macro"
    )

    print(f"OOF accuracy:  {oof_acc:.4f}")
    print(f"OOF macro-F1:  {oof_f1:.4f}")

    results.to_csv(OUTPUT, index=False)

    print()
    print("6. SAVING")
    print("-" * 70)
    print(f"Saved: {OUTPUT}")

    print()
    print("=" * 70)
    print("STEP 30 COMPLETE")
    print("=" * 70)
    print()
    print("Feature selection was performed independently inside")
    print("each training fold.")
    print()
    print("No FDI data was merged.")
    print("fdi_train.csv was NOT modified.")
    print("fdi_test.csv was NOT modified.")
    print("Original SEAD feature tables were NOT modified.")


if __name__ == "__main__":
    main()

