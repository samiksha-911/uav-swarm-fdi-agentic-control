#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


# ================================================================
# STEP 32 — SEAD + FDI LEAKAGE-SAFE FEATURE AUDIT
# ================================================================

TRAIN_FILE = "step31_sead_fdi_train_run_level.csv"
TEST_FILE = "step31_sead_fdi_test_run_level.csv"

OUT_FEATURES = "step32_sead_fdi_feature_audit.csv"
OUT_SUMMARY = "step32_sead_fdi_feature_audit_summary.csv"
OUT_REDUCED_TRAIN = "step32_sead_fdi_train_reduced.csv"
OUT_REDUCED_TEST = "step32_sead_fdi_test_reduced.csv"

LABEL_COL = "fdi_failure_type"

TOP_K = [25, 50, 100, 200, 300, 500, 750, 1000]


def eta_squared(X, y):
    """
    Correct one-way ANOVA eta-squared for each feature.

    eta^2 = SS_between / SS_total

    Calculated using TRAINING DATA ONLY.
    """
    y_codes = pd.Series(y).astype("category").cat.codes.to_numpy()

    X = np.asarray(X, dtype=float)

    global_mean = np.nanmean(X, axis=0)

    ss_total = np.nansum((X - global_mean) ** 2, axis=0)
    ss_between = np.zeros(X.shape[1], dtype=float)

    for cls in np.unique(y_codes):
        mask = y_codes == cls
        n = np.sum(mask)

        if n == 0:
            continue

        cls_mean = np.nanmean(X[mask], axis=0)
        ss_between += n * (cls_mean - global_mean) ** 2

    result = np.divide(
        ss_between,
        ss_total,
        out=np.zeros_like(ss_between),
        where=ss_total > 0,
    )

    result[~np.isfinite(result)] = 0.0
    result = np.clip(result, 0.0, 1.0)

    return result


def main():

    print("=" * 70)
    print("STEP 32 — SEAD + FDI LEAKAGE-SAFE FEATURE AUDIT")
    print("=" * 70)
    print()
    print("RUN-LEVEL INTEGRATION")
    print("SEAD + FDI")
    print("TRAIN ONLY FOR FEATURE DECISIONS")
    print("FDI TEST REMAINS COMPLETELY HELD OUT")
    print("NO ORIGINAL DATA MODIFICATION")
    print("NO YOLO")
    print()

    # ------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------
    print("1. LOADING")
    print("-" * 70)

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print(f"Train rows: {len(train)}")
    print(f"Test rows:  {len(test)}")
    print(f"Train columns: {len(train.columns)}")
    print(f"Test columns:  {len(test.columns)}")

    if LABEL_COL not in train.columns:
        raise ValueError(
            f"Missing training label column: {LABEL_COL}"
        )

    if LABEL_COL not in test.columns:
        raise ValueError(
            f"Missing test label column: {LABEL_COL}"
        )

    # ------------------------------------------------------------
    # 2. RUN CHECK
    # ------------------------------------------------------------
    print()
    print("2. RUN-LEVEL CHECK")
    print("-" * 70)

    if "run_key" in train.columns:
        print("Train unique runs:", train["run_key"].nunique())

    if "run_key" in test.columns:
        print("Test unique runs: ", test["run_key"].nunique())

    if "run_key" in train.columns and "run_key" in test.columns:

        overlap = set(train["run_key"]) & set(test["run_key"])

        print("Train/test run overlap:", len(overlap))

        if overlap:
            raise RuntimeError(
                "DATA LEAKAGE: train/test run overlap detected."
            )

    # ------------------------------------------------------------
    # 3. LABEL CHECK
    # ------------------------------------------------------------
    print()
    print("3. LABEL DISTRIBUTION")
    print("-" * 70)

    print(train[LABEL_COL].value_counts().sort_index())

    # ------------------------------------------------------------
    # 4. IDENTIFY FEATURES
    # ------------------------------------------------------------
    print()
    print("4. FEATURE IDENTIFICATION")
    print("-" * 70)

    metadata_cols = {
        "run_key",
        "label",
        "fdi_failure_type",
        "available_required_streams",
        "total_generated_features",
    }

    metadata_cols = [
        c for c in metadata_cols
        if c in train.columns
    ]

    candidate_features = [
        c for c in train.columns
        if c not in metadata_cols
    ]

    # Only columns present in both datasets
    candidate_features = [
        c for c in candidate_features
        if c in test.columns
    ]

    numeric_features = []

    for c in candidate_features:
        if pd.api.types.is_numeric_dtype(train[c]):
            numeric_features.append(c)

    print("Candidate features:", len(candidate_features))
    print("Numeric features:", len(numeric_features))

    # ------------------------------------------------------------
    # 5. REMOVE CONSTANT FEATURES
    # ------------------------------------------------------------
    print()
    print("5. TRAIN-ONLY CONSTANT FEATURE REMOVAL")
    print("-" * 70)

    constant_features = []
    nonconstant_features = []

    for c in numeric_features:

        n_unique = train[c].nunique(dropna=False)

        if n_unique <= 1:
            constant_features.append(c)
        else:
            nonconstant_features.append(c)

    print("Constant features:", len(constant_features))
    print("Non-constant features:", len(nonconstant_features))

    # ------------------------------------------------------------
    # 6. MISSING / INFINITE CHECK
    # ------------------------------------------------------------
    print()
    print("6. NUMERIC QUALITY CHECK")
    print("-" * 70)

    X_train = train[nonconstant_features].copy()
    X_test = test[nonconstant_features].copy()

    train_inf = np.isinf(
        X_train.select_dtypes(include=np.number).to_numpy()
    ).sum()

    test_inf = np.isinf(
        X_test.select_dtypes(include=np.number).to_numpy()
    ).sum()

    train_missing = X_train.isna().sum().sum()
    test_missing = X_test.isna().sum().sum()

    print("Train missing:", int(train_missing))
    print("Test missing:", int(test_missing))
    print("Train infinite:", int(train_inf))
    print("Test infinite:", int(test_inf))

    # Replace infinities with NaN.
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Median imputation is FIT ON TRAIN ONLY.
    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Any column that remains unusable is dropped.
    usable_features = []

    for c in X_train.columns:

        if X_train[c].isna().all():
            continue

        if X_train[c].nunique() <= 1:
            continue

        usable_features.append(c)

    X_train = X_train[usable_features]
    X_test = X_test[usable_features]

    print("Usable training features:", len(usable_features))

    # ------------------------------------------------------------
    # 7. ETA-SQUARED
    # ------------------------------------------------------------
    print()
    print("7. TRAIN-ONLY ETA-SQUARED RANKING")
    print("-" * 70)

    y_train = train[LABEL_COL].astype(str)

    eta = eta_squared(
        X_train.to_numpy(dtype=float),
        y_train,
    )

    ranking = pd.DataFrame({
        "feature": usable_features,
        "eta_squared": eta,
    })

    ranking = ranking.sort_values(
        "eta_squared",
        ascending=False
    ).reset_index(drop=True)

    ranking["rank"] = np.arange(1, len(ranking) + 1)

    print("Features ranked:", len(ranking))
    print()
    print(ranking.head(30).to_string(index=False))

    # ------------------------------------------------------------
    # 8. HIGH-RISK PROXY AUDIT
    # ------------------------------------------------------------
    print()
    print("8. STATUS / LABEL-PROXY AUDIT")
    print("-" * 70)

    proxy_keywords = [
        "failure",
        "fault",
        "label",
        "class",
        "target",
        "status",
        "health",
        "err",
        "error",
        "fail",
        "flag",
    ]

    ranking["proxy_keyword"] = ranking["feature"].str.lower().apply(
        lambda x: any(k in x for k in proxy_keywords)
    )

    proxy_count = int(ranking["proxy_keyword"].sum())

    print("Potential status/error/proxy-related features:", proxy_count)

    print()
    print(
        ranking[
            ranking["proxy_keyword"]
        ].head(50).to_string(index=False)
    )

    # ------------------------------------------------------------
    # 9. FEATURE COUNT SUMMARY
    # ------------------------------------------------------------
    print()
    print("9. FEATURE COUNT SUMMARY")
    print("-" * 70)

    available_k = [
        k for k in TOP_K
        if k <= len(ranking)
    ]

    summary_rows = []

    for k in available_k:

        subset = ranking.head(k)

        summary_rows.append({
            "top_k": k,
            "features_retained": len(subset),
            "eta_min": subset["eta_squared"].min(),
            "eta_median": subset["eta_squared"].median(),
            "eta_max": subset["eta_squared"].max(),
            "proxy_keyword_features": int(
                subset["proxy_keyword"].sum()
            ),
        })

    summary_df = pd.DataFrame(summary_rows)

    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------
    # 10. STREAM DISTRIBUTION
    # ------------------------------------------------------------
    print()
    print("10. TOP-100 STREAM DISTRIBUTION")
    print("-" * 70)

    top100 = ranking.head(
        min(100, len(ranking))
    )

    def stream_name(feature):

        if "__" in feature:
            return feature.split("__", 1)[0]

        return feature.split("_", 1)[0]

    top100_streams = (
        top100["feature"]
        .apply(stream_name)
        .value_counts()
    )

    print(top100_streams.to_string())

    # ------------------------------------------------------------
    # 11. SAVE FEATURE AUDIT
    # ------------------------------------------------------------
    print()
    print("11. SAVING FEATURE AUDIT")
    print("-" * 70)

    ranking.to_csv(
        OUT_FEATURES,
        index=False
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False
    )

    print("Saved:", OUT_FEATURES)
    print("Saved:", OUT_SUMMARY)

    # ------------------------------------------------------------
    # 12. BUILD REDUCED TABLE
    # ------------------------------------------------------------
    print()
    print("12. BUILDING REDUCED TABLES")
    print("-" * 70)

    # Use the experimentally strongest count from Step 29.
    # This is NOT evaluated on FDI test here.
    selected_k = min(
        750,
        len(ranking)
    )

    selected_features = ranking.head(
        selected_k
    )["feature"].tolist()

    train_reduced = train[
        ["run_key", LABEL_COL] +
        selected_features
    ].copy()

    test_reduced = test[
        ["run_key", LABEL_COL] +
        selected_features
    ].copy()

    print("Selected features:", selected_k)
    print("Reduced train shape:", train_reduced.shape)
    print("Reduced test shape:", test_reduced.shape)

    # ------------------------------------------------------------
    # 13. SAVE REDUCED TABLES
    # ------------------------------------------------------------
    print()
    print("13. SAVING REDUCED TABLES")
    print("-" * 70)

    train_reduced.to_csv(
        OUT_REDUCED_TRAIN,
        index=False
    )

    test_reduced.to_csv(
        OUT_REDUCED_TEST,
        index=False
    )

    print("Saved:", OUT_REDUCED_TRAIN)
    print("Saved:", OUT_REDUCED_TEST)

    # ------------------------------------------------------------
    # 14. QUICK TRAIN-ONLY MODEL SANITY CHECK
    # ------------------------------------------------------------
    print()
    print("14. TRAIN-ONLY MODEL SANITY CHECK")
    print("-" * 70)

    X = train_reduced[selected_features]
    y = train_reduced[LABEL_COL].astype(str)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = {

        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    C=1.0,
                    solver="lbfgs"
                )
            )
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

    model_rows = []

    for name, model in models.items():

        print()
        print(name)

        scores = cross_validate(
            model,
            X,
            y,
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

        print(
            f"  Accuracy: "
            f"{acc.mean():.4f} ± {acc.std():.4f}"
        )

        print(
            f"  Macro-F1: "
            f"{f1.mean():.4f} ± {f1.std():.4f}"
        )

        model_rows.append({
            "model": name,
            "accuracy_mean": acc.mean(),
            "accuracy_std": acc.std(),
            "macro_f1_mean": f1.mean(),
            "macro_f1_std": f1.std(),
        })

    model_df = pd.DataFrame(model_rows)

    model_df.to_csv(
        "step32_sead_fdi_model_sanity.csv",
        index=False
    )

    print()
    print("Saved: step32_sead_fdi_model_sanity.csv")

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 32 COMPLETE")
    print("=" * 70)

    print()
    print("Train runs:", len(train))
    print("Test runs:", len(test))
    print("Constant features removed:", len(constant_features))
    print("Usable features:", len(usable_features))
    print("Selected features:", selected_k)

    print()
    print("IMPORTANT:")
    print("FDI test runs were NOT used for feature ranking.")
    print("FDI test runs were NOT used for model selection.")
    print("Original files were NOT modified.")
    print("No sample-level merge was performed.")
    print("No YOLO was used.")

    print()
    print("NEXT:")
    print("Step 33 should perform final model selection")
    print("using training-run CV, followed by ONE evaluation")
    print("on the untouched 14-run FDI test set.")

    print("=" * 70)


if __name__ == "__main__":
    main()
