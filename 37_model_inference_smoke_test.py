#!/usr/bin/env python3

import joblib
import pandas as pd
import numpy as np


MODEL_FILE = "uav_fdi_extratrees_model.joblib"
TEST_FILE = "step32_sead_fdi_test_reduced.csv"
OUTPUT_FILE = "step37_model_inference_test.csv"


def main():

    print("=" * 70)
    print("STEP 37 — FROZEN MODEL INFERENCE TEST")
    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # 1. LOAD MODEL BUNDLE
    # ------------------------------------------------------------

    print("1. LOADING MODEL")
    print("-" * 70)

    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]
    features = bundle["features"]
    medians = bundle["medians"]
    classes = list(bundle["classes"])

    print("Model:", type(model).__name__)
    print("Selected features:", len(features))
    print("Saved medians:", len(medians))
    print("Classes:", len(classes))

    if len(features) != 750:
        raise RuntimeError(
            f"Expected 750 features, found {len(features)}."
        )

    if len(medians) != 750:
        raise RuntimeError(
            f"Expected 750 medians, found {len(medians)}."
        )

    # ------------------------------------------------------------
    # 2. LOAD EXISTING REDUCED TEST DATA
    # ------------------------------------------------------------

    print()
    print("2. LOADING TEST DATA")
    print("-" * 70)

    df = pd.read_csv(TEST_FILE)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # ------------------------------------------------------------
    # 3. VERIFY EXACT FEATURE SET
    # ------------------------------------------------------------

    print()
    print("3. FEATURE CHECK")
    print("-" * 70)

    missing_features = [
        f for f in features
        if f not in df.columns
    ]

    if missing_features:
        print("Missing features:", len(missing_features))
        for f in missing_features[:20]:
            print(" ", f)

        raise RuntimeError(
            "Required model features are missing from test data."
        )

    print("All 750 model features are present.")

    # ------------------------------------------------------------
    # 4. BUILD MODEL INPUT IN EXACT ORDER
    # ------------------------------------------------------------

    X = df[features].copy()

    # Replace non-finite values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Apply the saved training medians.
    for feature in features:
        X[feature] = X[feature].fillna(medians[feature])

    # Final numeric conversion.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Any remaining missing values are replaced using the
    # corresponding saved training median.
    for feature in features:
        if X[feature].isna().any():
            X[feature] = X[feature].fillna(medians[feature])

    if X.isna().any().any():
        raise RuntimeError(
            "NaN values remain after preprocessing."
        )

    print("Inference matrix shape:", X.shape)

    # ------------------------------------------------------------
    # 5. PREDICT
    # ------------------------------------------------------------

    print()
    print("4. RUNNING INFERENCE")
    print("-" * 70)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    confidence = probabilities.max(axis=1)

    # ------------------------------------------------------------
    # 6. BUILD RESULT TABLE
    # ------------------------------------------------------------

    result = pd.DataFrame({
        "run_key": df["run_key"].astype(str),
        "predicted_fault": predictions,
        "confidence": confidence,
    })

    if "fdi_failure_type" in df.columns:
        result["actual_fault"] = df["fdi_failure_type"].astype(str)

    # Add per-class probabilities.
    for i, cls in enumerate(classes):
        safe_name = (
            str(cls)
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        result[f"prob_{safe_name}"] = probabilities[:, i]

    # ------------------------------------------------------------
    # 7. SUMMARY
    # ------------------------------------------------------------

    print()
    print("5. PREDICTION SUMMARY")
    print("-" * 70)

    print(
        result[
            ["run_key", "predicted_fault", "confidence"]
            + (
                ["actual_fault"]
                if "actual_fault" in result.columns
                else []
            )
        ].to_string(index=False)
    )

    if "actual_fault" in result.columns:

        accuracy = (
            result["predicted_fault"]
            == result["actual_fault"]
        ).mean()

        print()
        print(f"Accuracy on supplied reduced test table: {accuracy:.4f}")

    # ------------------------------------------------------------
    # 8. SAVE
    # ------------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Saved:", OUTPUT_FILE)

    print()
    print("=" * 70)
    print("STEP 37 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
