#!/usr/bin/env python3

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier


# ============================================================
# STEP 35 — EXPORT FINAL TRAINED EXTRATREES MODEL
# ============================================================

TRAIN_FILE = "step32_sead_fdi_train_reduced.csv"
MODEL_FILE = "uav_fdi_extratrees_model.joblib"

LABEL_COL = "fdi_failure_type"
RUN_COL = "run_key"


def main():

    print("=" * 70)
    print("STEP 35 — EXPORT FINAL TRAINED EXTRATREES MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD TRAINING DATA
    # --------------------------------------------------------

    train = pd.read_csv(TRAIN_FILE)

    print()
    print("Training shape:", train.shape)

    # --------------------------------------------------------
    # 2. IDENTIFY THE 750 FEATURES
    # --------------------------------------------------------

    metadata = {
        RUN_COL,
        LABEL_COL,
        "label",
        "available_required_streams",
        "total_generated_features",
    }

    features = [
        c for c in train.columns
        if c not in metadata
    ]

    print("Features found:", len(features))

    if len(features) != 750:
        raise RuntimeError(
            f"Expected 750 features, found {len(features)}."
        )

    X = train[features].copy()
    y = train[LABEL_COL].astype(str)

    # --------------------------------------------------------
    # 3. TRAIN-ONLY PREPROCESSING
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    medians = X.median()

    X = X.fillna(medians)

    # --------------------------------------------------------
    # 4. TRAIN FINAL EXTRATREES MODEL
    # --------------------------------------------------------

    print()
    print("Training final ExtraTrees model...")

    model = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        X,
        y
    )

    print("Training complete.")

    # --------------------------------------------------------
    # 5. CREATE ONE MODEL BUNDLE
    # --------------------------------------------------------

    bundle = {
        "model": model,
        "features": features,
        "medians": medians.to_dict(),
        "label_column": LABEL_COL,
        "run_column": RUN_COL,
        "n_features": len(features),
        "model_name": "ExtraTreesClassifier",
        "n_estimators": 500,
        "random_state": 42,
        "classes": model.classes_.tolist(),
        "training_runs": len(train),
        "description":
            "Final SEAD + FDI run-level ExtraTrees model "
            "from Step 33."
    }

    # --------------------------------------------------------
    # 6. SAVE
    # --------------------------------------------------------

    joblib.dump(
        bundle,
        MODEL_FILE
    )

    size_mb = (
        os.path.getsize(MODEL_FILE)
        / (1024 * 1024)
    )

    print()
    print("=" * 70)
    print("MODEL EXPORTED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("File:", MODEL_FILE)
    print("Model:", bundle["model_name"])
    print("Training runs:", bundle["training_runs"])
    print("Features:", bundle["n_features"])
    print("Saved feature names:", len(bundle["features"]))
    print("Saved medians:", len(bundle["medians"]))
    print("Classes:", bundle["classes"])
    print(f"File size: {size_mb:.2f} MB")
    print()
    print("STEP 35 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
