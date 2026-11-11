import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "step22_sead_run_features.csv"
OUTPUT_FILE = "step25_sead_feature_discrimination.csv"

LABEL_COL = "label"
RUN_KEY_COL = "run_key"

METADATA_COLS = [
    RUN_KEY_COL,
    LABEL_COL,
    "available_required_streams",
    "total_generated_features",
]

CLASS_ORDER = [
    "No Failure",
    "RC Failure",
    "GPS Failure",
    "Accelerometer Failure",
    "Gyro Failure",
    "Compass Failure",
    "Barometer Failure",
]


def eta_squared(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)

    valid = np.isfinite(values)
    values = values[valid]
    labels = labels[valid]

    if len(values) == 0:
        return np.nan

    grand_mean = np.mean(values)

    total_ss = np.sum((values - grand_mean) ** 2)

    if total_ss == 0:
        return 0.0

    between_ss = 0.0

    for cls in np.unique(labels):
        cls_values = values[labels == cls]

        if len(cls_values) == 0:
            continue

        cls_mean = np.mean(cls_values)

        between_ss += (
            len(cls_values)
            * (cls_mean - grand_mean) ** 2
        )

    return between_ss / total_ss


def class_mean_range(values, labels):
    means = []

    for cls in CLASS_ORDER:
        cls_values = values[labels == cls]

        if len(cls_values) == 0:
            continue

        means.append(np.mean(cls_values))

    if len(means) < 2:
        return np.nan

    return float(max(means) - min(means))


def class_mean_cv(values, labels):
    means = []

    for cls in CLASS_ORDER:
        cls_values = values[labels == cls]

        if len(cls_values) == 0:
            continue

        means.append(np.mean(cls_values))

    if len(means) < 2:
        return np.nan

    global_mean = np.mean(values)

    if abs(global_mean) < 1e-12:
        return np.nan

    return float(
        (max(means) - min(means))
        / abs(global_mean)
    )


def main():

    print("=" * 70)
    print("STEP 25 — SEAD FEATURE DISCRIMINATION ANALYSIS")
    print("=" * 70)

    print()
    print("SEAD ONLY")
    print("NO FDI MERGE")
    print("NO TEST MODIFICATION")
    print("NO MODEL TRAINING")

    # ================================================================
    # 1. LOAD
    # ================================================================

    print()
    print("1. LOADING SEAD FEATURE TABLE")
    print("-" * 70)

    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"Required file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    if LABEL_COL not in df.columns:
        raise RuntimeError(
            f"Expected '{LABEL_COL}' column not found."
        )

    if RUN_KEY_COL not in df.columns:
        raise RuntimeError(
            f"Expected '{RUN_KEY_COL}' column not found."
        )

    # ================================================================
    # 2. FEATURE IDENTIFICATION
    # ================================================================

    print()
    print("2. FEATURE IDENTIFICATION")
    print("-" * 70)

    missing_metadata = [
        c for c in METADATA_COLS
        if c not in df.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            f"Missing metadata columns: {missing_metadata}"
        )

    feature_cols = [
        c for c in df.columns
        if c not in METADATA_COLS
    ]

    print(f"Total feature columns: {len(feature_cols)}")

    # ================================================================
    # 3. CONSTANT FEATURE DETECTION
    # ================================================================

    print()
    print("3. CONSTANT FEATURE DETECTION")
    print("-" * 70)

    constant_features = []
    nonconstant_features = []

    for col in feature_cols:

        numeric = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if numeric.nunique(dropna=False) <= 1:
            constant_features.append(col)
        else:
            nonconstant_features.append(col)

    print(
        f"Globally constant features: "
        f"{len(constant_features)}"
    )

    print(
        f"Non-constant features: "
        f"{len(nonconstant_features)}"
    )

    if len(constant_features) != 397:
        print()
        print(
            "WARNING: Step 24 reported 397 constant features, "
            f"but direct calculation found {len(constant_features)}."
        )

    if not nonconstant_features:
        raise RuntimeError(
            "No non-constant features available."
        )

    # ================================================================
    # 4. NUMERIC CHECK
    # ================================================================

    print()
    print("4. NUMERIC FEATURE VALIDATION")
    print("-" * 70)

    usable_features = []
    non_numeric = []

    for col in nonconstant_features:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        # A feature is considered usable if at least one
        # finite numeric value exists.
        if np.isfinite(converted.to_numpy()).any():
            usable_features.append(col)
        else:
            non_numeric.append(col)

    print(f"Usable numeric features: {len(usable_features)}")
    print(f"Non-usable features: {len(non_numeric)}")

    if non_numeric:
        print()
        print("Non-usable features:")
        for col in non_numeric[:20]:
            print(f"  {col}")

    # ================================================================
    # 5. CLASS VALIDATION
    # ================================================================

    print()
    print("5. CLASS VALIDATION")
    print("-" * 70)

    labels = df[LABEL_COL].astype(str).to_numpy()

    counts = (
        df[LABEL_COL]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
    )

    for cls, count in counts.items():
        print(f"{cls:<35} {int(count):>3}")

    unknown = [
        c for c in df[LABEL_COL].unique()
        if c not in CLASS_ORDER
    ]

    if unknown:
        raise RuntimeError(
            f"Unexpected class labels: {unknown}"
        )

    # ================================================================
    # 6. DISCRIMINATION
    # ================================================================

    print()
    print("6. CALCULATING CLASS-LEVEL DISCRIMINATION")
    print("-" * 70)

    results = []

    total = len(usable_features)

    for i, col in enumerate(usable_features, start=1):

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        ).to_numpy(dtype=float)

        finite = np.isfinite(values)

        eta = eta_squared(
            values,
            labels
        )

        mean_range = class_mean_range(
            values[finite],
            labels[finite]
        )

        mean_cv = class_mean_cv(
            values[finite],
            labels[finite]
        )

        row = {
            "feature": col,
            "eta_squared": eta,
            "class_mean_range": mean_range,
            "class_mean_cv": mean_cv,
            "missing_values": int((~finite).sum()),
        }

        for cls in CLASS_ORDER:

            cls_values = values[
                (labels == cls) & finite
            ]

            safe = (
                cls.lower()
                .replace(" ", "_")
            )

            if len(cls_values) == 0:
                row[f"mean__{safe}"] = np.nan
            else:
                row[f"mean__{safe}"] = float(
                    np.mean(cls_values)
                )

        results.append(row)

        if i % 250 == 0 or i == total:
            print(
                f"  processed {i}/{total} features"
            )

    result_df = pd.DataFrame(results)

    # ================================================================
    # 7. RANK
    # ================================================================

    print()
    print("7. FEATURE RANKING")
    print("-" * 70)

    result_df = result_df.sort_values(
        "eta_squared",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    result_df.insert(
        0,
        "rank",
        np.arange(1, len(result_df) + 1)
    )

    result_df["stream"] = (
        result_df["feature"]
        .str.split("__")
        .str[0]
    )

    print()
    print("TOP 30 FEATURES BY ETA-SQUARED")
    print("-" * 70)

    print(
        result_df[
            [
                "rank",
                "feature",
                "eta_squared",
                "class_mean_range",
                "class_mean_cv",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    # ================================================================
    # 8. SUMMARY
    # ================================================================

    print()
    print("8. DISCRIMINATION SUMMARY")
    print("-" * 70)

    eta = result_df["eta_squared"].dropna()

    print(f"Features analyzed: {len(result_df)}")

    if len(eta) > 0:

        print(
            f"Eta-squared minimum: {eta.min():.6f}"
        )

        print(
            f"Eta-squared median:  {eta.median():.6f}"
        )

        print(
            f"Eta-squared maximum: {eta.max():.6f}"
        )

        print()
        print("FEATURE COUNTS BY ETA-SQUARED")

        for threshold in [
            0.00,
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
        ]:

            count = int(
                (eta >= threshold).sum()
            )

            print(
                f"  eta² >= {threshold:.2f}: {count}"
            )

    # ================================================================
    # 9. TOP FEATURES BY STREAM
    # ================================================================

    print()
    print("9. TOP FEATURES BY TELEMETRY STREAM")
    print("-" * 70)

    for stream in sorted(
        result_df["stream"].unique()
    ):

        subset = (
            result_df[
                result_df["stream"] == stream
            ]
            .sort_values(
                "eta_squared",
                ascending=False
            )
            .head(5)
        )

        print()
        print(stream)

        for _, row in subset.iterrows():

            print(
                f"  {row['feature']}"
                f" | eta²={row['eta_squared']:.6f}"
            )

    # ================================================================
    # 10. SAVE
    # ================================================================

    print()
    print("10. SAVING RESULTS")
    print("-" * 70)

    save_cols = [
        "rank",
        "stream",
        "feature",
        "eta_squared",
        "class_mean_range",
        "class_mean_cv",
        "missing_values",
    ]

    save_cols += [
        c for c in result_df.columns
        if c.startswith("mean__")
    ]

    result_df[save_cols].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Saved: {OUTPUT_FILE}")

    # ================================================================
    # 11. INTERPRETATION
    # ================================================================

    print()
    print("=" * 70)
    print("STEP 25 INTERPRETATION")
    print("=" * 70)

    print()
    print(
        "SEAD class-level feature discrimination was analyzed."
    )

    print(
        f"Globally constant features excluded: "
        f"{len(constant_features)}"
    )

    print(
        f"Non-constant features analyzed: "
        f"{len(usable_features)}"
    )

    print()
    print(
        "Eta-squared is used only as a descriptive "
        "class-separation measure."
    )

    print(
        "No classifier was trained."
    )

    print(
        "No feature selection was applied to the original "
        "SEAD feature table."
    )

    print(
        "step22_sead_run_features.csv remains unchanged."
    )

    print(
        "No FDI data was merged."
    )

    print(
        "fdi_train.csv was not modified."
    )

    print(
        "fdi_test.csv was not modified."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Review the Step 25 discrimination results before "
        "performing feature reduction or model training."
    )

    print()
    print("=" * 70)
    print("STEP 25 COMPLETE")
    print("=" * 70)

    print(
        f"SEAD runs analyzed: {len(df)}"
    )

    print(
        f"Features analyzed: {len(usable_features)}"
    )

    print(
        f"Constant features excluded: "
        f"{len(constant_features)}"
    )

    print(
        f"Results saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

