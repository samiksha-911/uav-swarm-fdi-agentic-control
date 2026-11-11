import os
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from joblib import dump


BASE = "/home/samiksha/uav_fdi_data"

DATASET = os.path.join(
    BASE,
    "day2_px4_native_window_dataset.csv"
)

OOF_OUTPUT = os.path.join(
    BASE,
    "step50_px4_aware_oof_predictions.csv"
)

MODEL_OUTPUT = os.path.join(
    BASE,
    "px4_native_fdi_model_step50.joblib"
)

RANDOM_STATE = 42
MAX_WINDOWS_PER_RUN = 10
WINDOW_SIZE = 50


# =========================================================
# Helper functions
# =========================================================

def transition_count(values):
    values = np.asarray(values, dtype=bool)

    if len(values) <= 1:
        return 0.0

    return float(np.sum(values[1:] != values[:-1]))


def longest_run(values, target=True):
    values = np.asarray(values, dtype=bool)

    best = 0
    current = 0

    for value in values:
        if value == target:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return float(best)


# =========================================================
# Raw recordings
# =========================================================

RUNS = [
    ("normal_run2", "step46_normal_run2.csv"),
    ("normal_run3", "step49_normal_run3.csv"),
    ("normal_run4", "step49_normal_run4.csv"),
    ("normal_run5", "step49_normal_run5.csv"),
    ("normal_run6", "step49_normal_run6.csv"),
    ("normal_run7", "step49_normal_run7.csv"),
    ("normal_run11", "step49_normal_run11.csv"),

    ("compass_fault_run2", "step47_compass_fault_run2.csv"),
    ("compass_fault_run3", "step49_compass_fault_run3.csv"),
    ("compass_fault_run4", "step49_compass_fault_run4.csv"),
    ("compass_fault_run5", "step49_compass_fault_run5.csv"),
]


# =========================================================
# Load authoritative Step 48 window list
# =========================================================

print("=" * 70)
print("STEP 50 — PX4-AWARE TEMPORAL/STATUS FEATURES")
print("=" * 70)

print("\nLoading authoritative Step 48 dataset...")

base_df = pd.read_csv(DATASET)

print("Step 48 dataset:", base_df.shape)

required_keys = [
    "run_id",
    "window_number",
    "label",
]

for col in required_keys:
    if col not in base_df.columns:
        raise RuntimeError(
            f"Required Step 48 column missing: {col}"
        )


# =========================================================
# Build lookup of exact Step 48 windows
# =========================================================

authoritative_windows = set(
    zip(
        base_df["run_id"].astype(str),
        base_df["window_number"].astype(int),
    )
)

print(
    "Authoritative Step 48 windows:",
    len(authoritative_windows)
)


# =========================================================
# Extract temporal features ONLY for those windows
# =========================================================

records = []

for run_id, filename in RUNS:

    path = os.path.join(BASE, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)

    print(
        f"\n{run_id}: "
        f"{len(raw)} raw samples"
    )

    n_windows = len(raw) // WINDOW_SIZE

    added = 0

    for window_number in range(n_windows):

        key = (
            run_id,
            window_number,
        )

        # -------------------------------------------------
        # CRITICAL:
        # Only create features for windows that actually
        # exist in the authoritative Step 48 dataset.
        # -------------------------------------------------

        if key not in authoritative_windows:
            continue

        start = window_number * WINDOW_SIZE
        end = start + WINDOW_SIZE

        w = raw.iloc[start:end].copy()

        feat = {
            "run_id": run_id,
            "window_number": window_number,
        }

        # =================================================
        # PX4 STATUS / BOOLEAN FEATURES
        # =================================================

        status_cols = [
            "mag_heading_ok",
            "mag_declination_ok",
            "mag_fault",
        ]

        for col in status_cols:

            if col not in w.columns:
                continue

            x = (
                w[col]
                .astype(bool)
                .to_numpy()
            )

            invalid = ~x

            transitions = transition_count(x)

            feat[
                f"{col}_invalid_fraction"
            ] = float(np.mean(invalid))

            feat[
                f"{col}_valid_fraction"
            ] = float(np.mean(x))

            feat[
                f"{col}_transition_count"
            ] = transitions

            feat[
                f"{col}_transition_rate"
            ] = transitions / max(len(x) - 1, 1)

            feat[
                f"{col}_first_state"
            ] = float(x[0])

            feat[
                f"{col}_last_state"
            ] = float(x[-1])

            feat[
                f"{col}_invalid_run_max"
            ] = longest_run(
                invalid,
                True
            )

            feat[
                f"{col}_valid_run_max"
            ] = longest_run(
                x,
                True
            )

            feat[
                f"{col}_invalid_duration_sec"
            ] = (
                longest_run(
                    invalid,
                    True
                ) / 10.0
            )

        # =================================================
        # OTHER BOOLEAN STATUS FEATURES
        # =================================================

        for col in w.columns:

            if col in status_cols:
                continue

            if w[col].dtype != bool:
                continue

            x = (
                w[col]
                .astype(bool)
                .to_numpy()
            )

            transitions = transition_count(x)

            feat[
                f"{col}_true_fraction"
            ] = float(np.mean(x))

            feat[
                f"{col}_transition_count"
            ] = transitions

            feat[
                f"{col}_transition_rate"
            ] = transitions / max(len(x) - 1, 1)

            feat[
                f"{col}_first_state"
            ] = float(x[0])

            feat[
                f"{col}_last_state"
            ] = float(x[-1])

        # =================================================
        # NUMERIC TEMPORAL FEATURES
        # =================================================

        numeric_cols = w.select_dtypes(
            include=[np.number]
        ).columns

        for col in numeric_cols:

            if col in status_cols:
                continue

            x = pd.to_numeric(
                w[col],
                errors="coerce"
            ).to_numpy(dtype=float)

            finite = np.isfinite(x)

            if not finite.any():
                continue

            x = x[finite]

            if len(x) == 0:
                continue

            feat[
                f"{col}_first"
            ] = float(x[0])

            feat[
                f"{col}_last"
            ] = float(x[-1])

            feat[
                f"{col}_delta"
            ] = float(x[-1] - x[0])

            if len(x) > 1:

                dx = np.diff(x)

                feat[
                    f"{col}_mean_abs_change"
                ] = float(
                    np.mean(np.abs(dx))
                )

                feat[
                    f"{col}_max_abs_change"
                ] = float(
                    np.max(np.abs(dx))
                )

                feat[
                    f"{col}_change_std"
                ] = float(
                    np.std(dx)
                )

                feat[
                    f"{col}_change_range"
                ] = float(
                    np.max(dx) - np.min(dx)
                )

            else:

                feat[
                    f"{col}_mean_abs_change"
                ] = 0.0

                feat[
                    f"{col}_max_abs_change"
                ] = 0.0

                feat[
                    f"{col}_change_std"
                ] = 0.0

                feat[
                    f"{col}_change_range"
                ] = 0.0

        records.append(feat)
        added += 1

    print(
        f"  Added Step 50 windows: {added}"
    )


temporal_df = pd.DataFrame(records)

print(
    "\nTemporal feature dataset:",
    temporal_df.shape
)


# =========================================================
# Verify exact window alignment
# =========================================================

base_keys = (
    base_df[
        ["run_id", "window_number"]
    ]
    .astype({
        "run_id": str,
        "window_number": int,
    })
    .sort_values(
        ["run_id", "window_number"]
    )
    .reset_index(drop=True)
)

temporal_keys = (
    temporal_df[
        ["run_id", "window_number"]
    ]
    .astype({
        "run_id": str,
        "window_number": int,
    })
    .sort_values(
        ["run_id", "window_number"]
    )
    .reset_index(drop=True)
)

print(
    "\nStep 48 windows:",
    len(base_keys)
)

print(
    "Step 50 windows:",
    len(temporal_keys)
)

if not base_keys.equals(temporal_keys):

    print(
        "\nERROR: window alignment still failed."
    )

    missing = (
        set(map(tuple, base_keys.values))
        -
        set(map(tuple, temporal_keys.values))
    )

    extra = (
        set(map(tuple, temporal_keys.values))
        -
        set(map(tuple, base_keys.values))
    )

    print(
        "Missing windows:",
        sorted(missing)
    )

    print(
        "Extra windows:",
        sorted(extra)
    )

    raise RuntimeError(
        "Step 50 window alignment failed."
    )

print(
    "Window alignment: PASS"
)


# =========================================================
# Attach authoritative Step 48 labels
# =========================================================

labels_df = base_df[
    [
        "run_id",
        "window_number",
        "label",
    ]
].copy()

temporal_df = temporal_df.merge(
    labels_df,
    on=[
        "run_id",
        "window_number",
    ],
    how="left",
    validate="one_to_one",
)


if temporal_df["label"].isna().any():
    raise RuntimeError(
        "Some Step 50 windows have no Step 48 label."
    )


print(
    "\nClass distribution before cap:"
)

print(
    temporal_df["label"].value_counts()
)


# =========================================================
# Same deterministic run cap as Step 49
# =========================================================

parts = []

for run_id, group in temporal_df.groupby(
    "run_id",
    sort=True
):

    if len(group) > MAX_WINDOWS_PER_RUN:

        group = group.sample(
            n=MAX_WINDOWS_PER_RUN,
            random_state=RANDOM_STATE
        )

    parts.append(group)

df = pd.concat(
    parts,
    ignore_index=True
)

print(
    "\nEvaluation dataset:",
    df.shape
)

print(
    "\nWindows per run:"
)

print(
    df.groupby("run_id").size()
)

print(
    "\nClass distribution:"
)

print(
    df["label"].value_counts()
)


# =========================================================
# Prepare ML data
# =========================================================

META = [
    "run_id",
    "window_number",
    "label",
]

feature_cols = [
    c for c in df.columns
    if c not in META
]

X = df[feature_cols].copy()

y = df["label"].astype(str)

groups = df["run_id"].astype(str)


# =========================================================
# Remove constant features
# =========================================================

constant_features = [
    c for c in X.columns
    if X[c].nunique(dropna=False) <= 1
]

print(
    "\nConstant features:",
    len(constant_features)
)

if constant_features:
    X = X.drop(
        columns=constant_features
    )

feature_cols = X.columns.tolist()

print(
    "Remaining features:",
    len(feature_cols)
)


# =========================================================
# Leave-One-Run-Out evaluation
# =========================================================

logo = LeaveOneGroupOut()

oof_true = []
oof_pred = []
oof_run = []
oof_window = []

print("\n" + "=" * 70)
print("LEAVE-ONE-RUN-OUT EVALUATION")
print("=" * 70)

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    start=1
):

    train_X = X.iloc[train_idx]
    test_X = X.iloc[test_idx]

    train_y = y.iloc[train_idx]
    test_y = y.iloc[test_idx]

    test_run = groups.iloc[test_idx].iloc[0]

    imputer = SimpleImputer(
        strategy="median"
    )

    train_X_imp = imputer.fit_transform(
        train_X
    )

    test_X_imp = imputer.transform(
        test_X
    )

    model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
    )

    model.fit(
        train_X_imp,
        train_y
    )

    pred = model.predict(
        test_X_imp
    )

    fold_acc = accuracy_score(
        test_y,
        pred
    )

    fold_bal = balanced_accuracy_score(
        test_y,
        pred
    )

    print(
        f"{test_run:25s} "
        f"accuracy={fold_acc:.4f} "
        f"balanced_accuracy={fold_bal:.4f} "
        f"n={len(test_idx)}"
    )

    oof_true.extend(
        test_y.tolist()
    )

    oof_pred.extend(
        pred.tolist()
    )

    oof_run.extend(
        groups.iloc[test_idx].tolist()
    )

    oof_window.extend(
        df.iloc[test_idx]["window_number"].tolist()
    )


# =========================================================
# Overall results
# =========================================================

accuracy = accuracy_score(
    oof_true,
    oof_pred
)

balanced_accuracy = balanced_accuracy_score(
    oof_true,
    oof_pred
)

print("\n" + "=" * 70)
print("STEP 50 RESULTS")
print("=" * 70)

print(
    f"\nOverall accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Overall balanced accuracy: "
    f"{balanced_accuracy:.4f}"
)

print(
    "\nClassification report:"
)

print(
    classification_report(
        oof_true,
        oof_pred,
        digits=4
    )
)

labels = [
    "Compass Failure",
    "No Failure",
]

print(
    "\nConfusion matrix:"
)

print(
    confusion_matrix(
        oof_true,
        oof_pred,
        labels=labels
    )
)


# =========================================================
# Save OOF predictions
# =========================================================

oof_df = pd.DataFrame({
    "run_id": oof_run,
    "window_number": oof_window,
    "true_label": oof_true,
    "predicted_label": oof_pred,
})

oof_df.to_csv(
    OOF_OUTPUT,
    index=False
)

print(
    "\nSaved OOF predictions:"
)

print(
    OOF_OUTPUT
)


# =========================================================
# Train final Step 50 model
# =========================================================

print(
    "\nTraining final Step 50 model..."
)

final_imputer = SimpleImputer(
    strategy="median"
)

X_final = final_imputer.fit_transform(
    X
)

final_model = ExtraTreesClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    max_features="sqrt",
    n_jobs=-1,
)

final_model.fit(
    X_final,
    y
)


# =========================================================
# Save model bundle
# =========================================================

bundle = {
    "model": final_model,
    "imputer": final_imputer,
    "features": feature_cols,
    "constant_features_removed": constant_features,
    "label_column": "label",
    "run_column": "run_id",
    "window_column": "window_number",
    "n_features": len(feature_cols),
    "model_name": "ExtraTreesClassifier",
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
    "max_features": "sqrt",
    "classes": final_model.classes_.tolist(),
    "max_windows_per_run": MAX_WINDOWS_PER_RUN,
    "step49_baseline_balanced_accuracy": 0.7281,
    "step50_balanced_accuracy": float(
        balanced_accuracy
    ),
    "description": (
        "PX4-native FDI model using telemetry-derived "
        "window statistics plus PX4-aware temporal/status "
        "features. The authoritative Step 48 window "
        "definitions and labels are retained. Labels are "
        "never used as model input features."
    ),
}

dump(
    bundle,
    MODEL_OUTPUT
)

print(
    "\nSaved Step 50 model:"
)

print(
    MODEL_OUTPUT
)

print("\nDONE.")
