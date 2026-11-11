#!/usr/bin/env python3
"""
step70_build_and_train_clean.py
================================
Builds a clean, balanced PX4-native FDI dataset from the newly
collected 30 runs (including magnetometer/barometer/GPS estimator flags),
applies proper boolean type mapping and temporal fault-window slicing,
evaluates using Leave-One-Run-Out (LOOR) cross-validation,
and trains the final production model.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut

BASE = Path.home() / "uav_fdi_data"
WINDOW = 50
STEP   = 25   # 50% overlap for sliding windows

# ------------------------------------------------------------------ #
# CLEAN RUN FILE MAP (Runs with complete 62-column telemetry)
# ------------------------------------------------------------------ #
CLEAN_RUNS = {
    "Accelerometer Failure": [f"step_auto_accel_fault_run{i}.csv" for i in range(17, 22)],
    "Gyroscope Failure":     [f"step_auto_gyro_fault_run{i}.csv" for i in range(16, 21)],
    "Compass Failure":       [f"step_auto_compass_fault_run{i}.csv" for i in range(16, 21)],
    "Barometer Failure":     [f"step_auto_baro_fault_run{i}.csv" for i in range(16, 21)],
    "GPS Failure":           [f"step_auto_gps_fault_run{i}.csv" for i in range(16, 21)],
    "No Failure":            [f"step_auto_normal_run{i}.csv" for i in range(6, 11)],
}

# ------------------------------------------------------------------ #
# FEATURE EXTRACTION
# ------------------------------------------------------------------ #
def extract_features(window_df):
    """Extract 12 temporal & statistical features per column."""
    feats = {}
    for col in window_df.columns:
        if col in ("time", "label", "run_id"):
            continue
        
        # Explicitly map string and boolean flags to 1.0 / 0.0 floats
        s = window_df[col].replace({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
        x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
            
        feats[f"{col}_mean"]  = float(np.mean(x))
        feats[f"{col}_std"]   = float(np.std(x))
        feats[f"{col}_min"]   = float(np.min(x))
        feats[f"{col}_max"]   = float(np.max(x))
        feats[f"{col}_range"] = float(np.max(x) - np.min(x))
        if len(x) > 1:
            feats[f"{col}_first"] = float(x[0])
            feats[f"{col}_last"]  = float(x[-1])
            feats[f"{col}_delta"] = float(x[-1] - x[0])
            dx = np.diff(x)
            feats[f"{col}_mean_abs_change"] = float(np.mean(np.abs(dx)))
            feats[f"{col}_max_abs_change"]  = float(np.max(np.abs(dx)))
            feats[f"{col}_change_std"]      = float(np.std(dx))
            feats[f"{col}_change_range"]    = float(np.max(dx) - np.min(dx))
    return feats


# ------------------------------------------------------------------ #
# BUILD DATASET
# ------------------------------------------------------------------ #
print("\n" + "="*60)
print("STEP 70 — BUILD CLEAN BALANCED PX4-NATIVE FDI DATASET")
print("="*60)

rows = []
run_id = 0

for label, files in CLEAN_RUNS.items():
    class_windows = 0
    for fname in files:
        fpath = BASE / fname
        if not fpath.exists():
            print(f"  [MISSING] {fname} — skipping")
            continue

        df = pd.read_csv(fpath)

        # Slice active window:
        # Fault runs: fault starts at t=10s, take t >= 11.0s to avoid boundary transition
        # Normal runs: discard startup settling, take t >= 2.0s
        if label != "No Failure":
            df = df[df["time"] >= 11.0].reset_index(drop=True)
        else:
            df = df[df["time"] >= 2.0].reset_index(drop=True)

        if len(df) < WINDOW:
            print(f"  [SKIP] {fname}: only {len(df)} rows after temporal slice")
            continue

        n_windows = 0
        for start in range(0, len(df) - WINDOW + 1, STEP):
            w = df.iloc[start:start + WINDOW]
            f = extract_features(w)
            f["label"]  = label
            f["run_id"] = run_id
            rows.append(f)
            n_windows += 1

        class_windows += n_windows
        run_id += 1

    print(f"  {label}: {len(files)} runs, {class_windows} windows")

dataset = pd.DataFrame(rows).fillna(0)

print(f"\nDataset shape : {dataset.shape}")
print(f"Total runs    : {run_id}")
print(f"\nWindows per class:")
print(dataset["label"].value_counts().to_string())

# Save dataset
dataset_path = BASE / "step70_clean_6class_dataset.csv"
dataset.to_csv(dataset_path, index=False)
print(f"\nDataset saved : {dataset_path}")

# ------------------------------------------------------------------ #
# PREPARE ML DATA
# ------------------------------------------------------------------ #
feature_cols = [c for c in dataset.columns if c not in ("label", "run_id")]
X = dataset[feature_cols].to_numpy(dtype=float)
y = dataset["label"].to_numpy()
groups = dataset["run_id"].to_numpy()

# ------------------------------------------------------------------ #
# LEAVE-ONE-RUN-OUT CROSS-VALIDATION
# ------------------------------------------------------------------ #
print("\n" + "="*60)
print("LEAVE-ONE-RUN-OUT CROSS-VALIDATION (LOOR)")
print("="*60)

logo = LeaveOneGroupOut()
oof_true = []
oof_pred = []

for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    clf = ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_val)
    oof_true.extend(y_val.tolist())
    oof_pred.extend(preds.tolist())

oof_true = np.array(oof_true)
oof_pred = np.array(oof_pred)

loor_acc     = accuracy_score(oof_true, oof_pred)
loor_bal_acc = balanced_accuracy_score(oof_true, oof_pred)
loor_f1      = f1_score(oof_true, oof_pred, average="macro", zero_division=0)

print(f"\nLOOR Accuracy          : {loor_acc:.4f} ({loor_acc*100:.2f}%)")
print(f"LOOR Balanced Accuracy : {loor_bal_acc:.4f} ({loor_bal_acc*100:.2f}%)")
print(f"LOOR Macro-F1          : {loor_f1:.4f}")

print(f"\nDetailed Classification Report:")
print(classification_report(oof_true, oof_pred, zero_division=0))

print("Confusion Matrix (rows=True, cols=Predicted):")
classes = sorted(set(oof_true))
cm = confusion_matrix(oof_true, oof_pred, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df.to_string())

# Save OOF predictions
oof_df = pd.DataFrame({"true": oof_true, "pred": oof_pred})
oof_path = BASE / "step70_loor_predictions.csv"
oof_df.to_csv(oof_path, index=False)
print(f"\nOOF predictions saved : {oof_path}")

# ------------------------------------------------------------------ #
# TRAIN FINAL MODEL ON ALL CLEAN DATA
# ------------------------------------------------------------------ #
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON ALL CLEAN DATA")
print("="*60)

final_clf = ExtraTreesClassifier(
    n_estimators=300,
    class_weight="balanced",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
final_clf.fit(X, y)

model_path = BASE / "step70_clean_fdi_model.joblib"
feature_names_path = BASE / "step70_feature_names.joblib"

joblib.dump(final_clf, model_path)
joblib.dump(feature_cols, feature_names_path)
print(f"Final model saved         : {model_path}")
print(f"Feature column list saved : {feature_names_path}")

print("\n" + "="*60)
print("COMPLETED SUCCESSFULLY")
print("="*60)
