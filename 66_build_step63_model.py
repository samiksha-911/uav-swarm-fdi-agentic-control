from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier

BASE = Path("/home/samiksha/uav_fdi_data")
FAIR = BASE / "step60_px4_native_6class_fair_dataset.csv"
OOF = BASE / "step63_status_aware_oof_predictions.csv"
MODEL_OUT = BASE / "px4_native_6class_status_aware_model.joblib"

WINDOW = 50

# ------------------------------------------------------------
# Generic feature definition: EXACT Step 59 representation
# ------------------------------------------------------------
def add_generic_features(df):
    features = {}

    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "time"]

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if len(x) == 0 or not np.isfinite(x).all():
            continue

        features[f"{col}_mean"] = float(np.mean(x))
        features[f"{col}_std"] = float(np.std(x))
        features[f"{col}_min"] = float(np.min(x))
        features[f"{col}_max"] = float(np.max(x))
        features[f"{col}_range"] = float(np.max(x) - np.min(x))

        features[f"{col}_first"] = float(x[0])
        features[f"{col}_last"] = float(x[-1])
        features[f"{col}_delta"] = float(x[-1] - x[0])

        if len(x) > 1:
            dx = np.diff(x)
            features[f"{col}_mean_abs_change"] = float(np.mean(np.abs(dx)))
            features[f"{col}_max_abs_change"] = float(np.max(np.abs(dx)))
            features[f"{col}_change_std"] = float(np.std(dx))
            features[f"{col}_change_range"] = float(np.max(dx) - np.min(dx))
        else:
            features[f"{col}_mean_abs_change"] = 0.0
            features[f"{col}_max_abs_change"] = 0.0
            features[f"{col}_change_std"] = 0.0
            features[f"{col}_change_range"] = 0.0

    return features


# ------------------------------------------------------------
# Step 63 status feature definition
# ------------------------------------------------------------
def add_status_features(df):
    features = {}

    for col in df.columns:
        if col == "time":
            continue

        x = pd.to_numeric(df[col], errors="coerce")

        if x.isna().any():
            continue

        vals = x.to_numpy(dtype=float)

        # Only binary/status-like signals
        unique = np.unique(vals)
        if len(unique) == 0:
            continue

        if not np.all(np.isin(unique, [0.0, 1.0])):
            continue

        b = vals.astype(int)

        transitions = int(np.sum(b[1:] != b[:-1])) if len(b) > 1 else 0

        # Consecutive-run lengths
        true_run_max = 0
        false_run_max = 0

        current = b[0]
        run = 1

        for v in b[1:]:
            if v == current:
                run += 1
            else:
                if current == 1:
                    true_run_max = max(true_run_max, run)
                else:
                    false_run_max = max(false_run_max, run)
                current = v
                run = 1

        if current == 1:
            true_run_max = max(true_run_max, run)
        else:
            false_run_max = max(false_run_max, run)

        duration = (len(b) - 1) * 0.1 if len(b) > 1 else 0.0

        features[f"{col}_true_fraction"] = float(np.mean(b))
        features[f"{col}_first_state"] = float(b[0])
        features[f"{col}_last_state"] = float(b[-1])
        features[f"{col}_transitions"] = float(transitions)

        if duration > 0:
            features[f"{col}_transition_rate"] = float(transitions / duration)
        else:
            features[f"{col}_transition_rate"] = 0.0

        features[f"{col}_true_run_max"] = float(true_run_max)
        features[f"{col}_false_run_max"] = float(false_run_max)

    return features


# ------------------------------------------------------------
# Load authoritative fair dataset + Step 63 OOF keys
# ------------------------------------------------------------
fair = pd.read_csv(FAIR)
oof = pd.read_csv(OOF)

keys = oof[["run_id", "window_id"]].drop_duplicates()

print("Fair dataset:", fair.shape)
print("Step 63 OOF keys:", keys.shape)

# Keep EXACT Step 63 windows, in OOF order
fair = fair.merge(keys, on=["run_id", "window_id"], how="inner")

# Preserve OOF order
order = pd.MultiIndex.from_frame(keys)
fair["_order"] = pd.MultiIndex.from_frame(
    fair[["run_id", "window_id"]]
).map({k: i for i, k in enumerate(order)})
fair = fair.sort_values("_order").drop(columns="_order").reset_index(drop=True)

print("Matched windows:", fair.shape)

# Generic features from Step 60
metadata = ["label", "run_id", "window_id"]
generic_cols = [c for c in fair.columns if c not in metadata]

X_generic = fair[generic_cols].copy()

# ------------------------------------------------------------
# Reconstruct status features from exact raw recordings
# ------------------------------------------------------------
status_rows = []

for _, row in fair.iterrows():
    run_id = row["run_id"]
    window_id = int(row["window_id"])

    raw_path = BASE / f"{run_id}.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw recording not found for {run_id}: {raw_path}"
        )

    raw = pd.read_csv(raw_path)

    start = window_id * WINDOW
    end = start + WINDOW

    if end > len(raw):
        raise ValueError(
            f"Window outside raw recording: {run_id}, "
            f"window={window_id}, rows={len(raw)}"
        )

    w = raw.iloc[start:end].reset_index(drop=True)
    status_rows.append(add_status_features(w))

status = pd.DataFrame(status_rows)

print("Generic features:", X_generic.shape[1])
print("Reconstructed status features:", status.shape[1])

# Align indices
X_generic.index = range(len(X_generic))
status.index = range(len(status))

# Remove any duplicated feature names
status = status.loc[:, ~status.columns.duplicated()]

X = pd.concat([X_generic, status], axis=1)

# Remove duplicated columns
X = X.loc[:, ~X.columns.duplicated()]

# Clean
X = X.replace([np.inf, -np.inf], np.nan)

bad = X.columns[X.isna().any()].tolist()
if bad:
    print("Dropping incomplete features:", len(bad))
    X = X.drop(columns=bad)

# Remove constant features exactly as dataset preparation does
constant = [
    c for c in X.columns
    if X[c].nunique(dropna=False) <= 1
]

if constant:
    X = X.drop(columns=constant)

print("Final feature matrix:", X.shape)

# Check expected Step 63 scale
if not (650 <= X.shape[1] <= 700):
    raise RuntimeError(
        f"Unexpected feature count {X.shape[1]}; "
        "refusing to train a potentially mismatched live model."
    )

y = fair["label"].to_numpy()

print("\nClass counts:")
print(pd.Series(y).value_counts())

# ------------------------------------------------------------
# Train final deployment model on ALL Step 63 windows
# Evaluation was already performed with LOGO in Step 63.
# ------------------------------------------------------------
model = ExtraTreesClassifier(
    n_estimators=500,
    class_weight="balanced",
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

model.fit(X, y)

# Save model + exact feature ordering
bundle = {
    "model": model,
    "feature_names": X.columns.tolist(),
    "window_size": WINDOW,
    "feature_type": "Step63_generic_plus_status",
    "classes": model.classes_.tolist(),
}

joblib.dump(bundle, MODEL_OUT)

# Also save the reconstructed matrix for reproducibility
out = X.copy()
out["label"] = y
out["run_id"] = fair["run_id"].values
out["window_id"] = fair["window_id"].values
out.to_csv(BASE / "step63_status_aware_feature_matrix.csv", index=False)

print("\n======================================")
print("STEP 63 MODEL SAVED")
print("======================================")
print("Model:", MODEL_OUT)
print("Feature matrix:", BASE / "step63_status_aware_feature_matrix.csv")
print("Features:", len(X.columns))
print("Classes:", model.classes_.tolist())
print("Training windows:", len(X))
print("======================================")
