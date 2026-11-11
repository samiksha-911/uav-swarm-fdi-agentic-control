import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE = Path.home() / "uav_fdi_data"

INPUT = BASE / "step60_no_gps_vspeed_fair_dataset.csv"
OOF_OUTPUT = BASE / "step61_px4_native_6class_oof_predictions.csv"
MODEL_OUTPUT = BASE / "px4_native_6class_model.joblib"

RANDOM_STATE = 42
N_ESTIMATORS = 500


print("=" * 70)
print("STEP 61 — PX4-NATIVE 6-CLASS RUN-AWARE EVALUATION")
print("=" * 70)

df = pd.read_csv(INPUT)

metadata = ["label", "run_id", "window_id"]

X = df.drop(columns=metadata)
y = df["label"]
groups = df["run_id"]

print("\nDataset shape:", df.shape)
print("Feature count:", X.shape[1])
print("Classes:", sorted(y.unique()))
print("Runs:", groups.nunique())

print("\nClass counts:")
print(y.value_counts())

print("\nRun counts:")
print(groups.value_counts())


# ------------------------------------------------------------
# Leave-One-Run-Out evaluation
# ------------------------------------------------------------

logo = LeaveOneGroupOut()

all_true = []
all_pred = []
all_prob = []
all_run = []
all_window = []

classes = sorted(y.unique())

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    test_runs = groups.iloc[test_idx].unique()

    print(
        f"\nFold {fold}: "
        f"test run(s) = {list(test_runs)}"
    )

    print(
        "  train samples:",
        len(train_idx),
        "test samples:",
        len(test_idx)
    )

    # Some folds contain only one class because a fault recording
    # contains only that fault. ExtraTrees cannot train with a
    # single class, so detect this explicitly.
    if y_train.nunique() < 2:
        print("  SKIPPED: training set has fewer than 2 classes.")
        continue

    model = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    all_true.extend(y_test.tolist())
    all_pred.extend(pred.tolist())
    all_run.extend(groups.iloc[test_idx].tolist())
    all_window.extend(
        df.iloc[test_idx]["window_id"].tolist()
    )

    # Probability columns are stored according to model.classes_.
    probabilities = model.predict_proba(X_test)

    for row in probabilities:
        prob_dict = dict(zip(model.classes_, row))
        all_prob.append(
            [prob_dict.get(c, 0.0) for c in classes]
        )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

if len(all_true) == 0:
    raise RuntimeError("No valid evaluation folds were produced.")

all_true = np.array(all_true)
all_pred = np.array(all_pred)

accuracy = accuracy_score(all_true, all_pred)
balanced_acc = balanced_accuracy_score(all_true, all_pred)
macro_f1 = f1_score(
    all_true,
    all_pred,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        all_true,
        all_pred,
        labels=classes,
        zero_division=0,
    )
)

print("\nConfusion matrix:")
cm = confusion_matrix(
    all_true,
    all_pred,
    labels=classes,
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
)

print(cm_df)


# ------------------------------------------------------------
# Per-run results
# ------------------------------------------------------------

result_df = pd.DataFrame({
    "run_id": all_run,
    "window_id": all_window,
    "true_label": all_true,
    "predicted_label": all_pred,
})

print("\n" + "=" * 70)
print("PER-RUN RESULTS")
print("=" * 70)

for run_id, group in result_df.groupby("run_id"):
    run_acc = accuracy_score(
        group["true_label"],
        group["predicted_label"],
    )

    print(
        f"{run_id}: "
        f"{run_acc:.4f} "
        f"({len(group)} windows)"
    )


# ------------------------------------------------------------
# Save OOF predictions
# ------------------------------------------------------------

result_df.to_csv(
    OOF_OUTPUT,
    index=False,
)

print("\nOOF predictions saved:")
print(OOF_OUTPUT)


# ------------------------------------------------------------
# Train final model on ALL fair data
#
# IMPORTANT:
# This model is for deployment/demo inference.
# Its performance must NOT be reported as evaluation
# performance because it was trained on all available runs.
# ------------------------------------------------------------

final_model = ExtraTreesClassifier(
    n_estimators=N_ESTIMATORS,
    class_weight="balanced",
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_model.fit(X, y)

import joblib

joblib.dump(
    final_model,
    MODEL_OUTPUT,
)

print("\nFinal deployment model saved:")
print(MODEL_OUTPUT)

print("\nFinal model classes:")
print(final_model.classes_)

print("\nDONE.")
