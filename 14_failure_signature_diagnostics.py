import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

TEST_FILE = "fdi_test.csv"
PRED_FILE = "run_level_predictions.csv"

print("=" * 70)
print("STEP 14 — FAILURE SIGNATURE & PREDICTION DIAGNOSTICS")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================

test = pd.read_csv(TEST_FILE)
pred = pd.read_csv(PRED_FILE)

print("\nTest rows:", len(test))
print("Test runs:", test["run"].nunique())

print("Prediction rows:", len(pred))
print("Prediction runs:", pred["run"].nunique())

# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_test_cols = {
    "run",
    "failure_type"
}

missing_test = required_test_cols - set(test.columns)

if missing_test:
    raise ValueError(
        f"Test file is missing required columns: {sorted(missing_test)}"
    )

required_pred_cols = {
    "run"
}

missing_pred = required_pred_cols - set(pred.columns)

if missing_pred:
    raise ValueError(
        f"Prediction file is missing required columns: {sorted(missing_pred)}"
    )

# Determine prediction column
if "predicted_failure_type" in pred.columns:
    prediction_col = "predicted_failure_type"

elif "prediction" in pred.columns:
    prediction_col = "prediction"

else:
    raise ValueError(
        "Prediction file must contain either "
        "'predicted_failure_type' or 'prediction'."
    )

print("\nPrediction column:", prediction_col)

# ============================================================
# 3. SAMPLE-LEVEL TRUE CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("1. SAMPLE-LEVEL CLASS DISTRIBUTION")
print("-" * 70)

print("\nTrue classes:")
print(
    test["failure_type"]
    .value_counts()
    .sort_index()
)

print("\nNumber of complete runs per true class:")

run_class_check = (
    test.groupby("run")["failure_type"]
    .nunique()
)

if (run_class_check > 1).any():
    print(
        "WARNING: Some runs contain multiple failure classes."
    )

else:
    print(
        "All runs contain exactly one failure class."
    )

print(
    test.groupby("failure_type")["run"]
    .nunique()
    .sort_index()
)

# ============================================================
# 4. CHECK PREDICTION FILE
# ============================================================

print("\n" + "-" * 70)
print("2. RUN-LEVEL PREDICTION FILE")
print("-" * 70)

print("\nPrediction distribution across runs:")

print(
    pred[prediction_col]
    .value_counts()
    .sort_index()
)

# Verify one prediction per run
prediction_counts = (
    pred.groupby("run")[prediction_col]
    .nunique()
)

bad_prediction_runs = prediction_counts[
    prediction_counts > 1
]

if len(bad_prediction_runs) > 0:

    print(
        "\nERROR: Some runs have multiple different predictions:"
    )

    print(
        bad_prediction_runs.to_string()
    )

    raise ValueError(
        "Prediction file contains multiple predictions for "
        "the same run."
    )

print(
    "\nUnique prediction runs:",
    pred["run"].nunique()
)

# ============================================================
# 5. VERIFY RUN MATCHING
# ============================================================

print("\n" + "-" * 70)
print("3. RUN MATCHING")
print("-" * 70)

test_runs = set(test["run"].unique())
prediction_runs = set(pred["run"].unique())

missing_predictions = test_runs - prediction_runs
extra_predictions = prediction_runs - test_runs

print("Test runs:", len(test_runs))
print("Prediction runs:", len(prediction_runs))

print(
    "Missing predictions:",
    len(missing_predictions)
)

print(
    "Extra prediction runs:",
    len(extra_predictions)
)

if missing_predictions:

    print("\nRuns missing predictions:")

    for run in sorted(missing_predictions):
        print(" ", run)

    raise ValueError(
        "Some test runs do not have predictions."
    )

if extra_predictions:

    print("\nWARNING: Prediction file contains runs "
          "not present in test data:")

    for run in sorted(extra_predictions):
        print(" ", run)

# ============================================================
# 6. MERGE RUN-LEVEL PREDICTIONS ONTO SAMPLE DATA
# ============================================================

print("\n" + "-" * 70)
print("4. MERGING RUN PREDICTIONS WITH SAMPLE DATA")
print("-" * 70)

# Keep one prediction per run
run_predictions = (
    pred[
        [
            "run",
            prediction_col
        ]
    ]
    .drop_duplicates("run")
    .copy()
)

# Merge one run prediction onto all samples
analysis = test[
    [
        "run",
        "failure_type"
    ]
].merge(
    run_predictions,
    on="run",
    how="left",
    validate="many_to_one"
)

analysis = analysis.rename(
    columns={
        prediction_col: "prediction"
    }
)

print("\nTest samples:", len(test))
print("Analysis samples:", len(analysis))

print(
    "Missing sample predictions:",
    analysis["prediction"].isna().sum()
)

if len(analysis) != len(test):

    raise RuntimeError(
        "Merge changed the number of test samples."
    )

if analysis["prediction"].isna().any():

    raise RuntimeError(
        "Some samples do not have a run-level prediction."
    )

# ============================================================
# 7. SAMPLE-LEVEL PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("5. SAMPLE-LEVEL PREDICTION DISTRIBUTION")
print("-" * 70)

print(
    analysis["prediction"]
    .value_counts()
    .sort_index()
)

# ============================================================
# 8. PER-RUN PREDICTION DISTRIBUTIONS
# ============================================================

print("\n" + "-" * 70)
print("6. PER-RUN PREDICTION DISTRIBUTIONS")
print("-" * 70)

for run, group in analysis.groupby(
    "run",
    sort=False
):

    true_class = group["failure_type"].iloc[0]

    prediction_counts = (
        group["prediction"]
        .value_counts()
    )

    prediction_percent = (
        prediction_counts
        .div(len(group))
        .mul(100)
        .round(2)
    )

    print("\nRun:")
    print(run)

    print("True class:")
    print(true_class)

    print("Samples:")
    print(len(group))

    print("Predictions:")

    print(
        prediction_percent.to_string()
    )

# ============================================================
# 9. RUN-LEVEL SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("7. RUN-LEVEL SUMMARY")
print("-" * 70)

run_records = []

for run, group in analysis.groupby(
    "run",
    sort=False
):

    true_class = group["failure_type"].iloc[0]

    prediction_counts = (
        group["prediction"]
        .value_counts()
    )

    dominant_prediction = (
        prediction_counts.index[0]
    )

    dominant_count = (
        prediction_counts.iloc[0]
    )

    dominant_percentage = (
        dominant_count
        / len(group)
        * 100
    )

    run_records.append(
        {
            "run": run,
            "true_failure_type": true_class,
            "predicted_failure_type": dominant_prediction,
            "n_samples": len(group),
            "dominant_prediction_pct":
                round(dominant_percentage, 2),
            "correct":
                true_class == dominant_prediction
        }
    )

run_summary = pd.DataFrame(
    run_records
)

print(
    run_summary.to_string(
        index=False
    )
)

# ============================================================
# 10. TRUE CLASS → PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("8. TRUE CLASS → PREDICTION DISTRIBUTION")
print("-" * 70)

cross = pd.crosstab(
    analysis["failure_type"],
    analysis["prediction"],
    normalize="index"
) * 100

cross = cross.round(2)

print(
    cross.to_string()
)

# ============================================================
# 11. RUN-LEVEL CONFUSION MATRIX
# ============================================================

print("\n" + "-" * 70)
print("9. RUN-LEVEL CONFUSION MATRIX")
print("-" * 70)

labels = sorted(
    analysis["failure_type"].unique()
)

run_true = (
    run_summary["true_failure_type"]
)

run_pred = (
    run_summary["predicted_failure_type"]
)

cm = confusion_matrix(
    run_true,
    run_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(
    cm_df.to_string()
)

# ============================================================
# 12. RUN-LEVEL ACCURACY
# ============================================================

run_accuracy = (
    run_summary["correct"]
    .mean()
)

print("\nRun-level accuracy:")
print(
    f"{run_accuracy:.4f} "
    f"({run_summary['correct'].sum()}/"
    f"{len(run_summary)})"
)

# ============================================================
# 13. DOMINANT CLASS BIAS
# ============================================================

print("\n" + "-" * 70)
print("10. DOMINANT PREDICTION BIAS")
print("-" * 70)

run_prediction_counts = (
    run_summary["predicted_failure_type"]
    .value_counts()
)

print(
    run_prediction_counts.to_string()
)

print("\nPercentage of runs predicted as each class:")

print(
    (
        run_prediction_counts
        / len(run_summary)
        * 100
    )
    .round(2)
    .to_string()
)

# ============================================================
# 14. FAILURE CLASS PERFORMANCE
# ============================================================

print("\n" + "-" * 70)
print("11. PER-CLASS RUN PERFORMANCE")
print("-" * 70)

for cls in labels:

    class_runs = run_summary[
        run_summary["true_failure_type"] == cls
    ]

    if len(class_runs) == 0:
        continue

    correct = class_runs["correct"].sum()
    total = len(class_runs)

    accuracy = correct / total

    print(
        f"{cls:25s} "
        f"{correct}/{total} "
        f"({accuracy:.2%})"
    )

# ============================================================
# 15. METADATA CHECK
# ============================================================

print("\n" + "-" * 70)
print("12. METADATA CHECK")
print("-" * 70)

metadata_cols = [
    "Status",
    "sample_index",
    "relative_to_failure",
    "run"
]

for col in metadata_cols:

    if col not in test.columns:
        continue

    print(f"\n{col}:")

    if col == "run":

        print(
            "Unique:",
            test[col].nunique()
        )

    else:

        print(
            "Non-null:",
            test[col].notna().sum()
        )

        print(
            "Null:",
            test[col].isna().sum()
        )

        if test[col].notna().any():

            print(
                "Min:",
                test[col].min()
            )

            print(
                "Max:",
                test[col].max()
            )

# ============================================================
# 16. FAILURE SIGNATURE ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("13. FAILURE SIGNATURE SUMMARY")
print("-" * 70)

exclude = {
    "failure_type",
    "run",
    "sample_index",
    "relative_to_failure",
    "Status"
}

numeric_cols = [
    col
    for col in test.select_dtypes(
        include=np.number
    ).columns
    if col not in exclude
]

print(
    "\nNumeric feature count:",
    len(numeric_cols)
)

# Class means
class_means = (
    test
    .groupby("failure_type")[numeric_cols]
    .mean()
)

# Global means
global_mean = (
    test[numeric_cols]
    .mean()
)

# Global standard deviation
global_std = (
    test[numeric_cols]
    .std()
)

# Standardized class differences
z = (
    class_means - global_mean
).div(
    global_std.replace(
        0,
        np.nan
    )
)

# Maximum absolute standardized difference
max_abs_z = (
    z.abs()
    .max(axis=0)
)

top_features = (
    max_abs_z
    .sort_values(
        ascending=False
    )
    .head(30)
)

print(
    "\nTop 30 features showing strongest "
    "class differences:"
)

for feature, score in top_features.items():

    print(
        f"{feature:25s} "
        f"max |z| = {score:.3f}"
    )

# ============================================================
# 17. CLASS MEAN COMPARISON FOR TOP FEATURES
# ============================================================

print("\n" + "-" * 70)
print("14. TOP FEATURE CLASS MEANS")
print("-" * 70)

top_feature_names = (
    top_features.index.tolist()
)

if len(top_feature_names) > 0:

    print(
        class_means[
            top_feature_names
        ]
        .round(4)
        .to_string()
    )

# ============================================================
# 18. SAVE OUTPUTS
# ============================================================

print("\n" + "-" * 70)
print("15. SAVING DIAGNOSTIC OUTPUTS")
print("-" * 70)

run_summary.to_csv(
    "step14_run_diagnostics.csv",
    index=False
)

cross.to_csv(
    "step14_prediction_distribution.csv"
)

class_means.to_csv(
    "step14_class_means.csv"
)

top_features.rename(
    "max_abs_z"
).to_csv(
    "step14_top_features.csv"
)

# Save merged sample-level analysis
analysis.to_csv(
    "step14_sample_analysis.csv",
    index=False
)

print("\nSaved:")
print("  step14_run_diagnostics.csv")
print("  step14_prediction_distribution.csv")
print("  step14_class_means.csv")
print("  step14_top_features.csv")
print("  step14_sample_analysis.csv")

# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)

print(
    f"\nComplete test runs: {len(run_summary)}"
)

print(
    f"Correct runs: "
    f"{run_summary['correct'].sum()}"
)

print(
    f"Incorrect runs: "
    f"{(~run_summary['correct']).sum()}"
)

print(
    f"Run-level accuracy: "
    f"{run_accuracy:.4f}"
)

print(
    "\nNext step: use these diagnostics to determine "
    "whether the problem is class collapse, weak "
    "failure signatures, feature selection, or "
    "training-window construction."
)

print("=" * 70)
