import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# STEP 13 — RUN-LEVEL / EPISODE-LEVEL EVALUATION
# ============================================================

PREDICTIONS = "baseline_predictions.csv"

print("=" * 70)
print("RUN-LEVEL / EPISODE-LEVEL EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD SAMPLE-LEVEL PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(PREDICTIONS)

print("\nColumns:")
print(df.columns.tolist())

required = {
    "run",
    "failure_type",
    "prediction"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

print("\nRows:", len(df))
print("Unique runs:", df["run"].nunique())

# ------------------------------------------------------------
# 2. VERIFY EACH RUN HAS ONE TRUE CLASS
# ------------------------------------------------------------

true_classes_per_run = (
    df.groupby("run")["failure_type"]
      .nunique()
)

bad_runs = true_classes_per_run[
    true_classes_per_run != 1
]

print("\nRuns with exactly 1 true class:",
      (true_classes_per_run == 1).sum())

print("Runs with multiple/zero true classes:",
      len(bad_runs))

if len(bad_runs) > 0:
    print("\nProblematic runs:")
    print(bad_runs)

    raise RuntimeError(
        "Run-level evaluation cannot continue because "
        "some runs do not have exactly one true class."
    )

# ------------------------------------------------------------
# 3. SAMPLE-LEVEL ACCURACY
# ------------------------------------------------------------

sample_accuracy = accuracy_score(
    df["failure_type"],
    df["prediction"]
)

print("\n" + "-" * 70)
print("SAMPLE-LEVEL PERFORMANCE")
print("-" * 70)

print(f"Sample accuracy: {sample_accuracy:.4f}")

# ------------------------------------------------------------
# 4. MAJORITY VOTE PER COMPLETE RUN
# ------------------------------------------------------------

def majority_vote(series):
    counts = series.value_counts()

    max_count = counts.max()

    winners = sorted(
        counts[counts == max_count].index.astype(str)
    )

    return winners[0]


run_predictions = (
    df.groupby("run")
      .agg(
          true_failure_type=(
              "failure_type",
              "first"
          ),
          predicted_failure_type=(
              "prediction",
              majority_vote
          ),
          n_samples=(
              "prediction",
              "size"
          )
      )
      .reset_index()
)

# ------------------------------------------------------------
# 5. RUN-LEVEL ACCURACY
# ------------------------------------------------------------

run_accuracy = accuracy_score(
    run_predictions["true_failure_type"],
    run_predictions["predicted_failure_type"]
)

print("\n" + "-" * 70)
print("RUN-LEVEL PERFORMANCE")
print("-" * 70)

print("Complete runs evaluated:",
      len(run_predictions))

print(f"Run-level accuracy: {run_accuracy:.4f}")

# ------------------------------------------------------------
# 6. PER-CLASS RUN PERFORMANCE
# ------------------------------------------------------------

print("\nRun-level classification report:")

print(
    classification_report(
        run_predictions["true_failure_type"],
        run_predictions["predicted_failure_type"],
        zero_division=0
    )
)

# ------------------------------------------------------------
# 7. CONFUSION MATRIX
# ------------------------------------------------------------

classes = sorted(
    set(run_predictions["true_failure_type"]) |
    set(run_predictions["predicted_failure_type"])
)

cm = confusion_matrix(
    run_predictions["true_failure_type"],
    run_predictions["predicted_failure_type"],
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

print("\nRun-level confusion matrix:")
print(cm_df.to_string())

# ------------------------------------------------------------
# 8. RUN-BY-RUN RESULTS
# ------------------------------------------------------------

run_predictions["correct"] = (
    run_predictions["true_failure_type"] ==
    run_predictions["predicted_failure_type"]
)

print("\n" + "-" * 70)
print("RUN-BY-RUN RESULTS")
print("-" * 70)

print(
    run_predictions[
        [
            "run",
            "true_failure_type",
            "predicted_failure_type",
            "n_samples",
            "correct"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 9. INCORRECT RUNS
# ------------------------------------------------------------

incorrect = run_predictions[
    ~run_predictions["correct"]
].copy()

print("\n" + "-" * 70)
print("INCORRECT RUNS")
print("-" * 70)

print("Incorrect complete runs:",
      len(incorrect))

if len(incorrect) > 0:
    print(
        incorrect[
            [
                "run",
                "true_failure_type",
                "predicted_failure_type"
            ]
        ].to_string(index=False)
    )
else:
    print("No incorrectly classified runs.")

# ------------------------------------------------------------
# 10. SAVE RUN-LEVEL RESULTS
# ------------------------------------------------------------

OUTPUT = "run_level_predictions.csv"

run_predictions.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 70)
print("STEP 13 COMPLETE")
print("=" * 70)
