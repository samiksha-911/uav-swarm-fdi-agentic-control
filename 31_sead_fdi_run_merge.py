import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 31 — SEAD + FDI RUN-LEVEL INTEGRATION")
print("=" * 70)
print()
print("RUN-LEVEL INTEGRATION")
print("SEAD FEATURES + FDI RUN-LEVEL FEATURES")
print("NO ORIGINAL DATA MODIFICATION")
print("NO MODEL TRAINING")
print()

SEAD_FILE = "step22_sead_run_features.csv"
FDI_TRAIN = "fdi_train.csv"
FDI_TEST = "fdi_test.csv"

OUT_TRAIN = "step31_sead_fdi_train_run_level.csv"
OUT_TEST = "step31_sead_fdi_test_run_level.csv"

# ------------------------------------------------------------
# 1. LOAD SEAD
# ------------------------------------------------------------
print("1. LOADING SEAD")
print("-" * 70)

sead = pd.read_csv(SEAD_FILE)

print(f"SEAD rows: {len(sead)}")
print(f"SEAD columns: {len(sead.columns)}")

if "run_key" not in sead.columns:
    raise RuntimeError("SEAD run_key column not found.")

if "label" not in sead.columns:
    raise RuntimeError("SEAD label column not found.")

if sead["run_key"].duplicated().any():
    raise RuntimeError("Duplicate SEAD run keys detected.")

# ------------------------------------------------------------
# 2. FDI RUN-LEVEL AGGREGATION
# ------------------------------------------------------------
def build_fdi_run_table(filename, name):

    print()
    print(f"2. PROCESSING {name}")
    print("-" * 70)

    # Read in chunks because the files are large.
    chunks = []

    usecols = None

    for chunk_no, chunk in enumerate(
        pd.read_csv(filename, chunksize=100000),
        start=1
    ):
        print(f"  reading chunk {chunk_no}: {len(chunk)} rows")

        required = {"run", "failure_type", "sample_index"}

        missing = required - set(chunk.columns)
        if missing:
            raise RuntimeError(
                f"{filename}: missing required columns: {sorted(missing)}"
            )

        chunks.append(chunk)

    fdi = pd.concat(chunks, ignore_index=True)

    print(f"  total samples: {len(fdi)}")
    print(f"  unique runs: {fdi['run'].nunique()}")

    # --------------------------------------------------------
    # Validate failure type consistency inside each run
    # --------------------------------------------------------
    failure_counts = (
        fdi.groupby("run")["failure_type"]
        .nunique(dropna=False)
    )

    inconsistent = failure_counts[failure_counts > 1]

    if len(inconsistent):
        print()
        print("WARNING: runs with multiple failure_type values:")
        print(inconsistent)
        raise RuntimeError(
            "Failure type is inconsistent within one or more FDI runs."
        )

    # --------------------------------------------------------
    # Columns that should NOT become numerical aggregates
    # --------------------------------------------------------
    exclude = {
        "run",
        "failure_type",
        "sample_index",
        "relative_to_failure",
    }

    feature_cols = [
        c for c in fdi.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(fdi[c])
    ]

    print(f"  numeric sample features: {len(feature_cols)}")

    # --------------------------------------------------------
    # Run-level statistics
    # --------------------------------------------------------
    print("  building run-level statistics...")

    grouped = fdi.groupby("run", sort=False)

    mean_df = grouped[feature_cols].mean().add_suffix("__mean")
    std_df = grouped[feature_cols].std(ddof=0).fillna(0).add_suffix("__std")
    min_df = grouped[feature_cols].min().add_suffix("__min")
    max_df = grouped[feature_cols].max().add_suffix("__max")

    count_df = grouped.size().rename("fdi_sample_count").to_frame()

    failure_df = (
        grouped["failure_type"]
        .first()
        .rename("fdi_failure_type")
        .to_frame()
    )

    result = pd.concat(
        [
            failure_df,
            count_df,
            mean_df,
            std_df,
            min_df,
            max_df,
        ],
        axis=1
    ).reset_index()

    print(f"  run-level rows: {len(result)}")
    print(f"  run-level columns: {len(result.columns)}")

    # --------------------------------------------------------
    # Basic quality checks
    # --------------------------------------------------------
    numeric = result.select_dtypes(include=[np.number])

    missing = int(numeric.isna().sum().sum())
    infinite = int(
        np.isinf(numeric.to_numpy()).sum()
    )

    print(f"  missing numeric values: {missing}")
    print(f"  infinite numeric values: {infinite}")

    if missing:
        raise RuntimeError(
            f"{name}: missing numeric values detected."
        )

    if infinite:
        raise RuntimeError(
            f"{name}: infinite numeric values detected."
        )

    return result


train_fdi = build_fdi_run_table(
    FDI_TRAIN,
    "FDI TRAIN"
)

test_fdi = build_fdi_run_table(
    FDI_TEST,
    "FDI TEST"
)

# ------------------------------------------------------------
# 3. RUN OVERLAP
# ------------------------------------------------------------
print()
print("3. RUN-KEY OVERLAP")
print("-" * 70)

sead_runs = set(sead["run_key"])

train_runs = set(train_fdi["run"])
test_runs = set(test_fdi["run"])

print(f"SEAD runs:       {len(sead_runs)}")
print(f"FDI train runs:  {len(train_runs)}")
print(f"FDI test runs:   {len(test_runs)}")

print(f"SEAD ∩ FDI train: {len(sead_runs & train_runs)}")
print(f"SEAD ∩ FDI test:  {len(sead_runs & test_runs)}")

missing_train = train_runs - sead_runs
missing_test = test_runs - sead_runs

if missing_train:
    print()
    print("FDI train runs missing from SEAD:")
    for x in sorted(missing_train):
        print(" ", x)

if missing_test:
    print()
    print("FDI test runs missing from SEAD:")
    for x in sorted(missing_test):
        print(" ", x)

# We only merge matching runs.
train_fdi = train_fdi[train_fdi["run"].isin(sead_runs)].copy()
test_fdi = test_fdi[test_fdi["run"].isin(sead_runs)].copy()

# ------------------------------------------------------------
# 4. MERGE
# ------------------------------------------------------------
print()
print("4. MERGING RUN-LEVEL TABLES")
print("-" * 70)

train = train_fdi.merge(
    sead,
    left_on="run",
    right_on="run_key",
    how="inner",
    validate="one_to_one",
    suffixes=("_fdi", "_sead")
)

test = test_fdi.merge(
    sead,
    left_on="run",
    right_on="run_key",
    how="inner",
    validate="one_to_one",
    suffixes=("_fdi", "_sead")
)

print(f"Combined train runs: {len(train)}")
print(f"Combined test runs:  {len(test)}")
print(f"Combined train columns: {len(train.columns)}")
print(f"Combined test columns:  {len(test.columns)}")

# ------------------------------------------------------------
# 5. LABEL CHECK
# ------------------------------------------------------------
print()
print("5. LABEL CHECK")
print("-" * 70)

print("FDI TRAIN labels:")
print(train["fdi_failure_type"].value_counts().to_string())

print()
print("SEAD labels:")
print(train["label"].value_counts().to_string())

# ------------------------------------------------------------
# 6. CHECK SEAD LABEL AGAINST FDI LABEL
# ------------------------------------------------------------
print()
print("6. SEAD/FDI LABEL AGREEMENT")
print("-" * 70)

agreement = (
    train["fdi_failure_type"].astype(str)
    == train["label"].astype(str)
)

print(f"Matching labels: {agreement.sum()}/{len(agreement)}")
print(f"Agreement rate: {agreement.mean():.4f}")

if not agreement.all():
    print()
    print("MISMATCHED RUNS:")
    print(
        train.loc[
            ~agreement,
            ["run", "fdi_failure_type", "label"]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------
print()
print("7. SAVING")
print("-" * 70)

train.to_csv(OUT_TRAIN, index=False)
test.to_csv(OUT_TEST, index=False)

print(f"Saved: {OUT_TRAIN}")
print(f"Saved: {OUT_TEST}")

# ------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------
print()
print("=" * 70)
print("STEP 31 COMPLETE")
print("=" * 70)
print()
print(f"SEAD source runs:       {len(sead)}")
print(f"FDI train runs matched: {len(train)}")
print(f"FDI test runs matched:  {len(test)}")
print()
print("Original files were NOT modified:")
print("  fdi_train.csv")
print("  fdi_test.csv")
print("  step22_sead_run_features.csv")
print()
print("No classifier was trained.")
print("SEAD features were joined at RUN LEVEL, not sample level.")
print("=" * 70)

