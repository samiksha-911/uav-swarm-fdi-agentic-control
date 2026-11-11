import pandas as pd
import numpy as np

INPUT = "/home/samiksha/uav_fdi_data/fdi_training_windows.csv"

TRAIN_OUTPUT = "/home/samiksha/uav_fdi_data/fdi_train.csv"
TEST_OUTPUT  = "/home/samiksha/uav_fdi_data/fdi_test.csv"

print("=" * 70)
print("RUN-LEVEL TRAIN / TEST SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT)

print("Full dataset:", df.shape)
print("Unique runs:", df["run"].nunique())

# ---------------------------------------------------------
# Get unique runs
# ---------------------------------------------------------

unique_runs = df["run"].unique()

rng = np.random.default_rng(42)

shuffled_runs = unique_runs.copy()
rng.shuffle(shuffled_runs)

# 80% train / 20% test
n_test = round(len(shuffled_runs) * 0.20)

test_runs = set(shuffled_runs[:n_test])
train_runs = set(shuffled_runs[n_test:])

# ---------------------------------------------------------
# Split rows according to run
# ---------------------------------------------------------

train = df[df["run"].isin(train_runs)].copy()
test = df[df["run"].isin(test_runs)].copy()

# ---------------------------------------------------------
# Verify no run leakage
# ---------------------------------------------------------

overlap = train_runs.intersection(test_runs)

print()
print("Training runs:", len(train_runs))
print("Testing runs :", len(test_runs))
print("Run overlap  :", len(overlap))

if overlap:
    raise RuntimeError("ERROR: run leakage detected!")

# ---------------------------------------------------------
# Display distributions
# ---------------------------------------------------------

print()
print("=" * 70)
print("TRAIN DISTRIBUTION")
print("=" * 70)

print("Shape:", train.shape)

print()
print("Status:")
print(train["Status"].value_counts())

print()
print("Failure type:")
print(train["failure_type"].value_counts())

print()
print("=" * 70)
print("TEST DISTRIBUTION")
print("=" * 70)

print("Shape:", test.shape)

print()
print("Status:")
print(test["Status"].value_counts())

print()
print("Failure type:")
print(test["failure_type"].value_counts())

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

train.to_csv(TRAIN_OUTPUT, index=False)
test.to_csv(TEST_OUTPUT, index=False)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(TRAIN_OUTPUT)
print(TEST_OUTPUT)
