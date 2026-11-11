import pandas as pd

INPUT = "/home/samiksha/uav_fdi_data/fdi_training_windows.csv"

TRAIN_OUTPUT = "/home/samiksha/uav_fdi_data/fdi_train_specified.csv"
TEST_OUTPUT = "/home/samiksha/uav_fdi_data/fdi_test_specified.csv"

print("=" * 70)
print("SPECIFIED RUN-LEVEL TRAIN / TEST SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT)

print("Full dataset:", df.shape)
print("Unique runs:", df["run"].nunique())

# =========================================================
# EXACTLY 2 TEST RUNS PER CLASS
# =========================================================

TEST_RUNS = {
    # No Failure
    "2022-07-26 05-49-12 (No Failure)",
    "2022-07-27 07-43-42 (No Failure)",

    # RC Failure
    "2022-07-26 06-25-08 (RC Failure)",
    "2022-07-27 16-13-54 (RC Failure)",

    # GPS Failure
    "2022-07-26 06-46-30 (GPS Failure)",
    "2022-07-27 16-20-44 (GPS Failure)",

    # Accelerometer Failure
    "2022-07-26 17-02-53 (Accelerometer Failure)",
    "2022-07-27 16-29-23 (Accelerometer Failure)",

    # Gyro Failure
    "2022-07-26 18-11-42 (Gyro Failure)",
    "2022-07-27 16-45-02 (Gyro Failure)",

    # Compass Failure
    "2022-07-26 18-26-44 (Compass Failure)",
    "2022-07-27 17-05-17 (Compass Failure)",

    # Barometer Failure
    "2022-07-26 18-37-31 (Barometer Failure)",
    "2022-07-27 17-16-34 (Barometer Failure)",
}

# =========================================================
# VERIFY SPECIFIED RUNS EXIST
# =========================================================

available_runs = set(df["run"].unique())

missing_runs = TEST_RUNS - available_runs

if missing_runs:
    print()
    print("ERROR: These specified runs were not found:")
    for run in sorted(missing_runs):
        print(" ", run)
    raise SystemExit(1)

print()
print("Specified test runs found:", len(TEST_RUNS))

# =========================================================
# SPLIT BY COMPLETE RUN
# =========================================================

test = df[df["run"].isin(TEST_RUNS)].copy()
train = df[~df["run"].isin(TEST_RUNS)].copy()

# =========================================================
# VERIFY NO RUN LEAKAGE
# =========================================================

train_runs = set(train["run"].unique())
test_runs = set(test["run"].unique())

overlap = train_runs.intersection(test_runs)

print("Training runs:", len(train_runs))
print("Testing runs :", len(test_runs))
print("Run overlap  :", len(overlap))

if overlap:
    print()
    print("ERROR: Run leakage detected!")
    for run in sorted(overlap):
        print(" ", run)
    raise SystemExit(1)

# =========================================================
# DISTRIBUTIONS
# =========================================================

print()
print("=" * 70)
print("TRAIN DISTRIBUTION")
print("=" * 70)

print("Shape:", train.shape)

print()
print("Status:")
print(train["Status"].value_counts().sort_index())

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
print(test["Status"].value_counts().sort_index())

print()
print("Failure type:")
print(test["failure_type"].value_counts())

# =========================================================
# SAVE
# =========================================================

train.to_csv(TRAIN_OUTPUT, index=False)
test.to_csv(TEST_OUTPUT, index=False)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(TRAIN_OUTPUT)
print(TEST_OUTPUT)

print()
print("=" * 70)
print("SPECIFIED SPLIT COMPLETE")
print("=" * 70)
