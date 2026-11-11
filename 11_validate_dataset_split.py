import os
import pandas as pd

BASE_DIR = "/home/samiksha/uav_fdi_data"

TRAIN_FILE = os.path.join(BASE_DIR, "fdi_train.csv")
TEST_FILE = os.path.join(BASE_DIR, "fdi_test.csv")

print("=" * 70)
print("STEP 11 - DATASET / TRAIN-TEST SPLIT VALIDATION")
print("=" * 70)

print("\nLoading labels only...")

train = pd.read_csv(
    TRAIN_FILE,
    usecols=["failure_type"]
)

test = pd.read_csv(
    TEST_FILE,
    usecols=["failure_type"]
)

print("\nTrain shape:", train.shape)
print("Test shape :", test.shape)

print("\n" + "=" * 70)
print("TRAIN CLASS DISTRIBUTION")
print("=" * 70)

train_counts = train["failure_type"].value_counts().sort_index()

print(train_counts)

print("\n" + "=" * 70)
print("TEST CLASS DISTRIBUTION")
print("=" * 70)

test_counts = test["failure_type"].value_counts().sort_index()

print(test_counts)

print("\n" + "=" * 70)
print("COMBINED DISTRIBUTION")
print("=" * 70)

combined = pd.concat(
    [
        train["failure_type"],
        test["failure_type"]
    ],
    ignore_index=True
)

combined_counts = combined.value_counts().sort_index()

print(combined_counts)

print("\n" + "=" * 70)
print("TRAIN / TEST PERCENTAGES")
print("=" * 70)

classes = sorted(combined.unique())

for cls in classes:

    train_n = train_counts.get(cls, 0)
    test_n = test_counts.get(cls, 0)
    total_n = combined_counts.get(cls, 0)

    train_pct = (
        train_n / total_n * 100
        if total_n > 0 else 0
    )

    test_pct = (
        test_n / total_n * 100
        if total_n > 0 else 0
    )

    print(
        f"{cls:25s} "
        f"Train={train_n:8,d} ({train_pct:6.2f}%)   "
        f"Test={test_n:8,d} ({test_pct:6.2f}%)"
    )

print("\n" + "=" * 70)
print("CHECKS")
print("=" * 70)

train_classes = set(train["failure_type"].unique())
test_classes = set(test["failure_type"].unique())

missing_from_test = sorted(
    train_classes - test_classes
)

missing_from_train = sorted(
    test_classes - train_classes
)

print("\nClasses in train but NOT test:")
print(missing_from_test if missing_from_test else "None")

print("\nClasses in test but NOT train:")
print(missing_from_train if missing_from_train else "None")

print("\nTotal train rows:", len(train))
print("Total test rows :", len(test))
print("Combined rows   :", len(combined))

print("\n" + "=" * 70)
print("STEP 11 COMPLETE")
print("=" * 70)

