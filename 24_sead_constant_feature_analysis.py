import pandas as pd
from pathlib import Path

INPUT = "step22_sead_run_features.csv"
OUTPUT = "step24_sead_constant_feature_analysis.csv"

print("=" * 70)
print("STEP 24 — SEAD CONSTANT FEATURE ANALYSIS")
print("=" * 70)
print()
print("SEAD ONLY")
print("NO FDI MERGE")
print("NO TEST MODIFICATION")
print("NO MODEL TRAINING")
print()

# ------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------
print("1. LOADING SEAD FEATURE TABLE")
print("-" * 70)

if not Path(INPUT).exists():
    raise FileNotFoundError(f"Missing input file: {INPUT}")

df = pd.read_csv(INPUT)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ------------------------------------------------------------
# 2. IDENTIFY FEATURES
# ------------------------------------------------------------
metadata = {
    "run_key",
    "label",
    "available_required_streams",
    "total_generated_features",
}

feature_columns = [
    c for c in df.columns
    if c not in metadata
]

print()
print("2. FEATURE INVENTORY")
print("-" * 70)

print(f"Feature columns: {len(feature_columns)}")

# ------------------------------------------------------------
# 3. CONSTANT FEATURES
# ------------------------------------------------------------
print()
print("3. CONSTANT FEATURE ANALYSIS")
print("-" * 70)

nunique = df[feature_columns].nunique(dropna=False)

constant = nunique[nunique == 1].index.tolist()
non_constant = nunique[nunique > 1].index.tolist()

print(f"Constant features:     {len(constant)}")
print(f"Non-constant features: {len(non_constant)}")

# ------------------------------------------------------------
# 4. CONSTANT FEATURE VALUES
# ------------------------------------------------------------
print()
print("4. CONSTANT FEATURE VALUES")
print("-" * 70)

rows = []

for col in constant:
    value = df[col].iloc[0]

    rows.append({
        "feature": col,
        "unique_values": 1,
        "constant_value": value,
    })

for row in rows[:30]:
    print(
        f"{row['feature']:<60} "
        f"value={row['constant_value']}"
    )

if len(rows) > 30:
    print(f"... and {len(rows) - 30} more constant features")

# ------------------------------------------------------------
# 5. CLASS-WISE VARIATION
# ------------------------------------------------------------
print()
print("5. CLASS-WISE CONSTANT FEATURE CHECK")
print("-" * 70)

# A feature may be constant across all 70 runs,
# or vary between classes but remain constant within each class.

classwise_constant = []

for col in feature_columns:
    is_constant_within_every_class = True

    for label, group in df.groupby("label"):
        if group[col].nunique(dropna=False) > 1:
            is_constant_within_every_class = False
            break

    if is_constant_within_every_class:
        classwise_constant.append(col)

print(
    "Features constant within every individual class: "
    f"{len(classwise_constant)}"
)

print(
    "These may still contain class-level information and "
    "must NOT automatically be removed."
)

# ------------------------------------------------------------
# 6. BETWEEN-CLASS VARIATION
# ------------------------------------------------------------
print()
print("6. CLASS-LEVEL FEATURE VARIATION")
print("-" * 70)

class_level_features = []

for col in feature_columns:
    class_means = df.groupby("label")[col].mean()

    if class_means.nunique(dropna=False) > 1:
        class_level_features.append(col)

print(
    "Features with different class means: "
    f"{len(class_level_features)}"
)

# ------------------------------------------------------------
# 7. FINAL RECOMMENDATION
# ------------------------------------------------------------
print()
print("7. STEP 24 DECISION")
print("-" * 70)

print(f"Total features: {len(feature_columns)}")
print(f"Globally constant: {len(constant)}")
print(f"Usable non-constant: {len(non_constant)}")

print()
print("IMPORTANT:")
print("Globally constant features contain no variation across")
print("the 70 SEAD runs and are candidates for removal during")
print("feature preprocessing.")
print()
print("However, NO features will be removed from the original")
print("SEAD feature table at this step.")
print("The original step22_sead_run_features.csv remains unchanged.")

# ------------------------------------------------------------
# 8. SAVE ANALYSIS
# ------------------------------------------------------------
print()
print("8. SAVING RESULTS")
print("-" * 70)

result = pd.DataFrame(rows)

if result.empty:
    result = pd.DataFrame(
        columns=["feature", "unique_values", "constant_value"]
    )

result.to_csv(OUTPUT, index=False)

print(f"Saved: {OUTPUT}")

# ------------------------------------------------------------
# 9. COMPLETE
# ------------------------------------------------------------
print()
print("=" * 70)
print("STEP 24 COMPLETE")
print("=" * 70)
print()
print(f"SEAD runs analyzed: {len(df)}")
print(f"Total features: {len(feature_columns)}")
print(f"Constant features: {len(constant)}")
print(f"Non-constant features: {len(non_constant)}")
print()
print("NO FDI MERGE")
print("NO TEST MODIFICATION")
print("NO MODEL TRAINING")
print("ORIGINAL SEAD FEATURE TABLE NOT MODIFIED")

