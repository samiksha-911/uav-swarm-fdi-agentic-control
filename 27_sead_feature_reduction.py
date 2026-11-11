import pandas as pd
import numpy as np

INPUT = "step22_sead_run_features.csv"
AUDIT = "step26_sead_leakage_audit.csv"

REDUCED = "step27_sead_nonconstant_features.csv"
RANKED = "step27_sead_ranked_features.csv"
SUMMARY = "step27_sead_feature_reduction_summary.csv"

print("=" * 70)
print("STEP 27 — SEAD FEATURE REDUCTION")
print("=" * 70)
print()
print("SEAD ONLY")
print("NO FDI MERGE")
print("NO TEST MODIFICATION")
print("NO MODEL TRAINING")
print()

df = pd.read_csv(INPUT)
audit = pd.read_csv(AUDIT)

metadata = {
    "run_key",
    "label",
    "available_required_streams",
    "total_generated_features",
}

features = [c for c in df.columns if c not in metadata]

print("1. LOADING")
print("-" * 70)
print(f"Rows: {len(df)}")
print(f"Original features: {len(features)}")
print()

# ------------------------------------------------------------
# CONSTANT FEATURES
# ------------------------------------------------------------
print("2. REMOVING ONLY GLOBALLY CONSTANT FEATURES")
print("-" * 70)

constant = []

for c in features:
    if df[c].nunique(dropna=False) <= 1:
        constant.append(c)

nonconstant = [c for c in features if c not in constant]

print(f"Constant features: {len(constant)}")
print(f"Non-constant features: {len(nonconstant)}")
print()

# ------------------------------------------------------------
# RANK USING CORRECTED ETA
# ------------------------------------------------------------
print("3. RANKING NON-CONSTANT FEATURES")
print("-" * 70)

ranked = audit[
    audit["feature"].isin(nonconstant)
].copy()

ranked = ranked.sort_values(
    "eta_squared",
    ascending=False
).reset_index(drop=True)

ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

print(f"Ranked features: {len(ranked)}")
print()

# ------------------------------------------------------------
# REDUCED TABLE
# ------------------------------------------------------------
print("4. BUILDING REDUCED SEAD TABLE")
print("-" * 70)

keep_columns = [
    "run_key",
    "label",
    "available_required_streams",
    "total_generated_features",
] + nonconstant

reduced = df[keep_columns].copy()

print(f"Reduced rows: {len(reduced)}")
print(f"Reduced columns: {len(reduced.columns)}")
print(f"Reduced feature columns: {len(nonconstant)}")
print()

# ------------------------------------------------------------
# QUALITY CHECK
# ------------------------------------------------------------
print("5. QUALITY CHECK")
print("-" * 70)

feature_matrix = reduced[nonconstant]

missing = int(feature_matrix.isna().sum().sum())
infinite = int(
    np.isinf(feature_matrix.select_dtypes(include=[np.number])).sum().sum()
)

print(f"Missing feature values: {missing}")
print(f"Infinite feature values: {infinite}")
print()

if missing != 0:
    raise RuntimeError("Reduced table contains missing values.")

if infinite != 0:
    raise RuntimeError("Reduced table contains infinite values.")

# ------------------------------------------------------------
# CLASS CHECK
# ------------------------------------------------------------
print("6. CLASS CHECK")
print("-" * 70)

print(
    reduced["label"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

# ------------------------------------------------------------
# TOP-K SUMMARY
# ------------------------------------------------------------
print("7. FEATURE COUNT SUMMARY")
print("-" * 70)

summary_rows = []

for k in [25, 50, 100, 200, 300, 500, 750, 1000, 1154]:
    k = min(k, len(ranked))

    summary_rows.append({
        "top_k": k,
        "features_retained": k,
        "eta_min": ranked.iloc[:k]["eta_squared"].min(),
        "eta_median": ranked.iloc[:k]["eta_squared"].median(),
        "eta_max": ranked.iloc[:k]["eta_squared"].max(),
    })

summary = pd.DataFrame(summary_rows)

print(summary.to_string(index=False))
print()

# ------------------------------------------------------------
# STREAM COUNTS IN TOP 100
# ------------------------------------------------------------
print("8. TOP-100 STREAM DISTRIBUTION")
print("-" * 70)

top100 = ranked.head(min(100, len(ranked))).copy()
top100["stream"] = top100["feature"].str.split("__").str[0]

print(
    top100["stream"]
    .value_counts()
    .to_string()
)

print()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
print("9. SAVING RESULTS")
print("-" * 70)

reduced.to_csv(REDUCED, index=False)
ranked.to_csv(RANKED, index=False)
summary.to_csv(SUMMARY, index=False)

print(f"Saved: {REDUCED}")
print(f"Saved: {RANKED}")
print(f"Saved: {SUMMARY}")
print()

print("=" * 70)
print("STEP 27 COMPLETE")
print("=" * 70)
print()
print(f"Original features:       {len(features)}")
print(f"Constant removed:        {len(constant)}")
print(f"Non-constant features:   {len(nonconstant)}")
print()
print("Original step22 table was NOT modified.")
print("No FDI data was merged.")
print("No classifier was trained.")
print()

