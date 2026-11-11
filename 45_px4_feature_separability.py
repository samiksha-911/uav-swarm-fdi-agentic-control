import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder


BASE = Path("/home/samiksha/uav_fdi_data")

INPUT_FILE = BASE / "step44_px4_window_features.csv"

OUTPUT_FILE = BASE / "step45_px4_feature_separability.csv"


print("=" * 70)
print("STEP 45 — PX4 FEATURE SEPARABILITY ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nLoaded:")
print(INPUT_FILE)

print(f"\nRows/windows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# METADATA
# ============================================================

metadata = {
    "source_file",
    "window_id",
    "label",
}

feature_columns = [
    c for c in df.columns
    if c not in metadata
]


print(f"ML features: {len(feature_columns)}")


# ============================================================
# CHECK CLASSES
# ============================================================

print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .to_string()
)


# ============================================================
# PREPARE X / y
# ============================================================

X = df[feature_columns].copy()

# Convert everything to numeric.
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace invalid values.
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Median imputation only for this analysis.
X = X.fillna(
    X.median()
)

y_encoder = LabelEncoder()

y = y_encoder.fit_transform(
    df["label"]
)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

variances = X.var(axis=0)

nonconstant_features = [
    c
    for c in X.columns
    if variances[c] > 0
]

constant_features = [
    c
    for c in X.columns
    if variances[c] == 0
]

print(
    f"\nConstant features: "
    f"{len(constant_features)}"
)

print(
    f"Nonconstant features: "
    f"{len(nonconstant_features)}"
)

X_nc = X[nonconstant_features]


# ============================================================
# ANOVA F-TEST
# ============================================================

print("\nCalculating feature separability...")

F, p = f_classif(
    X_nc,
    y
)


# ============================================================
# BUILD RESULTS
# ============================================================

results = pd.DataFrame({
    "feature": nonconstant_features,
    "f_score": F,
    "p_value": p,
})


# ============================================================
# EFFECT SIZE: ETA SQUARED
# ============================================================

# For two-class comparison, eta-squared is:
#
# eta² = between-group SS / total SS
#
# This measures how much of the variance is associated
# with the class separation.

y_numeric = y.astype(float)

grand_mean = X_nc.mean(axis=0).to_numpy()

classes = np.unique(y)

between_ss = np.zeros(len(nonconstant_features))
total_ss = np.zeros(len(nonconstant_features))

for cls in classes:

    mask = y == cls

    group = X_nc.iloc[mask]

    n = len(group)

    if n == 0:
        continue

    group_mean = group.mean(axis=0).to_numpy()

    between_ss += (
        n * (group_mean - grand_mean) ** 2
    )

total_ss = (
    (
        X_nc
        - X_nc.mean(axis=0)
    ) ** 2
).sum(axis=0).to_numpy()

eta_squared = np.divide(
    between_ss,
    total_ss,
    out=np.zeros_like(between_ss),
    where=total_ss > 0
)

results["eta_squared"] = eta_squared


# ============================================================
# SORT
# ============================================================

results = results.sort_values(
    "eta_squared",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nTop 30 features by eta-squared:")

print(
    results.head(30).to_string(
        index=False
    )
)


print("\nEta-squared summary:")

print(
    results["eta_squared"].describe().to_string()
)


# ============================================================
# SAVE TOP FEATURES
# ============================================================

print("\nSaved:")
print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("STEP 45 COMPLETE")
print("=" * 70)
