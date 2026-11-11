import pandas as pd
import numpy as np

INPUT = "step22_sead_run_features.csv"
OUTPUT = "step26_sead_leakage_audit.csv"

print("=" * 70)
print("STEP 26 — SEAD FEATURE LEAKAGE / LABEL-PROXY AUDIT")
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
df = pd.read_csv(INPUT)

label_col = "label"
run_col = "run_key"

if label_col not in df.columns:
    raise RuntimeError("Missing label column.")
if run_col not in df.columns:
    raise RuntimeError("Missing run_key column.")

metadata = {
    run_col,
    label_col,
    "available_required_streams",
    "total_generated_features",
}

features = [c for c in df.columns if c not in metadata]

print("1. LOADING FEATURE TABLE")
print("-" * 70)
print(f"Rows: {len(df)}")
print(f"Feature columns: {len(features)}")
print()

# ------------------------------------------------------------
# 2. CORRECT ETA-SQUARED
# ------------------------------------------------------------
print("2. CORRECT ETA-SQUARED CALCULATION")
print("-" * 70)

classes = df[label_col].dropna().unique()
results = []

for i, feature in enumerate(features, 1):
    x = pd.to_numeric(df[feature], errors="coerce")

    valid = x.notna()
    x = x[valid]
    y = df.loc[valid, label_col]

    if len(x) == 0 or x.nunique(dropna=True) <= 1:
        continue

    grand_mean = x.mean()

    ss_between = 0.0
    ss_total = ((x - grand_mean) ** 2).sum()

    class_means = {}

    for cls in classes:
        vals = x[y == cls]
        if len(vals) == 0:
            continue

        m = vals.mean()
        class_means[str(cls)] = m
        ss_between += len(vals) * (m - grand_mean) ** 2

    eta2 = ss_between / ss_total if ss_total > 0 else 0.0

    # Numerical safety: valid eta² must lie in [0, 1]
    eta2 = max(0.0, min(1.0, eta2))

    means = list(class_means.values())

    class_mean_range = (
        max(means) - min(means)
        if means else np.nan
    )

    # Detect exact class separation:
    # each class has a different single-valued feature value
    class_values = []
    all_single_valued = True

    for cls in classes:
        vals = x[y == cls]
        if len(vals) == 0 or vals.nunique() != 1:
            all_single_valued = False
            break
        class_values.append(vals.iloc[0])

    unique_class_values = len(set(class_values))

    label_proxy = (
        all_single_valued
        and unique_class_values == len(classes)
    )

    results.append({
        "feature": feature,
        "eta_squared": eta2,
        "class_mean_range": class_mean_range,
        "label_proxy_candidate": label_proxy,
        "n_unique": x.nunique(),
    })

    if i % 250 == 0 or i == len(features):
        print(f"  processed {i}/{len(features)} features")

audit = pd.DataFrame(results)

audit = audit.sort_values(
    ["label_proxy_candidate", "eta_squared"],
    ascending=[False, False]
).reset_index(drop=True)

print()

# ------------------------------------------------------------
# 3. VALIDATE ETA
# ------------------------------------------------------------
print("3. ETA-SQUARED SANITY CHECK")
print("-" * 70)

print(
    f"Minimum η²: {audit['eta_squared'].min():.6f}"
)
print(
    f"Median η²:  {audit['eta_squared'].median():.6f}"
)
print(
    f"Maximum η²: {audit['eta_squared'].max():.6f}"
)

bad_eta = audit[
    (audit["eta_squared"] < -1e-12) |
    (audit["eta_squared"] > 1 + 1e-12)
]

print(f"Invalid η² values: {len(bad_eta)}")
print()

# ------------------------------------------------------------
# 4. LABEL-PROXY AUDIT
# ------------------------------------------------------------
proxy = audit[audit["label_proxy_candidate"]]

print("4. LABEL-PROXY CANDIDATES")
print("-" * 70)
print(f"Exact class-coded feature candidates: {len(proxy)}")
print()

if len(proxy):
    print(proxy[
        ["feature", "eta_squared", "class_mean_range", "n_unique"]
    ].head(50).to_string(index=False))
else:
    print("No exact class-coded feature candidates found.")

print()

# ------------------------------------------------------------
# 5. TOP FEATURES
# ------------------------------------------------------------
print("5. TOP FEATURES BY CORRECTED ETA-SQUARED")
print("-" * 70)

top = audit.sort_values(
    "eta_squared",
    ascending=False
).head(30)

print(
    top[
        ["feature", "eta_squared",
         "class_mean_range", "label_proxy_candidate"]
    ].to_string(index=False)
)

print()

# ------------------------------------------------------------
# 6. STATUS / HEALTH / ERROR FEATURES
# ------------------------------------------------------------
print("6. POTENTIAL STATUS / HEALTH / ERROR FEATURES")
print("-" * 70)

keywords = [
    "health",
    "status",
    "err",
    "gps",
    "fail",
    "fault",
    "valid",
    "good",
    "quality",
    "sat",
]

mask = audit["feature"].str.lower().apply(
    lambda s: any(k in s for k in keywords)
)

status_df = audit[mask].sort_values(
    "eta_squared",
    ascending=False
)

print(f"Potential status/proxy-related features: {len(status_df)}")
print()

print(
    status_df[
        ["feature", "eta_squared",
         "class_mean_range", "label_proxy_candidate"]
    ].head(50).to_string(index=False)
)

print()

# ------------------------------------------------------------
# 7. STREAM SUMMARY
# ------------------------------------------------------------
print("7. STREAM-LEVEL SUMMARY")
print("-" * 70)

audit["stream"] = audit["feature"].str.split("__").str[0]

stream_summary = (
    audit.groupby("stream")
    .agg(
        features=("feature", "count"),
        mean_eta=("eta_squared", "mean"),
        max_eta=("eta_squared", "max"),
        proxy_candidates=("label_proxy_candidate", "sum"),
    )
    .sort_values("max_eta", ascending=False)
)

print(stream_summary.to_string())

print()

# ------------------------------------------------------------
# 8. SAVE
# ------------------------------------------------------------
audit.to_csv(OUTPUT, index=False)

print("8. SAVING RESULTS")
print("-" * 70)
print(f"Saved: {OUTPUT}")
print()

print("=" * 70)
print("STEP 26 RESULT")
print("=" * 70)
print()
print(f"SEAD runs: {len(df)}")
print(f"Features audited: {len(features)}")
print(f"Invalid η² values: {len(bad_eta)}")
print(f"Exact label-proxy candidates: {len(proxy)}")
print()
print("Original step22_sead_run_features.csv was NOT modified.")
print("No FDI data was merged.")
print("No classifier was trained.")
print("No features were removed.")
print()
print("=" * 70)
print("STEP 26 COMPLETE")
print("=" * 70)

