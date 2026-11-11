import os
import numpy as np
import pandas as pd
from joblib import load


BASE = "/home/samiksha/uav_fdi_data"

MODEL_PATH = os.path.join(
    BASE,
    "px4_native_fdi_model_step50.joblib"
)

OUTPUT_PATH = os.path.join(
    BASE,
    "step51_feature_importance.csv"
)


print("=" * 70)
print("STEP 51 — STEP 50 FEATURE AUDIT")
print("=" * 70)


# ---------------------------------------------------------
# Load model bundle
# ---------------------------------------------------------

print("\nLoading Step 50 model...")

bundle = load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]

print("Model:", bundle["model_name"])
print("Number of model features:", len(features))
print("Classes:", bundle["classes"])


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

importance = model.feature_importances_

if len(importance) != len(features):
    raise RuntimeError(
        "Feature count does not match model importance count."
    )

audit = pd.DataFrame({
    "feature": features,
    "importance": importance,
})

audit = audit.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

audit["rank"] = np.arange(
    1,
    len(audit) + 1
)


# ---------------------------------------------------------
# Save complete audit
# ---------------------------------------------------------

audit.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nSaved complete feature audit:"
)

print(
    OUTPUT_PATH
)


# ---------------------------------------------------------
# Top 30 features
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 30 FEATURES")
print("=" * 70)

print(
    audit[
        [
            "rank",
            "feature",
            "importance",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Feature family analysis
# ---------------------------------------------------------

def classify_feature(name):

    if (
        "mag_heading_ok" in name
        or "mag_declination_ok" in name
        or "mag_fault" in name
    ):
        return "Magnetometer/status"

    temporal_tokens = [
        "_first",
        "_last",
        "_delta",
        "_mean_abs_change",
        "_max_abs_change",
        "_change_std",
        "_change_range",
        "_transition_count",
        "_transition_rate",
        "_invalid_fraction",
        "_valid_fraction",
        "_invalid_run_max",
        "_valid_run_max",
        "_invalid_duration_sec",
        "_first_state",
        "_last_state",
        "_true_fraction",
    ]

    if any(
        token in name
        for token in temporal_tokens
    ):
        return "Temporal/status"

    return "Original window statistics"


audit["feature_family"] = audit[
    "feature"
].apply(classify_feature)


family_summary = (
    audit
    .groupby("feature_family")["importance"]
    .agg(
        count="count",
        total_importance="sum",
        mean_importance="mean",
        max_importance="max",
    )
    .sort_values(
        "total_importance",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("FEATURE FAMILY IMPORTANCE")
print("=" * 70)

print(
    family_summary.to_string()
)


# ---------------------------------------------------------
# Top magnetometer/status features
# ---------------------------------------------------------

mag_audit = audit[
    audit["feature_family"]
    == "Magnetometer/status"
]

print("\n" + "=" * 70)
print("MAGNETOMETER / STATUS FEATURES")
print("=" * 70)

if len(mag_audit) == 0:

    print(
        "No magnetometer/status features found."
    )

else:

    print(
        mag_audit[
            [
                "rank",
                "feature",
                "importance",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Top temporal features
# ---------------------------------------------------------

temporal_audit = audit[
    audit["feature_family"]
    == "Temporal/status"
]

print("\n" + "=" * 70)
print("TOP TEMPORAL / STATUS FEATURES")
print("=" * 70)

print(
    temporal_audit[
        [
            "rank",
            "feature",
            "importance",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Importance concentration
# ---------------------------------------------------------

total = audit["importance"].sum()

print("\n" + "=" * 70)
print("IMPORTANCE CONCENTRATION")
print("=" * 70)

for n in [5, 10, 20, 30, 50]:

    value = (
        audit.head(n)["importance"].sum()
        / total
    )

    print(
        f"Top {n:2d}: "
        f"{value:.4f} "
        f"({value * 100:.2f}%)"
    )


print("\nDONE.")
