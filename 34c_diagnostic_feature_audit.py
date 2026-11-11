#!/usr/bin/env python3

"""
STEP 34C
DIAGNOSTIC / POTENTIAL DIRECT-INDICATOR AUDIT

Purpose:
    Final audit of the strongest fault-discriminative telemetry
    features from all 28 streams.

This does NOT change the model or dataset.

It flags features whose names suggest:
    - status
    - health
    - error
    - fault
    - failure
    - flag
    - fix/state indicators
    - estimator diagnostic/error quantities

Important:
    This is a SCREENING audit based on feature names and statistics.
    A keyword match alone does NOT prove data leakage.
"""

from pathlib import Path
import pandas as pd


BASE = Path(
    "/home/samiksha/uav_fdi_data"
)

INPUT_FILE = (
    BASE /
    "step34B_fault_analysis" /
    "stream_column_discrimination.csv"
)

OUTPUT_FILE = (
    BASE /
    "step34B_fault_analysis" /
    "step34C_diagnostic_feature_audit.csv"
)

SUMMARY_FILE = (
    BASE /
    "step34B_fault_analysis" /
    "step34C_audit_summary.csv"
)


# ============================================================
# KEYWORD GROUPS
# ============================================================

DIAGNOSTIC_KEYWORDS = {

    "status": [
        "status",
        "state",
        "health",
    ],

    "fault": [
        "fault",
        "failure",
        "fail",
    ],

    "error": [
        "err",
        "error",
        "re",
    ],

    "flag": [
        "flag",
        "flags",
    ],

    "estimator_diagnostic": [
        "fix",
        "gps",
        "af",
        "fs",
        "fault",
    ],
}


def find_categories(feature_name):

    name = str(
        feature_name
    ).lower()

    matches = []

    for category, keywords in (
        DIAGNOSTIC_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in name:

                matches.append(
                    category
                )

                break

    return sorted(
        set(matches)
    )


def main():

    print("=" * 75)
    print(
        "STEP 34C — DIAGNOSTIC FEATURE AUDIT"
    )
    print("=" * 75)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        "Total analyzed features:",
        len(df)
    )

    # --------------------------------------------------------
    # FLAG FEATURES
    # --------------------------------------------------------

    df["diagnostic_categories"] = (
        df["column"]
        .apply(find_categories)
    )

    df["potential_diagnostic"] = (
        df["diagnostic_categories"]
        .apply(
            lambda x: len(x) > 0
        )
    )

    df["diagnostic_categories"] = (
        df["diagnostic_categories"]
        .apply(
            lambda x:
            " | ".join(x)
            if x else ""
        )
    )

    # --------------------------------------------------------
    # POTENTIAL DIAGNOSTIC FEATURES
    # --------------------------------------------------------

    flagged = (
        df[
            df["potential_diagnostic"]
        ]
        .sort_values(
            "eta_squared",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 75)
    print(
        "TOP POTENTIAL DIAGNOSTIC FEATURES"
    )
    print("=" * 75)

    columns_to_print = [
        "stream",
        "column",
        "eta_squared",
        "diagnostic_categories",
        "stream_rank",
    ]

    print(
        flagged[
            columns_to_print
        ]
        .head(50)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # TOP FEATURES WITHOUT DIAGNOSTIC KEYWORDS
    # --------------------------------------------------------

    non_diagnostic = (
        df[
            ~df["potential_diagnostic"]
        ]
        .sort_values(
            "eta_squared",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 75)
    print(
        "TOP NON-DIAGNOSTIC-NAMED FEATURES"
    )
    print("=" * 75)

    print(
        non_diagnostic[
            [
                "stream",
                "column",
                "eta_squared",
                "stream_rank",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # STREAM SUMMARY
    # --------------------------------------------------------

    stream_summary = (
        df
        .groupby("stream")
        .agg(
            total_features=(
                "column",
                "count"
            ),
            potential_diagnostic_features=(
                "potential_diagnostic",
                "sum"
            ),
            max_eta_squared=(
                "eta_squared",
                "max"
            ),
            mean_eta_squared=(
                "eta_squared",
                "mean"
            ),
        )
        .reset_index()
    )

    stream_summary[
        "diagnostic_feature_percentage"
    ] = (
        stream_summary[
            "potential_diagnostic_features"
        ]
        /
        stream_summary[
            "total_features"
        ]
        * 100
    )

    stream_summary = (
        stream_summary
        .sort_values(
            "potential_diagnostic_features",
            ascending=False
        )
    )

    print()
    print("=" * 75)
    print(
        "STREAM-LEVEL DIAGNOSTIC SUMMARY"
    )
    print("=" * 75)

    print(
        stream_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # ETA DISTRIBUTION OF FLAGGED VS NON-FLAGGED
    # --------------------------------------------------------

    flagged_eta_mean = (
        flagged["eta_squared"].mean()
        if not flagged.empty
        else 0.0
    )

    non_flagged_eta_mean = (
        non_diagnostic["eta_squared"].mean()
        if not non_diagnostic.empty
        else 0.0
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    flagged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary = pd.DataFrame([{

        "total_features":
            len(df),

        "potential_diagnostic_features":
            len(flagged),

        "non_diagnostic_named_features":
            len(non_diagnostic),

        "diagnostic_percentage":
            (
                len(flagged) /
                len(df) *
                100
            ),

        "mean_eta_diagnostic_features":
            flagged_eta_mean,

        "mean_eta_non_diagnostic_features":
            non_flagged_eta_mean,

    }])

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FINAL INTERPRETATION
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "FINAL INTERPRETATION"
    )
    print("=" * 75)

    print()

    print(
        "This audit is a SCREENING analysis."
    )

    print(
        "A keyword match does NOT by itself prove leakage."
    )

    print(
        "Health/status/error features may be legitimate UAV telemetry."
    )

    print(
        "They require domain-level interpretation before removal."
    )

    print()

    print(
        "Potential diagnostic features:",
        len(flagged)
    )

    print(
        "Percentage of analyzed features:",
        f"{len(flagged) / len(df) * 100:.2f}%"
    )

    print()

    print(
        "Saved:",
        OUTPUT_FILE
    )

    print(
        "Saved:",
        SUMMARY_FILE
    )

    print()
    print("=" * 75)
    print(
        "STEP 34C COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
