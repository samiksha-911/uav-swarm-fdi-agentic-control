import joblib
import pandas as pd


MODEL_FILE = "uav_fdi_extratrees_model.joblib"
MERGED_FILE = "step31_sead_fdi_train_run_level.csv"
OUTPUT_FILE = "step40_fdi_signal_inventory.csv"


SEAD_PREFIXES = (
    "ATT__",
    "IMU__",
    "MAG__",
    "GPS__",
    "BARO__",
    "RATE__",
    "XKF1__",
    "XKF2__",
    "XKF3__",
    "XKF4__",
    "XKF5__",
)


def main():

    bundle = joblib.load(MODEL_FILE)

    selected = bundle["features"]

    # Identify the 298 non-SEAD-style selected features.
    fdi_features = [
        f for f in selected
        if not f.startswith(SEAD_PREFIXES)
    ]

    # Recover original signal + aggregation statistic.
    rows = []

    for feature in fdi_features:

        if "__" not in feature:
            continue

        signal, statistic = feature.rsplit("__", 1)

        rows.append({
            "selected_feature": feature,
            "original_signal": signal,
            "aggregation": statistic,
        })

    df = pd.DataFrame(rows)

    # One row per original source signal.
    inventory = (
        df.groupby("original_signal")["aggregation"]
        .apply(lambda x: ", ".join(sorted(set(x))))
        .reset_index()
    )

    inventory["n_selected_features"] = (
        df.groupby("original_signal")
        .size()
        .values
    )

    inventory = inventory.sort_values(
        "original_signal"
    ).reset_index(drop=True)

    inventory.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 70)
    print("STEP 40 — COMPLETE FDI SIGNAL INVENTORY")
    print("=" * 70)

    print()
    print("Selected FDI-derived features:", len(fdi_features))
    print("Unique original FDI signals:", len(inventory))
    print()

    print(inventory.to_string(index=False))

    print()
    print("Saved:", OUTPUT_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()
