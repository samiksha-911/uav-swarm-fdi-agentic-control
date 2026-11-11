import joblib
import re


MODEL_FILE = "uav_fdi_extratrees_model.joblib"


def main():

    bundle = joblib.load(MODEL_FILE)
    features = bundle["features"]

    print("=" * 70)
    print("STEP 39 — FDI FEATURE SOURCE MAPPING")
    print("=" * 70)

    # These are the known SEAD-style names.
    # Everything else in the selected 750 is FDI-derived.
    fdi_features = [
        f for f in features
        if not (
            f.startswith("ATT__")
            or f.startswith("IMU__")
            or f.startswith("MAG__")
            or f.startswith("GPS__")
            or f.startswith("BARO__")
            or f.startswith("RATE__")
            or f.startswith("XKF1__")
            or f.startswith("XKF2__")
            or f.startswith("XKF3__")
            or f.startswith("XKF4__")
            or f.startswith("XKF5__")
        )
    ]

    print()
    print("Total selected features:", len(features))
    print("FDI-derived selected features:", len(fdi_features))

    # Extract the original FDI signal and aggregation statistic.
    pairs = []

    for f in fdi_features:

        if "__" not in f:
            pairs.append((f, "UNKNOWN"))
            continue

        base, stat = f.rsplit("__", 1)

        pairs.append((base, stat))

    unique_bases = sorted(set(base for base, _ in pairs))

    print()
    print("Unique original FDI signals:", len(unique_bases))

    print()
    print("Original FDI signals:")
    print("-" * 70)

    for base in unique_bases:
        stats = sorted(
            stat
            for b, stat in pairs
            if b == base
        )

        print(f"{base:<30} {', '.join(stats)}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
