import joblib
import pandas as pd

MODEL = "uav_fdi_extratrees_model.joblib"
SEAD = "step22_sead_run_features.csv"
MERGED = "step31_sead_fdi_train_run_level.csv"


def main():
    bundle = joblib.load(MODEL)

    selected = bundle["features"]

    sead = pd.read_csv(SEAD, nrows=1)
    merged = pd.read_csv(MERGED, nrows=1)

    sead_features = set(sead.columns)
    merged_features = set(merged.columns)

    sead_only = []
    merged_only = []
    missing = []

    for f in selected:
        if f in sead_features:
            sead_only.append(f)
        elif f in merged_features:
            merged_only.append(f)
        else:
            missing.append(f)

    print("=" * 70)
    print("STEP 38 — FEATURE PROVENANCE CHECK")
    print("=" * 70)

    print()
    print("Total selected features:", len(selected))
    print("SEAD-source features:", len(sead_only))
    print("Merged/FDI-source features:", len(merged_only))
    print("Missing from both:", len(missing))

    print()
    print("SEAD-source examples:")
    for f in sead_only[:20]:
        print(" ", f)

    print()
    print("Merged/FDI-source examples:")
    for f in merged_only[:20]:
        print(" ", f)

    if missing:
        print()
        print("MISSING FEATURES:")
        for f in missing[:50]:
            print(" ", f)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
