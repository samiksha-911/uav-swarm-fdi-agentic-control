import joblib

path = "uav_fdi_extratrees_model.joblib"
b = joblib.load(path)

features = b["features"]

print("MODEL BUNDLE CHECK")
print("------------------")
print("Bundle keys:", sorted(b.keys()))
print("Number of selected features:", len(features))
print()

print("First 50 selected features:")
for f in features[:50]:
    print(f)

print()
print("Feature naming pattern counts:")

patterns = {
    "SEAD_STYLE__": 0,
    "contains_double_underscore": 0,
    "other": 0,
}

for f in features:
    if "__" in f:
        patterns["contains_double_underscore"] += 1
        if f.count("__") == 2:
            patterns["SEAD_STYLE__"] += 1
    else:
        patterns["other"] += 1

print(patterns)
