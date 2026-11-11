import os
import pandas as pd

BASE = "/home/samiksha/uav_fdi_data"

INPUT = os.path.join(
    BASE,
    "step41C_px4_interface_inventory.csv"
)

OUTPUT = os.path.join(
    BASE,
    "step41D_px4_output_fields.csv"
)

df = pd.read_csv(INPUT)

rows = []

for _, row in df.iterrows():

    topic = str(row["topic"])
    msg_type = str(row["message_type"])
    interface = str(row["interface"])

    # Only PX4 published telemetry.
    if not topic.startswith("/fmu/out/"):
        continue

    current_section = ""

    for line in interface.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        # Ignore constants.
        if "=" in line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        field_type = parts[0]
        field_name = parts[1]

        # Remove array declarations from field name.
        field_name = field_name.split("[")[0]

        if not field_name:
            continue

        rows.append({
            "topic": topic,
            "message_type": msg_type,
            "field_type": field_type,
            "field": field_name
        })


result = pd.DataFrame(rows)

result = result.drop_duplicates()

result = result.sort_values(
    ["topic", "field"]
).reset_index(drop=True)

result.to_csv(
    OUTPUT,
    index=False
)

print("=" * 70)
print("STEP 41D — PX4 OUTPUT TELEMETRY INVENTORY")
print("=" * 70)

print(f"PX4 output topics: {df['topic'].astype(str).str.startswith('/fmu/out/').sum()}")
print(f"Telemetry fields: {len(result)}")
print(f"Unique topics: {result['topic'].nunique()}")

print("\nFields by topic:")

counts = (
    result.groupby("topic")
    .size()
    .sort_values(ascending=False)
)

for topic, count in counts.items():
    print(f"{topic:55s} {count:4d}")

print("\nSaved:")
print(OUTPUT)

print("\nDone.")
