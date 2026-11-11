import os
import subprocess
import pandas as pd

BASE = "/home/samiksha/uav_fdi_data"

OUTPUT = os.path.join(
    BASE,
    "step41C_px4_interface_inventory.csv"
)

print("=" * 70)
print("STEP 41C — PX4 ROS 2 INTERFACE INVENTORY")
print("=" * 70)

# ------------------------------------------------------------
# Get PX4 topics
# ------------------------------------------------------------

result = subprocess.run(
    ["ros2", "topic", "list"],
    capture_output=True,
    text=True,
    check=True
)

topics = [
    x.strip()
    for x in result.stdout.splitlines()
    if x.strip().startswith("/fmu/")
]

print(f"PX4 topics found: {len(topics)}")

rows = []

# ------------------------------------------------------------
# Inspect each topic
# ------------------------------------------------------------

for topic in topics:

    try:

        type_result = subprocess.run(
            ["ros2", "topic", "type", topic],
            capture_output=True,
            text=True,
            timeout=5
        )

        msg_type = type_result.stdout.strip()

        if not msg_type:
            continue

        interface_result = subprocess.run(
            ["ros2", "interface", "show", msg_type],
            capture_output=True,
            text=True,
            timeout=5
        )

        interface = interface_result.stdout.strip()

        rows.append({
            "topic": topic,
            "message_type": msg_type,
            "interface": interface
        })

    except Exception as e:

        rows.append({
            "topic": topic,
            "message_type": "",
            "interface": f"ERROR: {e}"
        })


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT,
    index=False
)

# ------------------------------------------------------------
# Print useful inventory
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PX4 TOPIC INVENTORY")
print("=" * 70)

for _, row in df.iterrows():

    print("\n" + "-" * 70)
    print("TOPIC:")
    print(row["topic"])

    print("TYPE:")
    print(row["message_type"])

    print("INTERFACE:")
    print(row["interface"])

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Topics inspected: {len(df)}")

print("\nSaved:")
print(OUTPUT)

print("\nDone.")
