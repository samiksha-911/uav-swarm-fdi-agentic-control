import subprocess
import pandas as pd
import re
from pathlib import Path

BASE = Path.home() / "uav_fdi_data"
FDI_FILE = BASE / "fdi_train.csv"
OUTPUT = BASE / "step41_live_ros2_signal_mapping.csv"


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


print("=" * 70)
print("STEP 41 — LIVE PX4 ROS 2 SIGNAL AVAILABILITY MAPPING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load original FDI signal names
# ------------------------------------------------------------
df = pd.read_csv(FDI_FILE, nrows=1)

excluded = {
    "failure_type",
    "run",
    "sample_index",
    "relative_to_failure",
}

signals = [
    c for c in df.columns
    if c not in excluded
]

print(f"\nTraining telemetry signals: {len(signals)}")

# ------------------------------------------------------------
# 2. Discover current ROS 2 topics
# ------------------------------------------------------------
print("\nDiscovering ROS 2 topics...")

topic_output = run_cmd(["ros2", "topic", "list"])

topics = [
    x.strip()
    for x in topic_output.splitlines()
    if x.strip()
]

fmu_topics = [
    t for t in topics
    if t.startswith("/fmu/")
]

print(f"Total ROS 2 topics: {len(topics)}")
print(f"PX4 /fmu topics: {len(fmu_topics)}")

# ------------------------------------------------------------
# 3. Get topic types
# ------------------------------------------------------------
topic_types = {}

for topic in fmu_topics:
    output = run_cmd(["ros2", "topic", "type", topic])

    if output:
        topic_types[topic] = output.splitlines()[0].strip()

# ------------------------------------------------------------
# 4. Inspect ROS 2 message definitions
# ------------------------------------------------------------
field_records = []

print("\nInspecting PX4 message fields...")

for topic, msg_type in topic_types.items():

    interface = run_cmd([
        "ros2",
        "interface",
        "show",
        msg_type
    ])

    if not interface:
        continue

    for line in interface.splitlines():

        line = line.strip()

        if not line:
            continue

        # Ignore comments
        if line.startswith("#"):
            continue

        # Basic ROS 2 field format:
        # type field_name
        parts = line.split()

        if len(parts) < 2:
            continue

        field_type = parts[0]
        field_name = parts[1]

        # Ignore nested/message declaration lines
        if "/" in field_type or field_type.startswith("px4"):
            pass

        field_records.append({
            "topic": topic,
            "message_type": msg_type,
            "field": field_name,
            "field_type": field_type,
        })

field_df = pd.DataFrame(field_records)

print(f"ROS 2 fields discovered: {len(field_df)}")

# ------------------------------------------------------------
# 5. Match training signal names against ROS 2 fields
# ------------------------------------------------------------
ros_fields = field_df["field"].tolist() if not field_df.empty else []

results = []

for signal in signals:

    signal_norm = normalize(signal)

    # Exact normalized field match
    exact = []

    for idx, field in enumerate(ros_fields):
        if normalize(field) == signal_norm:
            exact.append(idx)

    if exact:
        for idx in exact:
            row = field_df.iloc[idx]

            results.append({
                "training_signal": signal,
                "status": "EXACT_FIELD_MATCH",
                "ros2_topic": row["topic"],
                "message_type": row["message_type"],
                "ros2_field": row["field"],
                "field_type": row["field_type"],
            })

        continue

    # Partial/token match
    candidates = []

    signal_tokens = re.findall(
        r"[A-Z]+[0-9]*|[a-z]+|[0-9]+",
        signal
    )

    signal_parts = [
        normalize(x) for x in signal_tokens if x
    ]

    for idx, field in enumerate(ros_fields):

        field_norm = normalize(field)

        if len(field_norm) >= 3 and (
            signal_norm in field_norm or
            field_norm in signal_norm
        ):
            candidates.append(idx)

    if candidates:

        # Keep at most 5 candidates
        for idx in candidates[:5]:

            row = field_df.iloc[idx]

            results.append({
                "training_signal": signal,
                "status": "POSSIBLE_FIELD_MATCH",
                "ros2_topic": row["topic"],
                "message_type": row["message_type"],
                "ros2_field": row["field"],
                "field_type": row["field_type"],
            })

    else:

        results.append({
            "training_signal": signal,
            "status": "NO_DIRECT_FIELD_MATCH",
            "ros2_topic": "",
            "message_type": "",
            "ros2_field": "",
            "field_type": "",
        })

# ------------------------------------------------------------
# 6. Save mapping
# ------------------------------------------------------------
mapping_df = pd.DataFrame(results)

mapping_df.to_csv(OUTPUT, index=False)

# ------------------------------------------------------------
# 7. Summary
# ------------------------------------------------------------
status_counts = (
    mapping_df
    .groupby("training_signal")["status"]
    .first()
    .value_counts()
)

print("\n" + "=" * 70)
print("STEP 41 SUMMARY")
print("=" * 70)

print(f"Training signals: {len(signals)}")
print(f"ROS 2 fields discovered: {len(field_df)}")

print("\nSignal mapping status:")

for status, count in status_counts.items():
    print(f"  {status}: {count}")

print(f"\nSaved:")
print(OUTPUT)

print("\nTop exact matches:")

exact_df = mapping_df[
    mapping_df["status"] == "EXACT_FIELD_MATCH"
]

if exact_df.empty:
    print("  None")
else:
    print(
        exact_df[
            [
                "training_signal",
                "ros2_topic",
                "ros2_field"
            ]
        ].head(30).to_string(index=False)
    )

print("\nDone.")
