import os
import re
import subprocess
import pandas as pd

BASE = "/home/samiksha/uav_fdi_data"

FDI_FILE = os.path.join(BASE, "fdi_train.csv")
MAPPING_FILE = os.path.join(BASE, "step41_live_ros2_signal_mapping.csv")
OUTPUT_FILE = os.path.join(BASE, "step41B_px4_signal_candidates.csv")


# ============================================================
# 1. LOAD TRAINING SIGNALS
# ============================================================

df = pd.read_csv(
    FDI_FILE,
    nrows=1
)

exclude = {
    "failure_type",
    "run",
    "sample_index",
    "relative_to_failure"
}

training_signals = [
    c for c in df.columns
    if c not in exclude
]

print("=" * 70)
print("STEP 41B — PX4 SIGNAL CANDIDATE MAPPING")
print("=" * 70)
print(f"Training telemetry signals: {len(training_signals)}")


# ============================================================
# 2. LOAD ROS2 DISCOVERY
# ============================================================

print("\nDiscovering ROS 2 topics...")

topics_result = subprocess.run(
    ["ros2", "topic", "list"],
    capture_output=True,
    text=True,
    check=True
)

topics = [
    x.strip()
    for x in topics_result.stdout.splitlines()
    if x.strip().startswith("/fmu/")
]

print(f"PX4 topics: {len(topics)}")


# ============================================================
# 3. DISCOVER TOPIC TYPES AND FIELDS
# ============================================================

records = []

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

        interface_text = interface_result.stdout

        for line in interface_text.splitlines():

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            # Ignore nested message declarations.
            if line.startswith("==="):
                continue

            parts = line.split()

            if len(parts) != 2:
                continue

            field_type = parts[0]
            field_name = parts[1]

            # Skip constants.
            if "=" in field_name:
                continue

            # Basic ROS message field.
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", field_name):
                continue

            records.append({
                "topic": topic,
                "message_type": msg_type,
                "field": field_name,
                "field_type": field_type
            })

    except Exception:
        continue


ros_fields = pd.DataFrame(records)

print(f"ROS 2 fields discovered: {len(ros_fields)}")


# ============================================================
# 4. NORMALIZATION
# ============================================================

def normalize(text):

    text = str(text).lower()

    replacements = {
        "vehicle_": "",
        "local_position": "position",
        "estimator_status_flags": "estimator",
        "vehicle_attitude": "attitude",
        "vehicle_gps_position": "gps",
        "battery_status": "battery",
        "sensor_combined": "imu",
        "vehicle_angular_velocity": "gyro",
        "vehicle_acceleration": "acc",
        "vehicle_magnetometer": "mag",
        "vehicle_air_data": "baro",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(r"[^a-z0-9]+", "_", text)

    return text.strip("_")


# ============================================================
# 5. MANUAL SEMANTIC GROUPS
# ============================================================

signal_groups = {

    "gps": [
        "gps",
        "nsats",
        "satellite",
        "latitude",
        "longitude",
        "altitude",
        "velocity",
        "vel",
        "ground",
        "course",
    ],

    "imu": [
        "acc",
        "acceleration",
        "gyro",
        "angular",
        "rate",
        "imu",
    ],

    "mag": [
        "mag",
        "magnetometer",
        "heading",
        "yaw",
    ],

    "baro": [
        "baro",
        "pressure",
        "temperature",
        "altitude",
        "height",
    ],

    "attitude": [
        "attitude",
        "roll",
        "pitch",
        "yaw",
        "quaternion",
    ],

    "battery": [
        "battery",
        "voltage",
        "current",
        "remaining",
        "energy",
    ],

    "position": [
        "position",
        "latitude",
        "longitude",
        "altitude",
        "velocity",
        "local",
    ],

    "estimator": [
        "estimator",
        "innovation",
        "innovation_test",
        "fault",
        "valid",
        "reset",
        "dead_reckoning",
        "gps",
        "mag",
        "baro",
    ],

    "vibration": [
        "vibration",
        "clip",
    ],

    "motor": [
        "actuator",
        "motor",
        "thrust",
        "output",
    ],

    "rc": [
        "rc",
        "manual",
        "input",
        "control",
    ],
}


def source_group(signal):

    s = signal.lower()

    if s.startswith("gps_"):
        return "gps"

    if s.startswith("imu_"):
        return "imu"

    if s.startswith("mag_"):
        return "mag"

    if s.startswith("baro_"):
        return "baro"

    if s.startswith("att_"):
        return "attitude"

    if s.startswith("bat_"):
        return "battery"

    if s.startswith("pos_"):
        return "position"

    if s.startswith("xkf"):
        return "estimator"

    if s.startswith("xkq"):
        return "attitude"

    if s.startswith("xktv"):
        return "estimator"

    if s.startswith("xky"):
        return "estimator"

    if s.startswith("rate_"):
        return "attitude"

    if s.startswith("vibe_"):
        return "vibration"

    if s.startswith("motb_"):
        return "motor"

    if s.startswith("mav_"):
        return "rc"

    if s.startswith("ctun_"):
        return "motor"

    if s.startswith("pscd_"):
        return "position"

    if s.startswith("psce_"):
        return "position"

    if s.startswith("pscn_"):
        return "position"

    if s.startswith("terr_"):
        return "position"

    if s.startswith("sim_"):
        return "position"

    return "unknown"


# ============================================================
# 6. GENERATE CANDIDATES
# ============================================================

candidate_rows = []

for signal in training_signals:

    group = source_group(signal)

    signal_norm = normalize(signal)

    candidates = []

    for _, row in ros_fields.iterrows():

        topic_norm = normalize(row["topic"])
        field_norm = normalize(row["field"])
        type_norm = normalize(row["message_type"])

        combined = (
            topic_norm + " " +
            field_norm + " " +
            type_norm
        )

        score = 0

        # ----------------------------------------------------
        # Exact normalized field match
        # ----------------------------------------------------

        if signal_norm == field_norm:
            score += 100

        # ----------------------------------------------------
        # Signal token matching
        # ----------------------------------------------------

        signal_tokens = set(signal_norm.split("_"))
        field_tokens = set(field_norm.split("_"))

        common = signal_tokens.intersection(field_tokens)

        score += 10 * len(common)

        # ----------------------------------------------------
        # Group matching
        # ----------------------------------------------------

        if group in signal_groups:

            keywords = signal_groups[group]

            keyword_hits = sum(
                1 for k in keywords
                if k in combined
            )

            score += min(keyword_hits * 4, 20)

        # ----------------------------------------------------
        # Topic relevance
        # ----------------------------------------------------

        if group == "gps" and (
            "gps" in topic_norm or
            "global_position" in topic_norm
        ):
            score += 20

        if group == "imu" and (
            "imu" in topic_norm or
            "sensor_combined" in topic_norm
        ):
            score += 20

        if group == "mag" and (
            "magnetometer" in topic_norm or
            "mag" in topic_norm
        ):
            score += 20

        if group == "baro" and (
            "air_data" in topic_norm or
            "baro" in topic_norm
        ):
            score += 20

        if group == "attitude" and (
            "attitude" in topic_norm or
            "angular_velocity" in topic_norm
        ):
            score += 20

        if group == "battery" and "battery" in topic_norm:
            score += 25

        if group == "position" and (
            "local_position" in topic_norm or
            "global_position" in topic_norm
        ):
            score += 20

        if group == "estimator" and (
            "estimator" in topic_norm
        ):
            score += 25

        if group == "vibration" and "vibration" in topic_norm:
            score += 25

        if group == "motor" and (
            "actuator" in topic_norm or
            "thrust" in topic_norm
        ):
            score += 20

        if group == "rc" and (
            "manual_control" in topic_norm or
            "rc" in topic_norm
        ):
            score += 20

        if score > 0:

            candidates.append({
                "score": score,
                "topic": row["topic"],
                "message_type": row["message_type"],
                "field": row["field"],
                "field_type": row["field_type"]
            })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    top = candidates[:5]

    if top:

        for rank, candidate in enumerate(top, start=1):

            candidate_rows.append({
                "training_signal": signal,
                "source_group": group,
                "candidate_rank": rank,
                "score": candidate["score"],
                "topic": candidate["topic"],
                "message_type": candidate["message_type"],
                "field": candidate["field"],
                "field_type": candidate["field_type"]
            })

    else:

        candidate_rows.append({
            "training_signal": signal,
            "source_group": group,
            "candidate_rank": None,
            "score": 0,
            "topic": "",
            "message_type": "",
            "field": "",
            "field_type": ""
        })


# ============================================================
# 7. SAVE
# ============================================================

result = pd.DataFrame(candidate_rows)

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

best = (
    result
    .sort_values(
        ["training_signal", "score"],
        ascending=[True, False]
    )
    .groupby("training_signal", as_index=False)
    .first()
)

print("\n" + "=" * 70)
print("STEP 41B SUMMARY")
print("=" * 70)

print(f"Training signals: {len(training_signals)}")
print(f"Signals with candidates: {(best['score'] > 0).sum()}")
print(f"Signals without candidates: {(best['score'] == 0).sum()}")

print("\nTop candidate examples:")

for _, row in best.sort_values(
    "score",
    ascending=False
).head(30).iterrows():

    print(
        f"{row['training_signal']:25s} -> "
        f"{row['topic']} :: "
        f"{row['field']} "
        f"(score={row['score']})"
    )

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDone.")
