import os
import re
import pandas as pd

BASE = "/home/samiksha/uav_fdi_data"

FDI_FILE = os.path.join(BASE, "fdi_train.csv")
PX4_FILE = os.path.join(BASE, "step41F_relevant_px4_fields.csv")
SOURCE_FILE = os.path.join(BASE, "step40_fdi_signal_inventory.csv")

OUTPUT = os.path.join(
    BASE,
    "step41G_model_source_availability.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    FDI_FILE,
    nrows=1
)

exclude = {
    "failure_type",
    "run",
    "sample_index",
    "relative_to_failure"
}

all_training_signals = [
    c for c in train.columns
    if c not in exclude
]

px4 = pd.read_csv(PX4_FILE)

source = pd.read_csv(SOURCE_FILE)

# The source inventory contains the original FDI signals.
# Keep only names that occur in the training telemetry.
if "signal" in source.columns:
    source_signals = source["signal"].astype(str).tolist()
elif "source_signal" in source.columns:
    source_signals = source["source_signal"].astype(str).tolist()
else:
    source_signals = all_training_signals


# ============================================================
# PX4 FIELD LOOKUP
# ============================================================

px4_fields = set(
    px4["field"].astype(str)
)

px4_topics = px4.groupby(
    "field"
)["topic"].apply(
    lambda x: ";".join(sorted(set(x.astype(str))))
).to_dict()


# ============================================================
# SOURCE FAMILY
# ============================================================

def get_family(signal):

    s = signal.upper()

    prefixes = [
        "AHR2_",
        "ATT_",
        "BARO_",
        "BAT_",
        "CTUN_",
        "GPS_",
        "IMU_",
        "MAG_",
        "MAV_",
        "MOTB_",
        "POS_",
        "PSCD_",
        "PSCE_",
        "PSCN_",
        "RATE_",
        "SIM_",
        "TERR_",
        "VIBE_",
        "XKF1_",
        "XKF2_",
        "XKF3_",
        "XKF4_",
        "XKF5_",
        "XKQ_",
        "XKTV_",
        "XKY0_",
        "XKY1_",
        "Status"
    ]

    for p in prefixes:

        if s.startswith(p):
            return p.rstrip("_")

    return "OTHER"


# ============================================================
# KNOWN PX4 CAPABILITY FAMILIES
# ============================================================

capability_topics = {

    "AHR2": [
        "vehicle_attitude",
        "vehicle_local_position_v1"
    ],

    "ATT": [
        "vehicle_attitude",
        "vehicle_local_position_v1"
    ],

    "BARO": [
        "vehicle_local_position_v1",
        "vehicle_global_position",
        "estimator_status_flags"
    ],

    "BAT": [
        "battery_status_v1"
    ],

    "CTUN": [
        "vehicle_control_mode",
        "vehicle_local_position_v1"
    ],

    "GPS": [
        "vehicle_gps_position",
        "vehicle_global_position",
        "vehicle_local_position_v1",
        "estimator_status_flags"
    ],

    "IMU": [
        "sensor_combined"
    ],

    "MAG": [
        "estimator_status_flags"
    ],

    "MAV": [
        "input_rc",
        "manual_control_setpoint"
    ],

    "MOTB": [
        "vehicle_control_mode",
        "vehicle_status_v4"
    ],

    "POS": [
        "vehicle_global_position",
        "vehicle_local_position_v1"
    ],

    "PSCD": [
        "vehicle_local_position_v1"
    ],

    "PSCE": [
        "vehicle_local_position_v1"
    ],

    "PSCN": [
        "vehicle_local_position_v1"
    ],

    "RATE": [
        "vehicle_odometry",
        "vehicle_local_position_v1"
    ],

    "SIM": [
        # Simulator-only source.
    ],

    "TERR": [
        "vehicle_local_position_v1",
        "vehicle_global_position"
    ],

    "VIBE": [
        "sensor_combined"
    ],

    "XKF1": [
        "vehicle_local_position_v1",
        "vehicle_odometry"
    ],

    "XKF2": [
        "sensor_combined",
        "vehicle_odometry"
    ],

    "XKF3": [
        "estimator_status_flags",
        "vehicle_local_position_v1"
    ],

    "XKF4": [
        "estimator_status_flags",
        "vehicle_gps_position"
    ],

    "XKF5": [
        "vehicle_local_position_v1",
        "vehicle_global_position"
    ],

    "XKQ": [
        "vehicle_attitude"
    ],

    "XKTV": [
        "vehicle_local_position_v1"
    ],

    "XKY0": [
        "estimator_status_flags"
    ],

    "XKY1": [
        "estimator_status_flags"
    ],

    "Status": [
        "estimator_status_flags",
        "vehicle_status_v4",
        "failsafe_flags"
    ],

    "OTHER": []
}


# ============================================================
# BUILD AVAILABILITY AUDIT
# ============================================================

rows = []

for signal in all_training_signals:

    family = get_family(signal)

    possible_topics = capability_topics.get(
        family,
        []
    )

    matching_topics = []

    for topic in px4["topic"].astype(str).unique():

        if any(
            capability in topic
            for capability in possible_topics
        ):
            matching_topics.append(topic)

    if signal in px4_fields:

        availability = "DIRECT_FIELD"

        topic = px4_topics.get(
            signal,
            ""
        )

    elif matching_topics:

        availability = "FAMILY_SOURCE_AVAILABLE"

        topic = ";".join(
            sorted(set(matching_topics))
        )

    else:

        availability = "NO_IDENTIFIED_SOURCE"

        topic = ""

    rows.append({
        "training_signal": signal,
        "family": family,
        "availability": availability,
        "candidate_px4_topics": topic
    })


result = pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("STEP 41G — MODEL SOURCE AVAILABILITY")
print("=" * 70)

print(
    f"Training telemetry signals: {len(result)}"
)

print("\nAvailability:")

print(
    result["availability"]
    .value_counts()
    .to_string()
)

print("\nBy source family:")

family_summary = pd.crosstab(
    result["family"],
    result["availability"]
)

print(
    family_summary.to_string()
)

print("\nSignals with no identified PX4 source:")

missing = result[
    result["availability"] == "NO_IDENTIFIED_SOURCE"
]

for signal in missing["training_signal"]:

    print(" ", signal)

print("\nSaved:")
print(OUTPUT)

print("\nDone.")

