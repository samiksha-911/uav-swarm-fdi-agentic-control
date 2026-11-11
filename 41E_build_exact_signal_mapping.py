import os
import re
import pandas as pd

BASE = "/home/samiksha/uav_fdi_data"

FDI_FILE = os.path.join(BASE, "fdi_train.csv")
PX4_FILE = os.path.join(BASE, "step41D_px4_output_fields.csv")

OUTPUT = os.path.join(
    BASE,
    "step41E_exact_signal_mapping.csv"
)


# ============================================================
# LOAD TRAINING SIGNALS
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

signals = [
    c for c in train.columns
    if c not in exclude
]


# ============================================================
# LOAD PX4 FIELDS
# ============================================================

px4 = pd.read_csv(PX4_FILE)

px4["field"] = px4["field"].astype(str)
px4["topic"] = px4["topic"].astype(str)


# ============================================================
# MANUAL VERIFIED PX4 MAPPINGS
#
# These are physical/semantic mappings, not fuzzy matches.
# ============================================================

mapping = {

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    "GPS_Lat": (
        "DIRECT",
        "/fmu/out/vehicle_gps_position",
        "latitude_deg"
    ),

    "GPS_Lng": (
        "DIRECT",
        "/fmu/out/vehicle_gps_position",
        "longitude_deg"
    ),

    "GPS_Alt": (
        "DIRECT",
        "/fmu/out/vehicle_gps_position",
        "altitude_msl_m"
    ),

    "GPS_NSats": (
        "DIRECT",
        "/fmu/out/vehicle_gps_position",
        "satellites_used"
    ),

    "GPS_Spd": (
        "DIRECT",
        "/fmu/out/vehicle_gps_position",
        "vel_m_s"
    ),

    "GPS_VZ": (
        "DIRECT",
        "/fmu/out/vehicle_gps_position",
        "vel_d_m_s"
    ),

    # --------------------------------------------------------
    # Global/local position
    # --------------------------------------------------------

    "POS_Lat": (
        "DIRECT",
        "/fmu/out/vehicle_global_position",
        "lat"
    ),

    "POS_Lng": (
        "DIRECT",
        "/fmu/out/vehicle_global_position",
        "lon"
    ),

    "POS_Alt": (
        "DIRECT",
        "/fmu/out/vehicle_global_position",
        "alt"
    ),

    # --------------------------------------------------------
    # IMU / acceleration / gyro
    # --------------------------------------------------------

    "IMU_GyrX": (
        "DIRECT",
        "/fmu/out/sensor_combined",
        "gyro_rad_s"
    ),

    "IMU_AccX": (
        "DIRECT",
        "/fmu/out/sensor_combined",
        "accelerometer_m_s2"
    ),

    # --------------------------------------------------------
    # Battery
    # --------------------------------------------------------

    "BAT_VoltR": (
        "DIRECT",
        "/fmu/out/battery_status_v1",
        "voltage_v"
    ),

    "BAT_Curr": (
        "DIRECT",
        "/fmu/out/battery_status_v1",
        "current_a"
    ),

    "BAT_RemPct": (
        "DIRECT",
        "/fmu/out/battery_status_v1",
        "remaining"
    ),

    # --------------------------------------------------------
    # Attitude
    # --------------------------------------------------------

    "ATT_Roll": (
        "DIRECT",
        "/fmu/out/vehicle_attitude",
        "q"
    ),

    # --------------------------------------------------------
    # RC
    # --------------------------------------------------------

    "MAV_rxp": (
        "STATUS_EQUIVALENT",
        "/fmu/out/input_rc",
        "timestamp"
    ),

    # --------------------------------------------------------
    # Estimator health
    # --------------------------------------------------------

    "XKF3_RErr": (
        "STATUS_EQUIVALENT",
        "/fmu/out/estimator_status_flags",
        "cs_mag_hdg"
    ),

    "XKF4_GPS": (
        "STATUS_EQUIVALENT",
        "/fmu/out/estimator_status_flags",
        "cs_gnss_pos"
    ),
}


# ============================================================
# FIELD EXISTENCE CHECK
# ============================================================

def field_exists(topic, field):

    hit = px4[
        (px4["topic"] == topic) &
        (px4["field"] == field)
    ]

    return len(hit) > 0


# ============================================================
# BUILD OUTPUT
# ============================================================

rows = []

for signal in signals:

    if signal in mapping:

        status, topic, field = mapping[signal]

        if field_exists(topic, field):

            rows.append({
                "training_signal": signal,
                "mapping_status": status,
                "px4_topic": topic,
                "px4_field": field,
                "verification": "FIELD_EXISTS"
            })

        else:

            rows.append({
                "training_signal": signal,
                "mapping_status": "UNAVAILABLE",
                "px4_topic": topic,
                "px4_field": field,
                "verification": "FIELD_NOT_FOUND"
            })

    else:

        rows.append({
            "training_signal": signal,
            "mapping_status": "UNMAPPED",
            "px4_topic": "",
            "px4_field": "",
            "verification": ""
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
print("STEP 41E — EXACT PX4 SIGNAL MAPPING")
print("=" * 70)

print(f"Training signals: {len(result)}")

print("\nMapping status:")

print(
    result["mapping_status"]
    .value_counts()
    .to_string()
)

print("\nVerified mappings:")

verified = result[
    result["verification"] == "FIELD_EXISTS"
]

for _, row in verified.iterrows():

    print(
        f"{row['training_signal']:20s} -> "
        f"{row['px4_topic']} :: "
        f"{row['px4_field']} "
        f"[{row['mapping_status']}]"
    )

print("\nSaved:")
print(OUTPUT)

print("\nIMPORTANT:")
print("UNMAPPED signals are intentionally not guessed.")
print("They will be handled after inspecting the actual PX4 interfaces.")

print("\nDone.")

