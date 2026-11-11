import os
import pandas as pd

BASE = "/home/samiksha/uav_fdi_data"

FDI_FILE = os.path.join(BASE, "fdi_train.csv")
PX4_FILE = os.path.join(BASE, "step41D_px4_output_fields.csv")

OUTPUT = os.path.join(
    BASE,
    "step41F_relevant_px4_fields.csv"
)

# ------------------------------------------------------------
# Load the 207 training telemetry signals
# ------------------------------------------------------------

train = pd.read_csv(FDI_FILE, nrows=1)

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

# ------------------------------------------------------------
# Load PX4 fields
# ------------------------------------------------------------

px4 = pd.read_csv(PX4_FILE)

# ------------------------------------------------------------
# Identify relevant PX4 topics
# ------------------------------------------------------------

important_topics = [
    "vehicle_gps_position",
    "vehicle_global_position",
    "vehicle_local_position",
    "vehicle_attitude",
    "sensor_combined",
    "battery_status",
    "estimator_status_flags",
    "failsafe_flags",
    "vehicle_status",
    "vehicle_odometry",
    "manual_control_setpoint",
    "input_rc",
    "airspeed_validated",
    "wind",
    "vehicle_control_mode",
    "vehicle_land_detected",
]

px4_relevant = px4[
    px4["topic"].apply(
        lambda x: any(t in str(x) for t in important_topics)
    )
].copy()

# ------------------------------------------------------------
# Signal family
# ------------------------------------------------------------

def family(signal):

    s = signal.lower()

    prefixes = [
        "gps_", "imu_", "mag_", "baro_", "att_",
        "bat_", "ctun_", "motb_", "pos_", "rate_",
        "vibe_", "xkf", "xkq_", "xktv_", "xky",
        "pscd_", "psce_", "pscn_", "terr_", "sim_",
        "mav_"
    ]

    for p in prefixes:
        if s.startswith(p):
            return p.rstrip("_")

    return "other"


# ------------------------------------------------------------
# Print relevant fields grouped by PX4 topic
# ------------------------------------------------------------

print("=" * 70)
print("STEP 41F — RELEVANT PX4 TELEMETRY FIELD REVIEW")
print("=" * 70)

print(f"Training telemetry signals: {len(signals)}")
print(f"Relevant PX4 fields: {len(px4_relevant)}")

for topic, group in px4_relevant.groupby("topic"):

    print("\n" + "-" * 70)
    print(topic)

    fields = group["field"].tolist()

    print("Fields:")

    print("  " + ", ".join(fields))


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

px4_relevant.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("Saved:")
print(OUTPUT)
print("=" * 70)

print("\nDone.")
