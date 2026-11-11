import os
import time
import math
import joblib
import numpy as np
import pandas as pd

BASE = "/home/samiksha/uav_fdi_data"

MODEL_FILE = os.path.join(
    BASE,
    "uav_fdi_extratrees_model.joblib"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "step42_live_feature_coverage.csv"
)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

bundle = joblib.load(MODEL_FILE)

model = bundle["model"]
features = list(bundle["features"])
medians = bundle["medians"]
classes = list(bundle["classes"])

print("=" * 70)
print("STEP 42 — LIVE PX4 → ML FEATURE ADAPTER")
print("=" * 70)

print(f"Model: {bundle['model_name']}")
print(f"Selected features: {len(features)}")
print(f"Classes: {classes}")


# ============================================================
# SOURCE SIGNALS
# ============================================================

# These are the raw FDI signals that we know the model's
# integrated feature set was derived from.

source_signals = set([
    "AHR2_Roll", "AHR2_Pitch", "AHR2_Yaw", "AHR2_Alt",
    "AHR2_Lat", "AHR2_Lng", "AHR2_Q1", "AHR2_Q2",
    "AHR2_Q3", "AHR2_Q4",

    "ATT_DesRoll", "ATT_Roll", "ATT_DesPitch", "ATT_Pitch",
    "ATT_DesYaw", "ATT_Yaw", "ATT_ErrRP", "ATT_ErrYaw",

    "BARO_Alt", "BARO_Press", "BARO_Temp", "BARO_CRt",

    "BAT_VoltR", "BAT_Curr", "BAT_CurrTot", "BAT_EnrgTot",
    "BAT_Res", "BAT_RemPct",

    "CTUN_ThI", "CTUN_ABst", "CTUN_ThO", "CTUN_ThH",
    "CTUN_DAlt", "CTUN_Alt", "CTUN_BAlt", "CTUN_TAlt",
    "CTUN_DCRt", "CTUN_CRt",

    "GPS_NSats", "GPS_Lat", "GPS_Lng", "GPS_Alt",
    "GPS_Spd", "GPS_GCrs", "GPS_VZ",

    "IMU_GyrX", "IMU_GyrY", "IMU_GyrZ",
    "IMU_AccX", "IMU_AccY", "IMU_AccZ",
    "IMU_T", "IMU_GHz", "IMU_AHz",

    "MAG_MagX", "MAG_MagY", "MAG_MagZ",
    "MAG_OfsX", "MAG_OfsY", "MAG_OfsZ",

    "MAV_txp", "MAV_rxp",

    "MOTB_LiftMax", "MOTB_BatVolt", "MOTB_ThrAvMx",
    "MOTB_FailFlags",

    "POS_Lat", "POS_Lng", "POS_Alt",
    "POS_RelHomeAlt", "POS_RelOriginAlt",

    "PSCD_TPD", "PSCD_PD", "PSCD_DVD", "PSCD_TVD",
    "PSCD_VD", "PSCD_DAD", "PSCD_TAD", "PSCD_AD",

    "PSCE_TPE", "PSCE_PE", "PSCE_DVE", "PSCE_TVE",
    "PSCE_VE", "PSCE_DAE", "PSCE_TAE", "PSCE_AE",

    "PSCN_TPN", "PSCN_PN", "PSCN_DVN", "PSCN_TVN",
    "PSCN_VN", "PSCN_DAN", "PSCN_TAN", "PSCN_AN",

    "RATE_RDes", "RATE_R", "RATE_ROut",
    "RATE_PDes", "RATE_P", "RATE_POut",
    "RATE_YDes", "RATE_Y", "RATE_YOut",
    "RATE_ADes", "RATE_A", "RATE_AOut",

    "SIM_Roll", "SIM_Pitch", "SIM_Yaw", "SIM_Alt",
    "SIM_Lat", "SIM_Lng",
    "SIM_Q1", "SIM_Q2", "SIM_Q3", "SIM_Q4",

    "TERR_Lat", "TERR_Lng", "TERR_TerrH",
    "TERR_CHeight", "TERR_Pending", "TERR_Loaded",
    "TERR_ROfs",

    "VIBE_VibeX", "VIBE_VibeY", "VIBE_VibeZ",

    "XKF1_Roll", "XKF1_Pitch", "XKF1_Yaw",
    "XKF1_VN", "XKF1_VE", "XKF1_VD",
    "XKF1_dPD", "XKF1_PN", "XKF1_PE", "XKF1_PD",
    "XKF1_GX", "XKF1_GY", "XKF1_GZ", "XKF1_OH",

    "XKF2_AX", "XKF2_AY", "XKF2_AZ",
    "XKF2_MN", "XKF2_ME", "XKF2_MD",
    "XKF2_MX", "XKF2_MY", "XKF2_MZ",

    "XKF3_IVN", "XKF3_IVE", "XKF3_IVD",
    "XKF3_IPN", "XKF3_IPE", "XKF3_IPD",
    "XKF3_IMX", "XKF3_IMY", "XKF3_IMZ",
    "XKF3_IYAW", "XKF3_RErr", "XKF3_ErSc",

    "XKF4_SV", "XKF4_SP", "XKF4_SH", "XKF4_SM",
    "XKF4_errRP", "XKF4_OFN", "XKF4_OFE",
    "XKF4_TS", "XKF4_SS", "XKF4_GPS",

    "XKF5_HAGL", "XKF5_offset", "XKF5_Herr",
    "XKF5_eAng", "XKF5_eVel", "XKF5_ePos",

    "XKQ_Q1", "XKQ_Q2", "XKQ_Q3", "XKQ_Q4",

    "XKTV_TVS", "XKTV_TVD",

    "XKY0_YC", "XKY0_YCS", "XKY0_Y0", "XKY0_Y1",
    "XKY0_Y2", "XKY0_Y3", "XKY0_Y4",
    "XKY0_W0", "XKY0_W1", "XKY0_W2",
    "XKY0_W3", "XKY0_W4",

    "XKY1_IVN0", "XKY1_IVN1", "XKY1_IVN2",
    "XKY1_IVN3", "XKY1_IVN4",
    "XKY1_IVE0", "XKY1_IVE1", "XKY1_IVE2",
    "XKY1_IVE3", "XKY1_IVE4",

    "Status"
])


# ============================================================
# FEATURE → SOURCE SIGNAL
# ============================================================

def feature_source(feature):

    # SEAD-style:
    # STREAM__COLUMN__STAT
    parts = feature.split("__")

    if len(parts) == 3:
        stream, column, stat = parts
        return f"{stream}::{column}", stat

    # FDI-derived:
    # SIGNAL__STAT
    if len(parts) == 2:
        signal, stat = parts
        return signal, stat

    return None, None


# ============================================================
# CLASSIFY FEATURE PROVENANCE
# ============================================================

rows = []

for feature in features:

    source, stat = feature_source(feature)

    if source is None:

        source_type = "UNKNOWN"

    elif "::" in source:

        source_type = "SEAD_FEATURE"

    elif source in source_signals:

        source_type = "FDI_SOURCE_FEATURE"

    else:

        source_type = "OTHER"


    # Determine whether the underlying source is something
    # we have explicitly identified as potentially available
    # from PX4.

    px4_status = "NOT_YET_MAPPED"

    if source is not None:

        raw_signal = source

        if "::" not in raw_signal:

            if raw_signal.startswith("SIM_"):

                px4_status = "UNAVAILABLE_SIM_SIGNAL"

            elif raw_signal == "Status":

                px4_status = "SPECIAL_STATUS_SIGNAL"

            elif raw_signal.startswith("XKF"):

                px4_status = "PX4_ESTIMATOR_EQUIVALENT_REQUIRED"

            elif raw_signal.startswith("PSCD"):
                px4_status = "PX4_POSITION_DERIVATION_REQUIRED"

            elif raw_signal.startswith("PSCE"):
                px4_status = "PX4_POSITION_DERIVATION_REQUIRED"

            elif raw_signal.startswith("PSCN"):
                px4_status = "PX4_POSITION_DERIVATION_REQUIRED"

            elif raw_signal.startswith("RATE_"):
                px4_status = "PX4_RATE_DERIVATION_REQUIRED"

            elif raw_signal.startswith("VIBE_"):
                px4_status = "PX4_VIBRATION_MAPPING_REQUIRED"

            elif raw_signal.startswith("MAG_"):
                px4_status = "PX4_MAGNETOMETER_MAPPING_REQUIRED"

            else:
                px4_status = "PX4_SOURCE_MAPPING_REQUIRED"

    rows.append({
        "model_feature": feature,
        "source": source,
        "statistic": stat,
        "source_type": source_type,
        "px4_status": px4_status
    })


coverage = pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

coverage.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE PROVENANCE")
print("=" * 70)

print(
    coverage["source_type"]
    .value_counts()
    .to_string()
)

print("\nPX4 adaptation status:")

print(
    coverage["px4_status"]
    .value_counts()
    .to_string()
)

print("\nModel features:", len(features))

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDone.")
