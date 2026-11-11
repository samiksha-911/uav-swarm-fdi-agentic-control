#!/usr/bin/env python3
"""
step73_autonomous_safety_governor.py
====================================
Autonomous Safety Governance Node for UAV Missions.

Pillars:
1. Online Fault Detection & Identification (using step70_clean_fdi_model.joblib).
2. Debounce & Fault Persistence Filter (eliminates transient sensor glitches).
3. Quantitative Mission Risk Matrix (Levels 0 to 3: Nominal, Low, Medium, Critical).
4. Autonomous Safety Governance Action Decision:
   - Nominal   -> NOMINAL_MISSION_CONTINUE
   - Baro/Mag  -> DEGRADE_NAVIGATION_AND_MONITOR
   - GPS       -> EMERGENCY_ALTCTL_OR_RTL (Position Hold Disabled)
   - Accel/Gyro-> EMERGENCY_FAILSAFE_LAND (Critical Flight Instability)
5. Audit Logging to safety_governance_audit_log.csv
"""

import time
from collections import deque
from pathlib import Path
import csv

import joblib
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import (
    SensorCombined,
    VehicleAttitude,
    SensorGps,
    VehicleGlobalPosition,
    VehicleLocalPosition,
    BatteryStatus,
    EstimatorStatusFlags,
    FailsafeFlags,
    VehicleStatus,
    VehicleAirData,
)

BASE = Path("/home/samiksha/uav_fdi_data")
MODEL_PATH = BASE / "step70_clean_fdi_model.joblib"
FEATURES_PATH = BASE / "step70_feature_names.joblib"
AUDIT_LOG = BASE / "safety_governance_audit_log.csv"

WINDOW = 50
DEBOUNCE_THRESHOLD = 3  # consecutive detections required to confirm a fault

qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def get(obj, name, default=0.0):
    if obj is None:
        return default
    return getattr(obj, name, default)


def extract_window_features(window_df):
    feats = {}
    for col in window_df.columns:
        if col in ("time", "label", "run_id"):
            continue
        s = window_df[col].replace({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
        x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        feats[f"{col}_mean"]  = float(np.mean(x))
        feats[f"{col}_std"]   = float(np.std(x))
        feats[f"{col}_min"]   = float(np.min(x))
        feats[f"{col}_max"]   = float(np.max(x))
        feats[f"{col}_range"] = float(np.max(x) - np.min(x))
        if len(x) > 1:
            feats[f"{col}_first"] = float(x[0])
            feats[f"{col}_last"]  = float(x[-1])
            feats[f"{col}_delta"] = float(x[-1] - x[0])
            dx = np.diff(x)
            feats[f"{col}_mean_abs_change"] = float(np.mean(np.abs(dx)))
            feats[f"{col}_max_abs_change"]  = float(np.max(np.abs(dx)))
            feats[f"{col}_change_std"]      = float(np.std(dx))
            feats[f"{col}_change_range"]    = float(np.max(dx) - np.min(dx))
    return feats


# -------------------------------------------------------------
# SAFETY GOVERNANCE POLICIES
# -------------------------------------------------------------
RISK_POLICIES = {
    "No Failure": {
        "risk_level": "LEVEL 0 - NOMINAL",
        "action": "NOMINAL_MISSION_CONTINUE",
        "description": "All subsystems healthy. Mission execution proceeds normally.",
    },
    "Compass Failure": {
        "risk_level": "LEVEL 1 - LOW RISK",
        "action": "DEGRADE_HEADING_ESTIMATION",
        "description": "Magnetometer degraded. Rely on GNSS ground course for yaw fusion.",
    },
    "Barometer Failure": {
        "risk_level": "LEVEL 1 - LOW RISK",
        "action": "FALLBACK_TO_GNSS_ALTITUDE",
        "description": "Barometric pressure degraded. Switch altitude reference to GPS/distance sensor.",
    },
    "GPS Failure": {
        "risk_level": "LEVEL 2 - MEDIUM RISK",
        "action": "ENGAGE_ALTITUDE_HOLD_AND_HOVER",
        "description": "Global positioning lost. Invalidate autonomous waypoint navigation and hold altitude.",
    },
    "Accelerometer Failure": {
        "risk_level": "LEVEL 3 - CRITICAL",
        "action": "TRIGGER_EMERGENCY_FAILSAFE_LAND",
        "description": "Attitude reference compromised. Immediate controlled descent to prevent crash.",
    },
    "Gyroscope Failure": {
        "risk_level": "LEVEL 3 - CRITICAL",
        "action": "TRIGGER_EMERGENCY_FAILSAFE_LAND",
        "description": "Angular rate control compromised. Immediate failsafe landing mandated.",
    },
}


class AutonomousSafetyGovernor(Node):

    def __init__(self):
        super().__init__("autonomous_safety_governor")

        self.get_logger().info("Initializing Autonomous Safety Governance Node...")
        self.model = joblib.load(MODEL_PATH)
        self.expected_features = joblib.load(FEATURES_PATH)
        self.rows = deque(maxlen=WINDOW)

        # Telemetry containers
        self.latest_sensor = None
        self.latest_att = None
        self.latest_gps = None
        self.latest_global = None
        self.latest_local = None
        self.latest_battery = None
        self.latest_estimator = None
        self.latest_failsafe = None
        self.latest_status = None
        self.latest_air = None

        # Fault debounce tracking
        self.current_candidate_fault = "No Failure"
        self.consecutive_fault_count = 0
        self.governed_fault_state = "No Failure"

        # Initialize audit log
        with open(AUDIT_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "fdi_prediction", "confidence", "confirmed_fault", "risk_level", "governance_action"])

        # Subscriptions
        self.create_subscription(SensorCombined, "/fmu/out/sensor_combined", self.sensor_cb, qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_cb, qos)
        self.create_subscription(SensorGps, "/fmu/out/vehicle_gps_position", self.gps_cb, qos)
        self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.global_cb, qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.local_cb, qos)
        self.create_subscription(BatteryStatus, "/fmu/out/battery_status_v1", self.battery_cb, qos)
        self.create_subscription(EstimatorStatusFlags, "/fmu/out/estimator_status_flags", self.estimator_cb, qos)
        self.create_subscription(FailsafeFlags, "/fmu/out/failsafe_flags", self.failsafe_cb, qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v4", self.status_cb, qos)
        self.create_subscription(VehicleAirData, "/fmu/out/vehicle_air_data", self.air_cb, qos)

        self.timer = self.create_timer(0.1, self.process)
        self.last_governance_time = 0.0
        self.get_logger().info("Autonomous Safety Governance Node is READY.")

    def sensor_cb(self, msg):    self.latest_sensor = msg
    def attitude_cb(self, msg):  self.latest_att = msg
    def gps_cb(self, msg):       self.latest_gps = msg
    def global_cb(self, msg):    self.latest_global = msg
    def local_cb(self, msg):     self.latest_local = msg
    def battery_cb(self, msg):   self.latest_battery = msg
    def estimator_cb(self, msg): self.latest_estimator = msg
    def failsafe_cb(self, msg):  self.latest_failsafe = msg
    def status_cb(self, msg):    self.latest_status = msg
    def air_cb(self, msg):       self.latest_air = msg

    def build_snapshot(self):
        s = self.latest_sensor
        att = self.latest_att
        gps = self.latest_gps
        gl = self.latest_global
        loc = self.latest_local
        bat = self.latest_battery
        est = self.latest_estimator
        fs = self.latest_failsafe
        stat = self.latest_status
        air = self.latest_air

        return {
            "time": time.time(),
            "acc_x": float(s.accelerometer_m_s2[0]) if s and hasattr(s, "accelerometer_m_s2") else 0.0,
            "acc_y": float(s.accelerometer_m_s2[1]) if s and hasattr(s, "accelerometer_m_s2") else 0.0,
            "acc_z": float(s.accelerometer_m_s2[2]) if s and hasattr(s, "accelerometer_m_s2") else 0.0,
            "gyro_x": float(s.gyro_rad[0]) if s and hasattr(s, "gyro_rad") else 0.0,
            "gyro_y": float(s.gyro_rad[1]) if s and hasattr(s, "gyro_rad") else 0.0,
            "gyro_z": float(s.gyro_rad[2]) if s and hasattr(s, "gyro_rad") else 0.0,
            "att_q0": float(att.q[0]) if att and hasattr(att, "q") else 1.0,
            "att_q1": float(att.q[1]) if att and hasattr(att, "q") else 0.0,
            "att_q2": float(att.q[2]) if att and hasattr(att, "q") else 0.0,
            "att_q3": float(att.q[3]) if att and hasattr(att, "q") else 0.0,
            "gps_lat": float(get(gps, "latitude_deg")),
            "gps_lon": float(get(gps, "longitude_deg")),
            "gps_alt": float(get(gps, "altitude_m")),
            "gps_fix_type": float(get(gps, "fix_type")),
            "gps_satellites": float(get(gps, "satellites_used")),
            "gps_velocity": float(get(gps, "vel_m_s")),
            "gps_vertical_speed": float(get(gps, "vel_d_m_s")),
            "global_lat": float(get(gl, "lat")),
            "global_lon": float(get(gl, "lon")),
            "global_alt": float(get(gl, "alt")),
            "global_alt_valid": float(get(gl, "alt_valid", False)),
            "global_position_valid": float(get(gl, "lat_lon_valid", False)),
            "local_x": float(get(loc, "x")),
            "local_y": float(get(loc, "y")),
            "local_z": float(get(loc, "z")),
            "local_vx": float(get(loc, "vx")),
            "local_vy": float(get(loc, "vy")),
            "local_vz": float(get(loc, "vz")),
            "local_z_valid": float(get(loc, "z_valid", True)),
            "local_vz_valid": float(get(loc, "v_z_valid", True)),
            "local_position_invalid": float(get(loc, "xy_valid", True) is False),
            "local_velocity_invalid": float(get(loc, "v_xy_valid", True) is False),
            "battery_voltage": float(get(bat, "voltage_v")),
            "battery_current": float(get(bat, "current_a")),
            "battery_remaining": float(get(bat, "remaining")),
            "attitude_invalid": float(get(fs, "attitude_invalid", False)),
            "angular_velocity_invalid": float(get(fs, "angular_velocity_invalid", False)),
            "local_position_invalid_est": float(get(fs, "local_position_invalid", False)),
            "local_velocity_invalid_est": float(get(fs, "local_velocity_invalid", False)),
            "global_position_invalid_est": float(get(fs, "global_position_invalid", False)),
            "inertial_dead_reckoning": float(get(est, "cs_inertial_dead_reckoning", 0)),
            "cs_mag_fault": float(get(est, "cs_mag_fault", 0)),
            "cs_mag": float(get(est, "cs_mag", 0)),
            "cs_mag_hdg": float(get(est, "cs_mag_hdg", 0)),
            "cs_mag_3d": float(get(est, "cs_mag_3d", 0)),
            "cs_mag_heading_consistent": float(get(est, "cs_mag_heading_consistent", 0)),
            "fs_bad_mag_x": float(get(est, "fs_bad_mag_x", 0)),
            "fs_bad_mag_y": float(get(est, "fs_bad_mag_y", 0)),
            "fs_bad_mag_z": float(get(est, "fs_bad_mag_z", 0)),
            "cs_baro_fault": float(get(est, "cs_baro_fault", 0)),
            "cs_gnss_fault": float(get(est, "cs_gnss_fault", 0)),
            "cs_gnss_pos": float(get(est, "cs_gnss_pos", 0)),
            "cs_baro_hgt": float(get(est, "cs_baro_hgt", 0)),
            "failsafe": float(get(fs, "failsafe", 0)),
            "nav_state": float(get(stat, "nav_state", 0)),
            "arming_state": float(get(stat, "arming_state", 0)),
            "baro_device_id": float(get(air, "baro_device_id", 0)),
            "baro_alt_meter": float(get(air, "baro_alt_meter", 0)),
            "baro_pressure_pa": float(get(air, "baro_pressure_pa", 0)),
            "ambient_temperature": float(get(air, "ambient_temperature", 0)),
            "rho": float(get(air, "rho", 0)),
        }

    def process(self):
        if self.latest_sensor is None:
            return

        self.rows.append(self.build_snapshot())
        if len(self.rows) < WINDOW:
            return

        now = time.time()
        if now - self.last_governance_time < 0.25:
            return
        self.last_governance_time = now

        df = pd.DataFrame(list(self.rows))
        features = extract_window_features(df)

        feat_vector = [features.get(col, 0.0) for col in self.expected_features]
        X = np.array([feat_vector], dtype=float)

        prediction = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        confidence = float(np.max(probs))

        # Debounce filter
        if prediction == self.current_candidate_fault:
            self.consecutive_fault_count += 1
        else:
            self.current_candidate_fault = prediction
            self.consecutive_fault_count = 1

        if self.consecutive_fault_count >= DEBOUNCE_THRESHOLD:
            self.governed_fault_state = self.current_candidate_fault

        # Retrieve governance policy
        policy = RISK_POLICIES.get(self.governed_fault_state, RISK_POLICIES["No Failure"])
        risk = policy["risk_level"]
        action = policy["action"]

        # Real-world benchmark accuracy map
        benchmark_acc_map = {
            "No Failure": 95.0,
            "Accelerometer Failure": 100.0,
            "Barometer Failure": 92.9,
            "Compass Failure": 75.0,
            "GPS Failure": 100.0,
            "Gyroscope Failure": 100.0,
        }
        real_acc = benchmark_acc_map.get(self.governed_fault_state, 90.0)

        # Color codes
        if "LEVEL 0" in risk:
            tag = "\033[92m[SAFETY GOVERNANCE: NOMINAL]\033[0m"
        elif "LEVEL 1" in risk:
            tag = "\033[93m[SAFETY GOVERNANCE: LOW RISK]\033[0m"
        elif "LEVEL 2" in risk:
            tag = "\033[94m[SAFETY GOVERNANCE: MEDIUM RISK]\033[0m"
        else:
            tag = "\033[91m[SAFETY GOVERNANCE: CRITICAL FAULT]\033[0m"

        self.get_logger().info(
            f"{tag} | State: {self.governed_fault_state} | Real Accuracy: {real_acc:.1f}% "
            f"| Confidence: {confidence:.1%} | Risk: {risk} | Action: \033[1m{action}\033[0m"
        )

        # Audit log append
        with open(AUDIT_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([round(now, 3), prediction, round(confidence, 4), self.governed_fault_state, real_acc, risk, action])


def main():
    rclpy.init()
    node = AutonomousSafetyGovernor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
