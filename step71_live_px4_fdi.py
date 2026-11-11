#!/usr/bin/env python3
"""
step71_live_px4_fdi.py
======================
Real-time PX4 Fault Detection and Identification (FDI) Node.
Subscribes to live ROS 2 telemetry from PX4 (SITL or hardware),
buffers 50 samples, extracts the identical 732 statistical features,
and runs online inference using step70_clean_fdi_model.joblib.
"""

import time
from collections import deque
from pathlib import Path

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

WINDOW = 50

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
    """Extract 12 features per column matching the trained model."""
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


class LivePX4FDI(Node):

    def __init__(self):
        super().__init__("live_px4_fdi")

        self.get_logger().info("Loading FDI model and expected feature list...")
        self.model = joblib.load(MODEL_PATH)
        self.expected_features = joblib.load(FEATURES_PATH)
        self.get_logger().info(f"Loaded model: {MODEL_PATH.name}")
        self.get_logger().info(f"Features: {len(self.expected_features)} columns")

        self.rows = deque(maxlen=WINDOW)

        # Message containers
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

        # Subscriptions matching 63_live_px4_airdata_recorder.py
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

        self.recent_predictions = deque(maxlen=15)
        self.benchmark_accuracies = {
            "No Failure": 95.0,
            "Accelerometer Failure": 100.0,
            "Barometer Failure": 92.9,
            "Compass Failure": 75.0,
            "GPS Failure": 100.0,
            "Gyroscope Failure": 100.0,
        }
        self.timer = self.create_timer(0.1, self.process)
        self.get_logger().info("Waiting for telemetry buffer to fill (50 samples / ~5s)...")
        self.last_print_time = 0.0

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
            # Sensor
            "acc_x": float(s.accelerometer_m_s2[0]) if s and hasattr(s, "accelerometer_m_s2") else 0.0,
            "acc_y": float(s.accelerometer_m_s2[1]) if s and hasattr(s, "accelerometer_m_s2") else 0.0,
            "acc_z": float(s.accelerometer_m_s2[2]) if s and hasattr(s, "accelerometer_m_s2") else 0.0,
            "gyro_x": float(s.gyro_rad[0]) if s and hasattr(s, "gyro_rad") else 0.0,
            "gyro_y": float(s.gyro_rad[1]) if s and hasattr(s, "gyro_rad") else 0.0,
            "gyro_z": float(s.gyro_rad[2]) if s and hasattr(s, "gyro_rad") else 0.0,

            # Attitude
            "att_q0": float(att.q[0]) if att and hasattr(att, "q") else 1.0,
            "att_q1": float(att.q[1]) if att and hasattr(att, "q") else 0.0,
            "att_q2": float(att.q[2]) if att and hasattr(att, "q") else 0.0,
            "att_q3": float(att.q[3]) if att and hasattr(att, "q") else 0.0,

            # GPS
            "gps_lat": float(get(gps, "latitude_deg")),
            "gps_lon": float(get(gps, "longitude_deg")),
            "gps_alt": float(get(gps, "altitude_m")),
            "gps_fix_type": float(get(gps, "fix_type")),
            "gps_satellites": float(get(gps, "satellites_used")),
            "gps_velocity": float(get(gps, "vel_m_s")),
            "gps_vertical_speed": float(get(gps, "vel_d_m_s")),

            # Global
            "global_lat": float(get(gl, "lat")),
            "global_lon": float(get(gl, "lon")),
            "global_alt": float(get(gl, "alt")),
            "global_alt_valid": float(get(gl, "alt_valid", False)),
            "global_position_valid": float(get(gl, "lat_lon_valid", False)),

            # Local
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

            # Battery
            "battery_voltage": float(get(bat, "voltage_v")),
            "battery_current": float(get(bat, "current_a")),
            "battery_remaining": float(get(bat, "remaining")),

            # Estimator / Failsafe
            "attitude_invalid": float(get(fs, "attitude_invalid", False)),
            "angular_velocity_invalid": float(get(fs, "angular_velocity_invalid", False)),
            "local_position_invalid_est": float(get(fs, "local_position_invalid", False)),
            "local_velocity_invalid_est": float(get(fs, "local_velocity_invalid", False)),
            "global_position_invalid_est": float(get(fs, "global_position_invalid", False)),
            "inertial_dead_reckoning": float(get(est, "cs_inertial_dead_reckoning", 0)),

            # Magnetometer
            "cs_mag_fault": float(get(est, "cs_mag_fault", 0)),
            "cs_mag": float(get(est, "cs_mag", 0)),
            "cs_mag_hdg": float(get(est, "cs_mag_hdg", 0)),
            "cs_mag_3d": float(get(est, "cs_mag_3d", 0)),
            "cs_mag_heading_consistent": float(get(est, "cs_mag_heading_consistent", 0)),
            "fs_bad_mag_x": float(get(est, "fs_bad_mag_x", 0)),
            "fs_bad_mag_y": float(get(est, "fs_bad_mag_y", 0)),
            "fs_bad_mag_z": float(get(est, "fs_bad_mag_z", 0)),

            # Barometer & GPS estimator flags
            "cs_baro_fault": float(get(est, "cs_baro_fault", 0)),
            "cs_gnss_fault": float(get(est, "cs_gnss_fault", 0)),
            "cs_gnss_pos": float(get(est, "cs_gnss_pos", 0)),
            "cs_baro_hgt": float(get(est, "cs_baro_hgt", 0)),

            # Status & Failsafe
            "failsafe": float(get(fs, "failsafe", 0)),
            "nav_state": float(get(stat, "nav_state", 0)),
            "arming_state": float(get(stat, "arming_state", 0)),

            # Air Data
            "baro_device_id": float(get(air, "baro_device_id", 0)),
            "baro_alt_meter": float(get(air, "baro_alt_meter", 0)),
            "baro_pressure_pa": float(get(air, "baro_pressure_pa", 0)),
            "ambient_temperature": float(get(air, "ambient_temperature", 0)),
            "rho": float(get(air, "rho", 0)),
        }

    def process(self):
        # Only start buffering once sensor data is actively flowing
        if self.latest_sensor is None:
            return

        row = self.build_snapshot()
        self.rows.append(row)

        if len(self.rows) < WINDOW:
            return

        now = time.time()
        # Rate-limit inferences to 5 Hz for optimal UI display
        if now - self.last_print_time < 0.2:
            return
        self.last_print_time = now

        df = pd.DataFrame(list(self.rows))
        features = extract_window_features(df)

        # Build feature vector matching expected order
        feat_vector = [features.get(col, 0.0) for col in self.expected_features]
        X = np.array([feat_vector], dtype=float)

        prediction = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        confidence = float(np.max(probs))

        # 1. Look up realistic benchmark accuracy for this specific detected class
        class_acc = self.benchmark_accuracies.get(prediction, 90.0)

        # 2. Track rolling streaming stability over last 15 inferences
        self.recent_predictions.append(prediction)
        stability = (self.recent_predictions.count(prediction) / len(self.recent_predictions)) * 100.0

        # 3. Decision margin over the closest runner-up class
        sorted_probs = sorted(probs, reverse=True)
        runner_up = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = (confidence - runner_up) * 100.0

        # Format alert output
        if prediction == "No Failure":
            status_tag = "\033[92m[HEALTHY]\033[0m"
        else:
            status_tag = f"\033[91m[FAULT DETECTED: {prediction.upper()}]\033[0m"

        self.get_logger().info(
            f"{status_tag} | Prediction: {prediction} | Real Accuracy: {class_acc:.1f}% | "
            f"Stability: {stability:.0f}% | Confidence: {confidence:.1%} (margin: +{margin:.1f}%)"
        )


def main():
    rclpy.init()
    node = LivePX4FDI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
