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
)

BASE = Path("/home/samiksha/uav_fdi_data")
MODEL_PATH = BASE / "px4_native_6class_model.joblib"

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


def add_window_features(df):
    features = {}

    numeric_cols = df.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c != "time"]

    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if len(x) != WINDOW or not np.isfinite(x).all():
            continue

        features[f"{col}_mean"] = float(np.mean(x))
        features[f"{col}_std"] = float(np.std(x))
        features[f"{col}_min"] = float(np.min(x))
        features[f"{col}_max"] = float(np.max(x))
        features[f"{col}_range"] = float(np.max(x) - np.min(x))

        features[f"{col}_first"] = float(x[0])
        features[f"{col}_last"] = float(x[-1])
        features[f"{col}_delta"] = float(x[-1] - x[0])

        dx = np.diff(x)

        features[f"{col}_mean_abs_change"] = float(np.mean(np.abs(dx)))
        features[f"{col}_max_abs_change"] = float(np.max(np.abs(dx)))
        features[f"{col}_change_std"] = float(np.std(dx))
        features[f"{col}_change_range"] = float(np.max(dx) - np.min(dx))

    return features


class LiveFDI(Node):

    def __init__(self):
        super().__init__("live_px4_fdi")

        self.model = joblib.load(MODEL_PATH)
        self.expected_features = list(self.model.feature_names_in_)

        self.rows = deque(maxlen=WINDOW)

        self.latest = {
            "acc_x": 0.0,
            "acc_y": 0.0,
            "acc_z": 0.0,

            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,

            "att_q1": 1.0,
            "att_q2": 0.0,
            "att_q3": 0.0,
            "att_q4": 0.0,

            "gps_lat": 0.0,
            "gps_lon": 0.0,
            "gps_alt": 0.0,
            "gps_vertical_speed": 0.0,
            "gps_fix_type": 0.0,
            "gps_satellites": 0.0,

            "global_x": 0.0,
            "global_y": 0.0,
            "global_z": 0.0,
            "global_lat": 0.0,
            "global_lon": 0.0,
            "global_alt": 0.0,

            "local_x": 0.0,
            "local_y": 0.0,
            "local_z": 0.0,

            "local_vx": 0.0,
            "local_vy": 0.0,
            "local_vz": 0.0,

            "attitude_invalid": 0.0,
            "local_position_invalid": 0.0,
            "local_velocity_invalid": 0.0,
            "global_position_invalid": 0.0,
            "global_position_valid": 0.0,
            "local_z_valid": 0.0,
            "local_vz_valid": 0.0,
            "angular_velocity_invalid": 0.0,
            "inertial_dead_reckoning": 0.0,
            "tilt_align": 0.0,
            "yaw_align": 0.0,
            "gnss_position_ok": 0.0,
            "gnss_fault": 0.0,
            "gnss_lost": 0.0,
            "failsafe": 0.0,

            "nav_state": 0.0,
            "arming_state": 0.0,

            "battery_voltage": 0.0,
            "battery_current": 0.0,
            "battery_remaining": 0.0,
        }

        self.create_subscription(
            SensorCombined,
            "/fmu/out/sensor_combined",
            self.sensor_cb,
            qos,
        )

        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.attitude_cb,
            qos,
        )

        self.create_subscription(
            SensorGps,
            "/fmu/out/vehicle_gps_position",
            self.gps_cb,
            qos,
        )

        self.create_subscription(
            VehicleGlobalPosition,
            "/fmu/out/vehicle_global_position",
            self.global_cb,
            qos,
        )

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.local_cb,
            qos,
        )

        self.create_subscription(
            BatteryStatus,
            "/fmu/out/battery_status",
            self.battery_cb,
            qos,
        )

        self.create_subscription(
            EstimatorStatusFlags,
            "/fmu/out/estimator_status_flags",
            self.estimator_cb,
            qos,
        )

        self.create_subscription(
            FailsafeFlags,
            "/fmu/out/failsafe_flags",
            self.failsafe_cb,
            qos,
        )

        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v4",
            self.status_cb,
            qos,
        )

        self.last_prediction = None
        self.debug_count = 0

        self.timer = self.create_timer(0.1, self.process)

        self.get_logger().info(
            f"Loaded model: {MODEL_PATH.name}"
        )

        self.get_logger().info(
            f"Expected features: {len(self.expected_features)}"
        )

        self.get_logger().info(
            "Waiting for 50 live telemetry samples..."
        )

    def sensor_cb(self, msg):
        acc = getattr(msg, "accelerometer_m_s2", [0.0, 0.0, 0.0])
        gyro = getattr(msg, "gyro_rad", [0.0, 0.0, 0.0])

        self.latest["acc_x"] = float(acc[0])
        self.latest["acc_y"] = float(acc[1])
        self.latest["acc_z"] = float(acc[2])

        self.latest["gyro_x"] = float(gyro[0])
        self.latest["gyro_y"] = float(gyro[1])
        self.latest["gyro_z"] = float(gyro[2])

    def attitude_cb(self, msg):
        q = list(get(msg, "q", [1, 0, 0, 0]))

        if len(q) >= 4:
            for i in range(4):
                self.latest[f"att_q{i+1}"] = float(q[i])

    def gps_cb(self, msg):
        self.latest["gps_lat"] = get(msg, "latitude_deg")
        self.latest["gps_lon"] = get(msg, "longitude_deg")
        self.latest["gps_alt"] = get(msg, "altitude_msl_m")
        self.latest["gps_vertical_speed"] = get(msg, "vel_d_m_s")
        self.latest["gps_fix_type"] = get(msg, "fix_type")
        self.latest["gps_satellites"] = get(msg, "satellites_used")

    def global_cb(self, msg):
        self.latest["global_x"] = get(msg, "lat")
        self.latest["global_y"] = get(msg, "lon")
        self.latest["global_z"] = get(msg, "alt")
        self.latest["global_lat"] = get(msg, "lat")
        self.latest["global_lon"] = get(msg, "lon")
        self.latest["global_alt"] = get(msg, "alt")
        self.latest["global_position_valid"] = float(get(msg, "lat_lon_valid", False) and get(msg, "alt_valid", False))

    def local_cb(self, msg):
        self.latest["local_x"] = get(msg, "x")
        self.latest["local_y"] = get(msg, "y")
        self.latest["local_z"] = get(msg, "z")

        self.latest["local_vx"] = get(msg, "vx")
        self.latest["local_vy"] = get(msg, "vy")
        self.latest["local_vz"] = get(msg, "vz")

        self.latest["attitude_invalid"] = float(
            get(msg, "heading_good", True) == False
        )

        self.latest["local_position_invalid"] = float(
            get(msg, "xy_valid", True) == False
        )

        self.latest["local_velocity_invalid"] = float(
            get(msg, "v_xy_valid", True) == False
        )

        self.latest["local_z_valid"] = float(
            get(msg, "z_valid", True)
        )

        self.latest["local_vz_valid"] = float(
            get(msg, "v_z_valid", True)
        )

    def battery_cb(self, msg):
        self.latest["battery_voltage"] = get(msg, "voltage_v")
        self.latest["battery_current"] = get(msg, "current_a")
        self.latest["battery_remaining"] = get(msg, "remaining")

    def estimator_cb(self, msg):
        mapping = {
            "tilt_align": "cs_tilt_align",
            "yaw_align": "cs_yaw_align",
            "gnss_position_ok": "cs_gnss_pos",
            "inertial_dead_reckoning": "cs_inertial_dead_reckoning",
            "gnss_fault": "cs_gnss_fault",
        }
        for out_name, msg_name in mapping.items():
            if hasattr(msg, msg_name):
                self.latest[out_name] = float(getattr(msg, msg_name))

        if hasattr(msg, "cs_baro_fault"):
            self.latest["baro_fault"] = float(msg.cs_baro_fault)

    def failsafe_cb(self, msg):
        values = []

        for name in dir(msg):
            if name.startswith("_"):
                continue

            try:
                value = getattr(msg, name)
                if isinstance(value, bool):
                    values.append(value)
            except Exception:
                pass

        self.latest["failsafe"] = float(any(values))

    def status_cb(self, msg):
        self.latest["nav_state"] = get(msg, "nav_state")
        self.latest["arming_state"] = get(msg, "arming_state")

    def process(self):
        row = dict(self.latest)
        row["time"] = time.time()

        self.rows.append(row)

        if len(self.rows) < WINDOW:
            return

        df = pd.DataFrame(list(self.rows))

        self.debug_count += 1

        if self.debug_count == 1:
            self.get_logger().info(
                "LIVE INPUT CHECK: "
                f"acc=({df['acc_x'].iloc[-1]:.3f},"
                f"{df['acc_y'].iloc[-1]:.3f},"
                f"{df['acc_z'].iloc[-1]:.3f}) "
                f"gyro=({df['gyro_x'].iloc[-1]:.3f},"
                f"{df['gyro_y'].iloc[-1]:.3f},"
                f"{df['gyro_z'].iloc[-1]:.3f}) "
                f"local=({df['local_x'].iloc[-1]:.3f},"
                f"{df['local_y'].iloc[-1]:.3f},"
                f"{df['local_z'].iloc[-1]:.3f}) "
                f"gps=({df['gps_lat'].iloc[-1]:.6f},"
                f"{df['gps_lon'].iloc[-1]:.6f},"
                f"{df['gps_alt'].iloc[-1]:.3f})"
            )

        features = add_window_features(df)

        X = pd.DataFrame([features])

        # Force EXACT model feature order.
        X = X.reindex(columns=self.expected_features)

        if X.isna().any().any():
            missing = X.columns[X.isna().any()].tolist()
            self.get_logger().warning(
                f"Missing model features: {len(missing)}"
            )
            return

        prediction = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]

        confidence = float(np.max(probabilities))

        probs = dict(zip(self.model.classes_, probabilities))

        self.get_logger().info(
            f"FDI: {prediction} | confidence={confidence:.3f} | "
            f"probs={probs}"
        )

        self.last_prediction = prediction


def main():
    rclpy.init()

    node = LiveFDI()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
