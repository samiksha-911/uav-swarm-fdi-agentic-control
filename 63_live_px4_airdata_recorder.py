#!/usr/bin/env python3

import csv
import time
import threading
from pathlib import Path

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

OUT = Path("live_px4_airdata_recording.csv")
RATE_HZ = 10.0


class Recorder(Node):
    def __init__(self):
        super().__init__("px4_airdata_recorder")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.lock = threading.Lock()
        self.data = {}

        self.create_subscription(
            SensorCombined, "/fmu/out/sensor_combined",
            lambda m: self.store("sensor", m), qos)

        self.create_subscription(
            VehicleAttitude, "/fmu/out/vehicle_attitude",
            lambda m: self.store("att", m), qos)

        self.create_subscription(
            SensorGps, "/fmu/out/vehicle_gps_position",
            lambda m: self.store("gps", m), qos)

        self.create_subscription(
            VehicleGlobalPosition, "/fmu/out/vehicle_global_position",
            lambda m: self.store("global", m), qos)

        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1",
            lambda m: self.store("local", m), qos)

        self.create_subscription(
            BatteryStatus, "/fmu/out/battery_status_v1",
            lambda m: self.store("battery", m), qos)

        self.create_subscription(
            EstimatorStatusFlags, "/fmu/out/estimator_status_flags",
            lambda m: self.store("estimator", m), qos)

        self.create_subscription(
            FailsafeFlags, "/fmu/out/failsafe_flags",
            lambda m: self.store("failsafe", m), qos)

        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v4",
            lambda m: self.store("status", m), qos)

        self.create_subscription(
            VehicleAirData, "/fmu/out/vehicle_air_data",
            lambda m: self.store("air", m), qos)

        self.timer = self.create_timer(1.0 / RATE_HZ, self.write_row)

        self.header_written = False
        self.start_time = time.time()
        self.rows = 0

    def store(self, key, msg):
        with self.lock:
            self.data[key] = msg

    def get(self, obj, name, default=0):
        return getattr(obj, name, default)

    def write_row(self):
        with self.lock:
            d = dict(self.data)

        now = time.time()

        row = {
            "time": now - self.start_time,

            # SensorCombined
            "acc_x": self.get(d.get("sensor"), "accelerometer_m_s2")[0]
                if d.get("sensor") and hasattr(d["sensor"], "accelerometer_m_s2") else 0,
            "acc_y": self.get(d.get("sensor"), "accelerometer_m_s2")[1]
                if d.get("sensor") and hasattr(d["sensor"], "accelerometer_m_s2") else 0,
            "acc_z": self.get(d.get("sensor"), "accelerometer_m_s2")[2]
                if d.get("sensor") and hasattr(d["sensor"], "accelerometer_m_s2") else 0,

            "gyro_x": self.get(d.get("sensor"), "gyro_rad")[0]
                if d.get("sensor") and hasattr(d["sensor"], "gyro_rad") else 0,
            "gyro_y": self.get(d.get("sensor"), "gyro_rad")[1]
                if d.get("sensor") and hasattr(d["sensor"], "gyro_rad") else 0,
            "gyro_z": self.get(d.get("sensor"), "gyro_rad")[2]
                if d.get("sensor") and hasattr(d["sensor"], "gyro_rad") else 0,

            # Attitude
            "att_q0": self.get(d.get("att"), "q")[0]
                if d.get("att") and hasattr(d["att"], "q") else 0,
            "att_q1": self.get(d.get("att"), "q")[1]
                if d.get("att") and hasattr(d["att"], "q") else 0,
            "att_q2": self.get(d.get("att"), "q")[2]
                if d.get("att") and hasattr(d["att"], "q") else 0,
            "att_q3": self.get(d.get("att"), "q")[3]
                if d.get("att") and hasattr(d["att"], "q") else 0,

            # GPS
            "gps_lat": self.get(d.get("gps"), "latitude_deg"),
            "gps_lon": self.get(d.get("gps"), "longitude_deg"),
            "gps_alt": self.get(d.get("gps"), "altitude_m"),
            "gps_fix_type": self.get(d.get("gps"), "fix_type"),
            "gps_satellites": self.get(d.get("gps"), "satellites_used"),
            "gps_velocity": self.get(d.get("gps"), "vel_m_s"),
            "gps_vertical_speed": self.get(d.get("gps"), "vel_d_m_s"),

            # Global position
            "global_lat": self.get(d.get("global"), "lat"),
            "global_lon": self.get(d.get("global"), "lon"),
            "global_alt": self.get(d.get("global"), "alt"),
            "global_alt_valid": self.get(d.get("global"), "alt_valid"),
            "global_position_valid": self.get(d.get("global"), "lat_lon_valid"),

            # Local position
            "local_x": self.get(d.get("local"), "x"),
            "local_y": self.get(d.get("local"), "y"),
            "local_z": self.get(d.get("local"), "z"),
            "local_vx": self.get(d.get("local"), "vx"),
            "local_vy": self.get(d.get("local"), "vy"),
            "local_vz": self.get(d.get("local"), "vz"),
            "local_z_valid": self.get(d.get("local"), "z_valid"),
            "local_vz_valid": self.get(d.get("local"), "v_z_valid"),
            "local_position_invalid": self.get(d.get("local"), "xy_valid") == False,
            "local_velocity_invalid": self.get(d.get("local"), "v_xy_valid") == False,

            # Battery
            "battery_voltage": self.get(d.get("battery"), "voltage_v"),
            "battery_current": self.get(d.get("battery"), "current_a"),
            "battery_remaining": self.get(d.get("battery"), "remaining"),

            # Estimator / Failsafe
            "attitude_invalid": self.get(d.get("failsafe"), "attitude_invalid", False),
            "angular_velocity_invalid": self.get(d.get("failsafe"), "angular_velocity_invalid", False),
            "local_position_invalid_est": self.get(d.get("failsafe"), "local_position_invalid", False),
            "local_velocity_invalid_est": self.get(d.get("failsafe"), "local_velocity_invalid", False),
            "global_position_invalid_est": self.get(d.get("failsafe"), "global_position_invalid", False),
            "inertial_dead_reckoning": self.get(d.get("estimator"), "cs_inertial_dead_reckoning", 0),

            # Magnetometer estimator flags
            "cs_mag_fault":              self.get(d.get("estimator"), "cs_mag_fault", 0),
            "cs_mag":                    self.get(d.get("estimator"), "cs_mag", 0),
            "cs_mag_hdg":                self.get(d.get("estimator"), "cs_mag_hdg", 0),
            "cs_mag_3d":                 self.get(d.get("estimator"), "cs_mag_3d", 0),
            "cs_mag_heading_consistent": self.get(d.get("estimator"), "cs_mag_heading_consistent", 0),
            "fs_bad_mag_x":              self.get(d.get("estimator"), "fs_bad_mag_x", 0),
            "fs_bad_mag_y":              self.get(d.get("estimator"), "fs_bad_mag_y", 0),
            "fs_bad_mag_z":              self.get(d.get("estimator"), "fs_bad_mag_z", 0),

            # Barometer and GPS estimator fault flags
            "cs_baro_fault":             self.get(d.get("estimator"), "cs_baro_fault", 0),
            "cs_gnss_fault":             self.get(d.get("estimator"), "cs_gnss_fault", 0),
            "cs_gnss_pos":               self.get(d.get("estimator"), "cs_gnss_pos", 0),
            "cs_baro_hgt":               self.get(d.get("estimator"), "cs_baro_hgt", 0),

            # Failsafe
            "failsafe": self.get(d.get("failsafe"), "failsafe"),

            # Vehicle status
            "nav_state": self.get(d.get("status"), "nav_state"),
            "arming_state": self.get(d.get("status"), "arming_state"),

            # NEW: VehicleAirData
            "baro_device_id": self.get(d.get("air"), "baro_device_id"),
            "baro_alt_meter": self.get(d.get("air"), "baro_alt_meter"),
            "baro_pressure_pa": self.get(d.get("air"), "baro_pressure_pa"),
            "ambient_temperature": self.get(d.get("air"), "ambient_temperature"),
            "rho": self.get(d.get("air"), "rho"),
        }

        fields = list(row.keys())

        with open(OUT, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            if not self.header_written:
                writer.writeheader()
                self.header_written = True

            writer.writerow(row)
            self.rows += 1

        if self.rows % 100 == 0:
            print(f"Recorded {self.rows} samples")


def main():
    rclpy.init()
    node = Recorder()

    print("PX4 AirData recorder started")
    print(f"Output: {OUT.resolve()}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Recorded {node.rows} samples")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
