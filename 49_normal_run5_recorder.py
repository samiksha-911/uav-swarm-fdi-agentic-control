import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import (
    SensorCombined,
    SensorGps,
    VehicleAttitude,
    VehicleGlobalPosition,
    VehicleLocalPosition,
    BatteryStatus,
    EstimatorStatusFlags,
    FailsafeFlags,
    VehicleStatus,
)


OUTPUT = os.path.expanduser(
    "~/uav_fdi_data/step49_normal_run5.csv"
)


class PX4Recorder(Node):

    def __init__(self):

        super().__init__("live_px4_recorder")

        self.data = {}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            SensorCombined,
            "/fmu/out/sensor_combined",
            self.sensor_callback,
            qos
        )

        self.create_subscription(
            SensorGps,
            "/fmu/out/vehicle_gps_position",
            self.gps_callback,
            qos
        )

        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.attitude_callback,
            qos
        )

        self.create_subscription(
            VehicleGlobalPosition,
            "/fmu/out/vehicle_global_position",
            self.global_position_callback,
            qos
        )

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.local_position_callback,
            qos
        )

        self.create_subscription(
            BatteryStatus,
            "/fmu/out/battery_status_v1",
            self.battery_callback,
            qos
        )

        self.create_subscription(
            EstimatorStatusFlags,
            "/fmu/out/estimator_status_flags",
            self.estimator_callback,
            qos
        )

        self.create_subscription(
            FailsafeFlags,
            "/fmu/out/failsafe_flags",
            self.failsafe_callback,
            qos
        )

        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v4",
            self.status_callback,
            qos
        )

        self.csv_file = open(
            OUTPUT,
            "w",
            newline=""
        )

        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.fieldnames()
        )

        self.writer.writeheader()

        self.start_time = time.time()
        self.sample_count = 0

        self.timer = self.create_timer(
            0.1,
            self.record_sample
        )

        self.get_logger().info(
            f"Recording live PX4 telemetry to: {OUTPUT}"
        )

    def fieldnames(self):

        fields = [
            "time",

            # IMU
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "acc_clip",
            "gyro_clip",

            # GPS
            "gps_satellites",
            "gps_lat",
            "gps_lon",
            "gps_alt",
            "gps_speed",
            "gps_vertical_speed",
            "gps_fix_type",

            # Attitude
            "att_q1",
            "att_q2",
            "att_q3",
            "att_q4",

            # Global position
            "global_lat",
            "global_lon",
            "global_alt",
            "global_alt_valid",
            "global_position_valid",
            "global_dead_reckoning",

            # Local position
            "local_x",
            "local_y",
            "local_z",
            "local_vx",
            "local_vy",
            "local_vz",
            "local_heading",
            "local_xy_valid",
            "local_z_valid",
            "local_vxy_valid",
            "local_vz_valid",
            "local_dead_reckoning",

            # Battery
            "battery_voltage",
            "battery_current",
            "battery_remaining",
            "battery_connected",
            "battery_warning",

            # Estimator
            "mag_heading_ok",
            "mag_3d_ok",
            "mag_declination_ok",
            "mag_fault",
            "mag_field_disturbed",
            "gnss_position_ok",
            "gnss_fault",
            "inertial_dead_reckoning",
            "yaw_align",
            "tilt_align",

            # Failsafe
            "global_position_invalid",
            "local_position_invalid",
            "local_velocity_invalid",
            "attitude_invalid",
            "angular_velocity_invalid",
            "gnss_lost",
            "battery_unhealthy",

            # Vehicle status
            "armed",
            "failsafe",
            "nav_state",
        ]

        return fields

    def sensor_callback(self, msg):
        self.data["sensor"] = msg

    def gps_callback(self, msg):
        self.data["gps"] = msg

    def attitude_callback(self, msg):
        self.data["attitude"] = msg

    def global_position_callback(self, msg):
        self.data["global"] = msg

    def local_position_callback(self, msg):
        self.data["local"] = msg

    def battery_callback(self, msg):
        self.data["battery"] = msg

    def estimator_callback(self, msg):
        self.data["estimator"] = msg

    def failsafe_callback(self, msg):
        self.data["failsafe"] = msg

    def status_callback(self, msg):
        self.data["status"] = msg

    def record_sample(self):

        required = [
            "sensor",
            "gps",
            "attitude",
            "global",
            "local",
            "battery",
            "estimator",
            "failsafe",
            "status",
        ]

        if not all(name in self.data for name in required):
            return

        s = self.data["sensor"]
        g = self.data["gps"]
        a = self.data["attitude"]
        gp = self.data["global"]
        lp = self.data["local"]
        b = self.data["battery"]
        e = self.data["estimator"]
        f = self.data["failsafe"]
        st = self.data["status"]

        row = {
            "time": time.time() - self.start_time,

            "acc_x": s.accelerometer_m_s2[0],
            "acc_y": s.accelerometer_m_s2[1],
            "acc_z": s.accelerometer_m_s2[2],

            "gyro_x": s.gyro_rad[0],
            "gyro_y": s.gyro_rad[1],
            "gyro_z": s.gyro_rad[2],

            "acc_clip": s.accelerometer_clipping,
            "gyro_clip": s.gyro_clipping,

            "gps_satellites": g.satellites_used,
            "gps_lat": g.latitude_deg,
            "gps_lon": g.longitude_deg,
            "gps_alt": g.altitude_msl_m,
            "gps_speed": g.vel_m_s,
            "gps_vertical_speed": g.vel_d_m_s,
            "gps_fix_type": g.fix_type,

            "att_q1": a.q[0],
            "att_q2": a.q[1],
            "att_q3": a.q[2],
            "att_q4": a.q[3],

            "global_lat": gp.lat,
            "global_lon": gp.lon,
            "global_alt": gp.alt,
            "global_alt_valid": gp.alt_valid,
            "global_position_valid": gp.lat_lon_valid,
            "global_dead_reckoning": gp.dead_reckoning,

            "local_x": lp.x,
            "local_y": lp.y,
            "local_z": lp.z,

            "local_vx": lp.vx,
            "local_vy": lp.vy,
            "local_vz": lp.vz,

            "local_heading": lp.heading,

            "local_xy_valid": lp.xy_valid,
            "local_z_valid": lp.z_valid,
            "local_vxy_valid": lp.v_xy_valid,
            "local_vz_valid": lp.v_z_valid,
            "local_dead_reckoning": lp.dead_reckoning,

            "battery_voltage": b.voltage_v,
            "battery_current": b.current_a,
            "battery_remaining": b.remaining,
            "battery_connected": b.connected,
            "battery_warning": b.warning,

            "mag_heading_ok": e.cs_mag_hdg,
            "mag_3d_ok": e.cs_mag_3d,
            "mag_declination_ok": e.cs_mag_dec,
            "mag_fault": e.cs_mag_fault,
            "mag_field_disturbed": e.cs_mag_field_disturbed,

            "gnss_position_ok": e.cs_gnss_pos,
            "gnss_fault": e.cs_gnss_fault,

            "inertial_dead_reckoning":
                e.cs_inertial_dead_reckoning,

            "yaw_align": e.cs_yaw_align,
            "tilt_align": e.cs_tilt_align,

            "global_position_invalid":
                f.global_position_invalid,

            "local_position_invalid":
                f.local_position_invalid,

            "local_velocity_invalid":
                f.local_velocity_invalid,

            "attitude_invalid":
                f.attitude_invalid,

            "angular_velocity_invalid":
                f.angular_velocity_invalid,

            "gnss_lost":
                f.gnss_lost,

            "battery_unhealthy":
                f.battery_unhealthy,

            "armed": st.arming_state,
            "failsafe": st.failsafe,
            "nav_state": st.nav_state,
        }

        self.writer.writerow(row)
        self.csv_file.flush()

        self.sample_count += 1

        if self.sample_count % 50 == 0:
            self.get_logger().info(
                f"Recorded {self.sample_count} samples"
            )

    def destroy_node(self):

        self.csv_file.close()

        self.get_logger().info(
            f"Saved {self.sample_count} samples"
        )

        super().destroy_node()


def main():

    rclpy.init()

    node = PX4Recorder()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
