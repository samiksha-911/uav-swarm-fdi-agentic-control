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
    InputRc,
    ManualControlSetpoint,
)


class PX4TelemetryCollector(Node):

    def __init__(self):
        super().__init__("step42b_live_px4_telemetry_collector")

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
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.attitude_callback,
            qos
        )

        self.create_subscription(
            SensorGps,
            "/fmu/out/vehicle_gps_position",
            self.gps_callback,
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
            self.vehicle_status_callback,
            qos
        )

        self.create_subscription(
            InputRc,
            "/fmu/out/input_rc",
            self.rc_callback,
            qos
        )

        self.create_subscription(
            ManualControlSetpoint,
            "/fmu/out/manual_control_setpoint",
            self.manual_callback,
            qos
        )

        self.timer = self.create_timer(2.0, self.report)

        self.get_logger().info(
            "PX4 telemetry collector started"
        )

    def sensor_callback(self, msg):
        self.data["sensor_combined"] = msg

    def attitude_callback(self, msg):
        self.data["vehicle_attitude"] = msg

    def gps_callback(self, msg):
        self.data["vehicle_gps_position"] = msg

    def global_position_callback(self, msg):
        self.data["vehicle_global_position"] = msg

    def local_position_callback(self, msg):
        self.data["vehicle_local_position"] = msg

    def battery_callback(self, msg):
        self.data["battery_status"] = msg

    def estimator_callback(self, msg):
        self.data["estimator_status_flags"] = msg

    def failsafe_callback(self, msg):
        self.data["failsafe_flags"] = msg

    def vehicle_status_callback(self, msg):
        self.data["vehicle_status"] = msg

    def rc_callback(self, msg):
        self.data["input_rc"] = msg

    def manual_callback(self, msg):
        self.data["manual_control_setpoint"] = msg

    def report(self):

        expected = [
            "sensor_combined",
            "vehicle_attitude",
            "vehicle_gps_position",
            "vehicle_global_position",
            "vehicle_local_position",
            "battery_status",
            "estimator_status_flags",
            "failsafe_flags",
            "vehicle_status",
            "input_rc",
            "manual_control_setpoint",
        ]

        print("\n" + "=" * 70)
        print("LIVE PX4 TELEMETRY STATUS")
        print("=" * 70)

        for name in expected:
            if name in self.data:
                print(f"[OK]      {name}")
            else:
                print(f"[MISSING] {name}")

        if "vehicle_gps_position" in self.data:
            msg = self.data["vehicle_gps_position"]

            print("\nGPS:")
            print(f"  satellites_used = {msg.satellites_used}")
            print(f"  latitude_deg    = {msg.latitude_deg}")
            print(f"  longitude_deg   = {msg.longitude_deg}")
            print(f"  altitude_msl_m  = {msg.altitude_msl_m}")
            print(f"  vel_m_s         = {msg.vel_m_s}")
            print(f"  vel_d_m_s       = {msg.vel_d_m_s}")
            print(f"  fix_type        = {msg.fix_type}")

        if "battery_status" in self.data:
            msg = self.data["battery_status"]

            print("\nBATTERY:")
            print(f"  voltage_v       = {msg.voltage_v}")
            print(f"  current_a       = {msg.current_a}")
            print(f"  remaining       = {msg.remaining}")
            print(f"  connected       = {msg.connected}")

        if "sensor_combined" in self.data:
            msg = self.data["sensor_combined"]

            print("\nIMU:")
            print(f"  accelerometer   = {list(msg.accelerometer_m_s2)}")
            print(f"  gyro            = {list(msg.gyro_rad)}")
            print(f"  accel_clip      = {msg.accelerometer_clipping}")
            print(f"  gyro_clip       = {msg.gyro_clipping}")

        if "vehicle_local_position" in self.data:
            msg = self.data["vehicle_local_position"]

            print("\nLOCAL POSITION:")
            print(f"  x               = {msg.x}")
            print(f"  y               = {msg.y}")
            print(f"  z               = {msg.z}")
            print(f"  vx              = {msg.vx}")
            print(f"  vy              = {msg.vy}")
            print(f"  vz              = {msg.vz}")
            print(f"  xy_valid        = {msg.xy_valid}")
            print(f"  z_valid         = {msg.z_valid}")
            print(f"  v_xy_valid      = {msg.v_xy_valid}")
            print(f"  v_z_valid       = {msg.v_z_valid}")
            print(f"  heading         = {msg.heading}")

        if "estimator_status_flags" in self.data:
            msg = self.data["estimator_status_flags"]

            print("\nESTIMATOR:")
            print(f"  cs_mag_hdg      = {msg.cs_mag_hdg}")
            print(f"  cs_mag_3d       = {msg.cs_mag_3d}")
            print(f"  cs_mag_dec      = {msg.cs_mag_dec}")
            print(f"  cs_mag_fault    = {msg.cs_mag_fault}")
            print(f"  cs_gnss_pos     = {msg.cs_gnss_pos}")
            print(f"  cs_gnss_fault   = {msg.cs_gnss_fault}")
            print(f"  dead_reckoning  = {msg.cs_inertial_dead_reckoning}")

        if "failsafe_flags" in self.data:
            msg = self.data["failsafe_flags"]

            print("\nFAILSAFE:")
            print(f"  global_pos_bad  = {msg.global_position_invalid}")
            print(f"  local_pos_bad   = {msg.local_position_invalid}")
            print(f"  local_vel_bad   = {msg.local_velocity_invalid}")
            print(f"  attitude_bad    = {msg.attitude_invalid}")
            print(f"  angular_vel_bad = {msg.angular_velocity_invalid}")
            print(f"  gnss_lost       = {msg.gnss_lost}")
            print(f"  battery_bad     = {msg.battery_unhealthy}")

        print()


def main():
    rclpy.init()

    node = PX4TelemetryCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
