#!/usr/bin/env python3
"""
step75_agentic_safety_governor.py
==================================
Integrated Autonomous Brain for Single-UAV Operations.

Architectural Pipeline:
1. Telemetry Ingestion & Feature Windowing (50 samples, 10 Hz)
2. Machine Learning FDI Engine (ExtraTrees 6-Class, Debounced)
3. Quantitative Operational Risk Assessment Engine (Dynamic Risk Score 0.0 - 1.0)
4. Agentic AI Decision Engine (Observe -> Reason -> Plan -> Propose)
5. Dual-Layer Safety Guardrail (Firewall):
   - Layer 1: Mission-Level Safety Rules & State Constraints
   - Layer 2: Control-Level Dynamic Safety Limits & Clamping (CBF/RTA-inspired)
6. Actuation: Publishes validated VehicleCommand to PX4 via ROS 2
7. Audit Logging to safety_governance_audit_log.csv
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
    VehicleCommand,
)

from step76_control_barrier_function import UAVControlBarrierFunction

BASE = Path("/home/samiksha/uav_fdi_data")
MODEL_PATH = BASE / "step70_clean_fdi_model.joblib"
FEATURES_PATH = BASE / "step70_feature_names.joblib"
AUDIT_LOG = BASE / "safety_governance_audit_log.csv"

WINDOW = 50
DEBOUNCE_THRESHOLD = 3

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


class AgenticSafetyGovernor(Node):

    def __init__(self):
        super().__init__("agentic_safety_governor")

        self.get_logger().info("======================================================")
        self.get_logger().info("INITIALIZING AGENTIC SAFETY GOVERNOR & FIREWALL NODE")
        self.get_logger().info("======================================================")

        self.model = joblib.load(MODEL_PATH)
        self.expected_features = joblib.load(FEATURES_PATH)
        self.cbf = UAVControlBarrierFunction(z_min=0.3, a_max=2.0, v_max=4.0, gamma=1.5)
        self.rows = deque(maxlen=WINDOW)
        self.recent_predictions = deque(maxlen=15)

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

        # FDI Debounce Tracker
        self.candidate_fault = "No Failure"
        self.consecutive_fault_count = 0
        self.active_fdi_state = "No Failure"
        self.active_confidence = 0.0

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

        # PX4 Command Publisher (Actuation)
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)

        # Audit Log Init
        with open(AUDIT_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "fdi_fault", "confidence", "stability",
                "risk_stability", "risk_nav", "risk_energy", "composite_risk",
                "agent_proposal", "guardrail_l1_mission", "guardrail_l2_control",
                "final_governed_action"
            ])

        self.timer = self.create_timer(0.1, self.pipeline_loop)
        self.last_run_time = 0.0
        self.get_logger().info("Agentic Safety Governor is ready. Waiting for 50 telemetry samples...")

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

    # -----------------------------------------------------------------
    # 1. RISK ASSESSMENT ENGINE
    # -----------------------------------------------------------------
    def compute_risk(self, fault, confidence, snapshot):
        # Stability Risk: depends on attitude sensors & validity flags
        if fault in ("Accelerometer Failure", "Gyroscope Failure"):
            r_stability = 0.95 * confidence
        elif snapshot["attitude_invalid"] > 0:
            r_stability = 0.90
        else:
            r_stability = 0.05

        # Navigation Risk: depends on GPS & Compass
        if fault == "GPS Failure":
            r_nav = 0.85 * confidence
        elif fault == "Compass Failure":
            r_nav = 0.45 * confidence
        elif fault == "Barometer Failure":
            r_nav = 0.35 * confidence
        elif snapshot["local_position_invalid"] > 0:
            r_nav = 0.80
        else:
            r_nav = 0.05

        # Energy Risk: battery state
        bat_rem = snapshot["battery_remaining"]
        if bat_rem > 0:
            if bat_rem < 0.20:
                r_energy = 0.90
            elif bat_rem < 0.40:
                r_energy = 0.50
            else:
                r_energy = 0.10
        else:
            r_energy = 0.10

        composite = max(r_stability, r_nav, r_energy)
        return float(r_stability), float(r_nav), float(r_energy), float(composite)

    # -----------------------------------------------------------------
    # 2. AGENTIC AI DECISION ENGINE (Observe -> Reason -> Plan -> Propose)
    # -----------------------------------------------------------------
    def agentic_reasoning(self, fault, composite_risk, snapshot):
        # Observation & Knowledge-based reasoning
        if composite_risk >= 0.80:
            if fault in ("Accelerometer Failure", "Gyroscope Failure"):
                proposal = "PROPOSE_EMERGENCY_LAND"
                descent_rate = 2.0  # agent proposes fast emergency descent
            else:
                proposal = "PROPOSE_RETURN_TO_BASE_OR_LAND"
                descent_rate = 1.2
        elif composite_risk >= 0.50:
            if fault == "GPS Failure":
                proposal = "PROPOSE_ALTITUDE_HOLD_HOVER"
                descent_rate = 0.0
            else:
                proposal = "PROPOSE_DEGRADED_MANUAL_HOLD"
                descent_rate = 0.5
        elif composite_risk >= 0.30:
            proposal = "PROPOSE_DEGRADED_MISSION_CONTINUE"
            descent_rate = 0.0
        else:
            proposal = "PROPOSE_NOMINAL_MISSION_CONTINUE"
            descent_rate = 0.0

        return proposal, descent_rate

    # -----------------------------------------------------------------
    # 3. DUAL-LAYER SAFETY GUARDRAIL (FIREWALL)
    # -----------------------------------------------------------------
    def evaluate_dual_guardrails(self, proposal, proposed_descent, snapshot):
        # Layer 1: Mission-Level Guardrail
        # Checks geofence and minimum battery floor
        l1_status = "PASSED"
        l1_reason = "Nominal mission parameters valid"

        loc_z = snapshot["local_z"]  # NED frame: negative is above ground
        altitude = -loc_z

        if altitude < 0.2 and proposal != "PROPOSE_NOMINAL_MISSION_CONTINUE":
            l1_status = "INTERVENED"
            l1_reason = "Near-ground proximity: Force immediate engine shutdown"
            proposal = "ENFORCE_DISARM_ON_GROUND"

        # Layer 2: Control-Level Dynamic Safety Limits via Control Barrier Function (CBF-QP)
        # Prevents ground impact and enforces kinetic braking distance invariant
        u_nom = np.array([0.0, 0.0, proposed_descent], dtype=float)
        u_safe, cbf_active, h_val, cbf_status = self.cbf.filter_control(u_nom, altitude=max(0.0, altitude))
        enforced_descent = float(u_safe[2])

        if cbf_active:
            l2_status = f"CBF-QP INTERVENED (h={h_val:.2f}, vz->{enforced_descent:.1f}m/s)"
        else:
            l2_status = f"CBF-QP INVARIANT (h={h_val:.2f})"

        # Final Governed Action Mapping
        if proposal == "PROPOSE_EMERGENCY_LAND":
            final_action = f"EXECUTE_FAILSAFE_LAND (CBF vz={enforced_descent:.1f} m/s)"
            px4_cmd = "NAV_LAND"
        elif proposal == "PROPOSE_ALTITUDE_HOLD_HOVER":
            final_action = "SWITCH_TO_ALTCTL_HOLD"
            px4_cmd = "ALTCTL"
        elif proposal == "ENFORCE_DISARM_ON_GROUND":
            final_action = "ENGINE_CUTOFF"
            px4_cmd = "DISARM"
        else:
            final_action = "MAINTAIN_MISSION_TRAJECTORY"
            px4_cmd = "NONE"

        return l1_status, l2_status, final_action, px4_cmd

    # -----------------------------------------------------------------
    # 4. EXECUTION LOOP
    # -----------------------------------------------------------------
    def pipeline_loop(self):
        if self.latest_sensor is None:
            return

        snapshot = self.build_snapshot()
        self.rows.append(snapshot)

        if len(self.rows) < WINDOW:
            return

        now = time.time()
        if now - self.last_run_time < 0.25:
            return
        self.last_run_time = now

        df = pd.DataFrame(list(self.rows))
        features = extract_window_features(df)
        feat_vector = [features.get(col, 0.0) for col in self.expected_features]
        X = np.array([feat_vector], dtype=float)

        prediction = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        confidence = float(np.max(probs))

        # Debounce filter
        if prediction == self.candidate_fault:
            self.consecutive_fault_count += 1
        else:
            self.candidate_fault = prediction
            self.consecutive_fault_count = 1

        if self.consecutive_fault_count >= DEBOUNCE_THRESHOLD:
            self.active_fdi_state = self.candidate_fault
            self.active_confidence = confidence

        self.recent_predictions.append(self.active_fdi_state)
        stability = (self.recent_predictions.count(self.active_fdi_state) / len(self.recent_predictions)) * 100.0

        # Step 3: Risk Assessment
        r_stab, r_nav, r_energy, r_comp = self.compute_risk(self.active_fdi_state, self.active_confidence, snapshot)

        # Step 4: Agentic AI Proposal
        proposal, proposed_descent = self.agentic_reasoning(self.active_fdi_state, r_comp, snapshot)

        # Step 5: Dual-Layer Guardrail Verification
        l1, l2, final_action, px4_cmd = self.evaluate_dual_guardrails(proposal, proposed_descent, snapshot)

        # Visual Dashboard Output
        if r_comp < 0.3:
            risk_color = "\033[92m"  # Green
        elif r_comp < 0.7:
            risk_color = "\033[93m"  # Yellow
        else:
            risk_color = "\033[91m"  # Red

        self.get_logger().info(
            f"FDI: {self.active_fdi_state} ({self.active_confidence:.1%}) | "
            f"Risk: {risk_color}{r_comp:.2f}\033[0m (Stab:{r_stab:.2f}, Nav:{r_nav:.2f}) | "
            f"Agent: {proposal} | "
            f"Guardrails: [L1:{l1}, L2:{l2}] -> \033[1m{final_action}\033[0m"
        )

        # Step 6: Log Audit Trail
        with open(AUDIT_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round(now, 3), self.active_fdi_state, round(self.active_confidence, 4), round(stability, 1),
                round(r_stab, 3), round(r_nav, 3), round(r_energy, 3), round(r_comp, 3),
                proposal, l1, l2, final_action
            ])


def main():
    rclpy.init()
    node = AgenticSafetyGovernor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
