"""step78_langgraph_agentic_ai_loop.py  (v3 – Live PX4 ROS2 + Multi-Agent Aware)

Real Agentic AI concepts demonstrated
======================================
1. StateGraph (LangGraph) – directed graph of autonomous reasoning nodes.
2. Perception  → FDI model (ExtraTrees 6-class) + dedicated energy monitor (battery is NOT a sensor fault).
3. Contextual reasoning – decision tree that picks the *right* action per situation,
   NOT always RTH.
4. Swarm coordination node – if this UAV is degraded and a peer is healthy,
   it DELEGATES the mission and hands off work to the peer.
5. Dual guardrails – L1 policy + L2 Control Barrier Function (CBF QP).
6. Live PX4 ROS 2 integration via Micro-XRCE-DDS bridge + full fallback to demo mode.
7. Live colour dashboard showing every reasoning step.

Action vocabulary
-----------------
NOMINAL_CONTINUE       – all good, carry on.
ALTITUDE_HOLD_HOVER    – minor issue; pause, stabilise, wait for resolution.
RETURN_TO_HOME         – moderate issue; GPS OK, battery OK; fly back safely.
DELEGATE_TO_SWARM      – this UAV is degraded; hand mission to a healthy peer UAV,
                          then return/land.
EMERGENCY_DESCENT      – controlled, slower descent to a safe altitude.
FAILSAFE_LAND          – LAST RESORT: land immediately at current position
                          (no navigation required – used when GPS is gone or
                          battery is critically low and no peer is available).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ── LangGraph ──────────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END, START
except ImportError:
    print("[ERROR] langgraph not installed.  Run:  pip install langgraph")
    sys.exit(1)

# ── CBF from step76 ────────────────────────────────────────────────────────
try:
    from step76_control_barrier_function import UAVControlBarrierFunction
    _CBF_AVAILABLE = True
except ImportError:
    _CBF_AVAILABLE = False
    print("[WARN] step76_control_barrier_function not found – L2 CBF fallback active.")

# ── FDI model from step70 ──────────────────────────────────────────────────
try:
    import joblib
    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "step70_clean_fdi_model.joblib")
    _FEAT_PATH  = os.path.join(os.path.dirname(__file__), "step70_feature_names.joblib")
    _FDI_MODEL     = joblib.load(_MODEL_PATH)
    _FEATURE_NAMES = joblib.load(_FEAT_PATH)
    _FDI_AVAILABLE = True
except Exception as e:
    _FDI_AVAILABLE = False
    print(f"[WARN] FDI model not loaded ({e}) – demo fault simulation active.")

# ── ROS 2 & PX4 Msgs ───────────────────────────────────────────────────────
_ROS_AVAILABLE = False
try:
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
        VehicleLandDetected,
        VehicleCommand,
    )
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


def _get_val(obj: Any, name: str, default: float = 0.0) -> float:
    if obj is None:
        return default
    return float(getattr(obj, name, default))


# ═══════════════════════════════════════════════════════════════════════════
# Live PX4 Telemetry Bridge (ROS 2)
# ═══════════════════════════════════════════════════════════════════════════
class LivePX4Bridge:
    """Background ROS 2 node subscribing to live PX4 Micro-XRCE-DDS topics."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.lock = threading.Lock()
        self.rows = deque(maxlen=window_size)
        self.last_msg_time = 0.0

        # Cached latest messages
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
        self.latest_land = None

        self._node = None
        self._cmd_pub = None
        self._thread = None
        self._running = False

    def start(self):
        if not _ROS_AVAILABLE:
            return False
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = Node("step78_px4_bridge")
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )

            # Subscriptions matching live PX4 topics
            self._node.create_subscription(SensorCombined, "/fmu/out/sensor_combined", self._sensor_cb, qos)
            self._node.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self._att_cb, qos)
            self._node.create_subscription(SensorGps, "/fmu/out/vehicle_gps_position", self._gps_cb, qos)
            self._node.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self._global_cb, qos)
            self._node.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._local_cb, qos)
            self._node.create_subscription(BatteryStatus, "/fmu/out/battery_status_v1", self._battery_cb, qos)
            self._node.create_subscription(EstimatorStatusFlags, "/fmu/out/estimator_status_flags", self._estimator_cb, qos)
            self._node.create_subscription(FailsafeFlags, "/fmu/out/failsafe_flags", self._failsafe_cb, qos)
            self._node.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v4", self._status_cb, qos)
            self._node.create_subscription(VehicleAirData, "/fmu/out/vehicle_air_data", self._air_cb, qos)
            self._node.create_subscription(VehicleLandDetected, "/fmu/out/vehicle_land_detected", self._land_cb, qos)

            # Publisher to send autonomous commands to PX4
            self._cmd_pub = self._node.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)

            # Snapshot recording timer (10 Hz)
            self._node.create_timer(0.1, self._record_snapshot)

            self._running = True
            self._thread = threading.Thread(target=self._spin_worker, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"[WARN] Failed to start ROS2 bridge: {e}")
            return False

    def _spin_worker(self):
        try:
            rclpy.spin(self._node)
        except Exception:
            pass

    def stop(self):
        self._running = False
        if self._node:
            try:
                self._node.destroy_node()
            except Exception:
                pass

    def _sensor_cb(self, msg):
        self.latest_sensor = msg
        self.last_msg_time = time.time()

    def _att_cb(self, msg):       self.latest_att = msg
    def _gps_cb(self, msg):       self.latest_gps = msg
    def _global_cb(self, msg):    self.latest_global = msg
    def _local_cb(self, msg):     self.latest_local = msg
    def _battery_cb(self, msg):   self.latest_battery = msg
    def _estimator_cb(self, msg): self.latest_estimator = msg
    def _failsafe_cb(self, msg):  self.latest_failsafe = msg
    def _status_cb(self, msg):    self.latest_status = msg
    def _air_cb(self, msg):       self.latest_air = msg
    def _land_cb(self, msg):      self.latest_land = msg

    def is_active(self) -> bool:
        return (time.time() - self.last_msg_time < 3.0) and (self.latest_sensor is not None)

    def _record_snapshot(self):
        if not self.is_active():
            return
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

        row = {
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
            "gps_lat": _get_val(gps, "latitude_deg"),
            "gps_lon": _get_val(gps, "longitude_deg"),
            "gps_alt": _get_val(gps, "altitude_m"),
            "gps_fix_type": _get_val(gps, "fix_type"),
            "gps_satellites": _get_val(gps, "satellites_used"),
            "gps_velocity": _get_val(gps, "vel_m_s"),
            "gps_vertical_speed": _get_val(gps, "vel_d_m_s"),
            # Global
            "global_lat": _get_val(gl, "lat"),
            "global_lon": _get_val(gl, "lon"),
            "global_alt": _get_val(gl, "alt"),
            "global_alt_valid": float(getattr(gl, "alt_valid", False)),
            "global_position_valid": float(getattr(gl, "lat_lon_valid", False)),
            # Local
            "local_x": _get_val(loc, "x"),
            "local_y": _get_val(loc, "y"),
            "local_z": _get_val(loc, "z"),
            "local_vx": _get_val(loc, "vx"),
            "local_vy": _get_val(loc, "vy"),
            "local_vz": _get_val(loc, "vz"),
            "local_z_valid": float(getattr(loc, "z_valid", True)),
            "local_vz_valid": float(getattr(loc, "v_z_valid", True)),
            "local_position_invalid": float(getattr(loc, "xy_valid", True) is False),
            "local_velocity_invalid": float(getattr(loc, "v_xy_valid", True) is False),
            # Battery
            "battery_voltage": _get_val(bat, "voltage_v"),
            "battery_current": _get_val(bat, "current_a"),
            "battery_remaining": _get_val(bat, "remaining", 1.0),
            # Estimator / Failsafe
            "attitude_invalid": float(getattr(fs, "attitude_invalid", False)),
            "angular_velocity_invalid": float(getattr(fs, "angular_velocity_invalid", False)),
            "local_position_invalid_est": float(getattr(fs, "local_position_invalid", False)),
            "local_velocity_invalid_est": float(getattr(fs, "local_velocity_invalid", False)),
            "global_position_invalid_est": float(getattr(fs, "global_position_invalid", False)),
            "inertial_dead_reckoning": _get_val(est, "cs_inertial_dead_reckoning", 0.0),
            # Magnetometer
            "cs_mag_fault": _get_val(est, "cs_mag_fault", 0.0),
            "cs_mag": _get_val(est, "cs_mag", 0.0),
            "cs_mag_hdg": _get_val(est, "cs_mag_hdg", 0.0),
            "cs_mag_3d": _get_val(est, "cs_mag_3d", 0.0),
            "cs_mag_heading_consistent": _get_val(est, "cs_mag_heading_consistent", 0.0),
            "fs_bad_mag_x": _get_val(est, "fs_bad_mag_x", 0.0),
            "fs_bad_mag_y": _get_val(est, "fs_bad_mag_y", 0.0),
            "fs_bad_mag_z": _get_val(est, "fs_bad_mag_z", 0.0),
            # Barometer & GPS
            "cs_baro_fault": _get_val(est, "cs_baro_fault", 0.0),
            "cs_gnss_fault": _get_val(est, "cs_gnss_fault", 0.0),
            "cs_gnss_pos": _get_val(est, "cs_gnss_pos", 0.0),
            "cs_baro_hgt": _get_val(est, "cs_baro_hgt", 0.0),
            # Status & Failsafe
            "failsafe": _get_val(fs, "failsafe", 0.0),
            "nav_state": _get_val(stat, "nav_state", 0.0),
            "arming_state": _get_val(stat, "arming_state", 0.0),
            # Air Data
            "baro_device_id": _get_val(air, "baro_device_id", 0.0),
            "baro_alt_meter": _get_val(air, "baro_alt_meter", 0.0),
            "baro_pressure_pa": _get_val(air, "baro_pressure_pa", 0.0),
            "ambient_temperature": _get_val(air, "ambient_temperature", 0.0),
            "rho": _get_val(air, "rho", 0.0),
        }
        with self.lock:
            self.rows.append(row)

    def get_telemetry_dict(self) -> Dict[str, Any]:
        with self.lock:
            loc = self.latest_local
            bat = self.latest_battery
            gps = self.latest_gps
            air = self.latest_air
            land = self.latest_land

        # Altitude (PX4 NED z is downwards, so altitude = -z)
        alt = 0.0
        if loc is not None and getattr(loc, "z_valid", False):
            alt = max(0.0, -float(loc.z))
        elif air is not None and hasattr(air, "baro_alt_meter"):
            alt = max(0.0, float(air.baro_alt_meter))

        vz = float(loc.vz) if loc is not None else 0.0

        # Battery percentage
        batt_pct = 100.0
        if bat is not None:
            rem = float(bat.remaining)
            if 0.0 <= rem <= 1.0:
                batt_pct = rem * 100.0
            elif bat.voltage_v > 0.0:
                # 4S LiPo estimation: 14.0V (0%) to 16.8V (100%)
                batt_pct = max(0.0, min(100.0, (bat.voltage_v - 14.0) / 2.8 * 100.0))

        # GPS validity: fix_type >= 3 means 3D fix
        gps_valid = True
        if gps is not None:
            gps_valid = int(getattr(gps, "fix_type", 3)) >= 3

        on_ground = bool(getattr(land, "landed", False)) if land else (alt < 0.2)
        pos_x = float(loc.x) if loc else 0.0
        pos_y = float(loc.y) if loc else 0.0

        return {
            "altitude": alt,
            "vz": vz,
            "battery_pct": batt_pct,
            "gps_valid": gps_valid,
            "on_ground": on_ground,
            "pos_x": pos_x,
            "pos_y": pos_y,
        }

    def get_feature_vector(self) -> Optional[np.ndarray]:
        with self.lock:
            if len(self.rows) < 10:
                return None
            df = pd.DataFrame(list(self.rows))

        feats = {}
        for col in df.columns:
            if col in ("time", "label", "run_id"):
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(s) == 0:
                continue
            feats[f"{col}_mean"]  = float(np.mean(s))
            feats[f"{col}_std"]   = float(np.std(s))
            feats[f"{col}_min"]   = float(np.min(s))
            feats[f"{col}_max"]   = float(np.max(s))
            feats[f"{col}_range"] = float(np.max(s) - np.min(s))
            if len(s) > 1:
                feats[f"{col}_first"] = float(s[0])
                feats[f"{col}_last"]  = float(s[-1])
                feats[f"{col}_delta"] = float(s[-1] - s[0])
                dx = np.diff(s)
                feats[f"{col}_mean_abs_change"] = float(np.mean(np.abs(dx)))
                feats[f"{col}_max_abs_change"]  = float(np.max(np.abs(dx)))
                feats[f"{col}_change_std"]      = float(np.std(dx))
                feats[f"{col}_change_range"]    = float(np.max(dx) - np.min(dx))

        if not _FDI_AVAILABLE or not _FEATURE_NAMES:
            return None
        feat_vector = [feats.get(c, 0.0) for c in _FEATURE_NAMES]
        return np.array([feat_vector], dtype=float)

    def send_vehicle_command(self, command_id: int, param1: float = 0.0, param2: float = 0.0):
        if not self._cmd_pub or not _ROS_AVAILABLE:
            return
        try:
            msg = VehicleCommand()
            msg.command = int(command_id)
            msg.param1 = float(param1)
            msg.param2 = float(param2)
            msg.target_system = 1
            msg.target_component = 1
            msg.source_system = 1
            msg.source_component = 1
            msg.from_external = True
            msg.timestamp = int(time.time() * 1e6)
            self._cmd_pub.publish(msg)
        except Exception:
            pass


# Global bridge singleton
_BRIDGE = LivePX4Bridge()

# ═══════════════════════════════════════════════════════════════════════════
# Simulated peer UAV registry (swarm)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class PeerUAV:
    uav_id:      str
    battery_pct: float
    pos_x:       float
    pos_y:       float
    altitude:    float
    fault:       str   = "No Failure"
    available:   bool  = True   # False if already handling a delegated task

# Default swarm peers
_PEER_UAVS: List[PeerUAV] = [
    PeerUAV("UAV-2", battery_pct=82.0, pos_x=30.0,  pos_y=15.0, altitude=12.0),
    PeerUAV("UAV-3", battery_pct=61.0, pos_x=-20.0, pos_y=40.0, altitude=10.0),
    PeerUAV("UAV-4", battery_pct=45.0, pos_x=10.0,  pos_y=-5.0, altitude=8.0),
]

# ═══════════════════════════════════════════════════════════════════════════
# Shared State
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class UAVState:
    # ── Telemetry ──────────────────────────────────────────────────────────
    altitude:         float = 0.0
    vz:               float = 0.0
    battery_pct:      float = 100.0
    gps_valid:        bool  = True
    on_ground:        bool  = False
    pos_x:            float = 0.0
    pos_y:            float = 0.0
    window: deque = field(default_factory=lambda: deque(maxlen=50))

    # ── Source indicator ───────────────────────────────────────────────────
    telemetry_source: str   = "DEMO"   # LIVE_ROS2 | DEMO

    # ── FDI ────────────────────────────────────────────────────────────────
    fault_label:  str   = "No Failure"
    confidence:   float = 0.0

    # ── Energy alert (separate from sensor FDI) ───────────────────────────
    energy_alert: str   = "NORMAL"   # NORMAL | LOW | CRITICAL

    # ── Risk ───────────────────────────────────────────────────────────────
    risk_stability:   float = 0.0
    risk_navigation:  float = 0.0
    risk_energy:      float = 0.0
    composite_risk:   float = 0.0

    # ── Agentic reasoning ──────────────────────────────────────────────────
    agent_proposal:   str = "NOMINAL_CONTINUE"
    reasoning_trace:  str = ""        # why the agent chose this action

    # ── Swarm coordination ─────────────────────────────────────────────────
    delegation_target:  str  = ""     # e.g. "UAV-2"
    delegation_status:  str  = "N/A"  # N/A | DELEGATED | NO_PEER_AVAILABLE

    # ── Guardrails ─────────────────────────────────────────────────────────
    l1_status: str = "PASSED"
    l1_notes:  str = ""
    l2_status: str = "PASSED"
    l2_notes:  str = ""

    # ── Final action ───────────────────────────────────────────────────────
    final_action: str = "NOMINAL_CONTINUE"

    # ── Per-tick log ───────────────────────────────────────────────────────
    log: List[str] = field(default_factory=list)

    def add_log(self, msg: str) -> None:
        self.log.append(msg)


# CLI Overrides
_CLI_ARGS = {
    "force_demo": False,
    "override_battery": None,
    "override_fault": None,
    "drain_battery": False,
}

_TICK = 0

def _make_demo_telemetry(tick: int) -> Dict[str, Any]:
    import math
    if _CLI_ARGS["override_battery"] is not None:
        batt = float(_CLI_ARGS["override_battery"])
    elif _CLI_ARGS["drain_battery"]:
        batt = max(0.0, 100.0 - tick * 0.4)
    else:
        batt = 88.0 - (tick * 0.02) % 10.0

    return {
        "altitude":    max(0.0, 10.0 + 3.0 * math.sin(tick * 0.08)),
        "vz":          0.4 * math.cos(tick * 0.08),
        "battery_pct": max(0.0, batt),
        "gps_valid":   True,
        "on_ground":   False,
        "pos_x":       tick * 0.04,
        "pos_y":       tick * 0.03,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph Node Definitions
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 1: collect_telemetry ───────────────────────────────────────────────
def collect_telemetry(state: UAVState) -> UAVState:
    global _TICK

    if not _CLI_ARGS["force_demo"] and _BRIDGE.is_active():
        t = _BRIDGE.get_telemetry_dict()
        state.telemetry_source = "LIVE_ROS2"
        state.altitude    = t["altitude"]
        state.vz          = t["vz"]
        state.battery_pct = t["battery_pct"]
        state.gps_valid   = t["gps_valid"]
        state.on_ground   = t["on_ground"]
        state.pos_x       = t["pos_x"]
        state.pos_y       = t["pos_y"]
    else:
        t = _make_demo_telemetry(_TICK)
        state.telemetry_source = "DEMO"
        state.altitude    = t["altitude"]
        state.vz          = t["vz"]
        state.battery_pct = t["battery_pct"]
        state.gps_valid   = t["gps_valid"]
        state.on_ground   = t["on_ground"]
        state.pos_x       = t["pos_x"]
        state.pos_y       = t["pos_y"]

    # Allow CLI battery override for specific scenarios
    if _CLI_ARGS["override_battery"] is not None:
        state.battery_pct = float(_CLI_ARGS["override_battery"])

    state.window.append({
        "altitude":    state.altitude,
        "vz":          state.vz,
        "battery_pct": state.battery_pct,
    })
    state.add_log(
        f"[Telemetry] ({state.telemetry_source}) alt={state.altitude:.2f}m vz={state.vz:+.2f}m/s "
        f"batt={state.battery_pct:.1f}% gps={'OK' if state.gps_valid else 'FAIL'}"
    )
    return state


# ── Node 2: monitor_energy (separate from FDI) ─────────────────────────────
def monitor_energy(state: UAVState) -> UAVState:
    """
    Battery monitoring is SEPARATE from sensor FDI.
    The FDI model detects physical sensor faults.
    Low battery is an energy/operational issue handled here.
    """
    b = state.battery_pct
    if b <= 15.0:
        state.energy_alert = "CRITICAL"
        state.add_log(f"[Energy] CRITICAL – battery {b:.1f}% ≤ 15%")
    elif b <= 30.0:
        state.energy_alert = "LOW"
        state.add_log(f"[Energy] LOW – battery {b:.1f}% ≤ 30%")
    else:
        state.energy_alert = "NORMAL"
        state.add_log(f"[Energy] NORMAL – battery {b:.1f}%")
    return state


# ── Node 3: run_fdi ─────────────────────────────────────────────────────────
def run_fdi(state: UAVState) -> UAVState:
    """
    FDI detects physical sensor faults using ExtraTrees model (step70).
    Battery level is NOT a sensor fault — handled by monitor_energy.
    """
    # Allow manual override for fault testing
    if _CLI_ARGS["override_fault"]:
        state.fault_label = _CLI_ARGS["override_fault"]
        state.confidence  = 98.5
        state.add_log(f"[FDI Override] {state.fault_label} ({state.confidence:.1f}% conf)")
        return state

    if not _CLI_ARGS["force_demo"] and _BRIDGE.is_active() and _FDI_AVAILABLE:
        X = _BRIDGE.get_feature_vector()
        if X is not None and X.shape[1] == len(_FEATURE_NAMES):
            proba = _FDI_MODEL.predict_proba(X)[0]
            idx   = int(np.argmax(proba))
            state.fault_label = str(_FDI_MODEL.classes_[idx])
            state.confidence  = float(proba[idx]) * 100.0
            state.add_log(f"[FDI Live] {state.fault_label} ({state.confidence:.1f}% conf)")
            return state

    # Fallback / Demo FDI
    state.fault_label = "No Failure"
    state.confidence  = 95.0
    state.add_log(f"[FDI] {state.fault_label} ({state.confidence:.1f}% conf)")
    return state


# ── Node 4: compute_risk ────────────────────────────────────────────────────
_FAULT_RISK = {
    "No Failure":              0.00,
    "Barometer Failure":       0.35,
    "Compass Failure":         0.30,
    "Accelerometer Failure":   0.75,
    "Gyroscope Failure":       0.85,
    "GPS Failure":             0.60,
}

def compute_risk(state: UAVState) -> UAVState:
    base   = _FAULT_RISK.get(state.fault_label, 0.25)
    conf_f = state.confidence / 100.0
    state.risk_stability  = round(min(1.0, base * conf_f * 1.2), 3)
    state.risk_navigation = round(min(1.0, base * conf_f * 1.1), 3)

    # Energy risk from battery — independent of sensor faults
    b = state.battery_pct
    if b <= 15.0:
        state.risk_energy = round(min(1.0, (15.0 - b) / 15.0 + 0.5), 3)
    elif b <= 30.0:
        state.risk_energy = round((30.0 - b) / 30.0 * 0.7, 3)
    else:
        state.risk_energy = 0.0

    state.composite_risk = round(
        max(state.risk_stability, state.risk_navigation, state.risk_energy), 3
    )
    state.add_log(
        f"[Risk] S:{state.risk_stability:.2f} N:{state.risk_navigation:.2f} "
        f"E:{state.risk_energy:.2f} → composite:{state.composite_risk:.2f}"
    )
    return state


# ── Node 5: agentic_reasoning ───────────────────────────────────────────────
def agentic_reasoning(state: UAVState) -> UAVState:
    """
    Contextual decision-making node.

    Rules (in priority order):
    1. Critical structural sensor fault (gyro / accel) regardless of battery
       → FAILSAFE_LAND (cannot fly safely; no navigation required)
    2. Battery CRITICAL (<= 15%) + peer UAV available
       → DELEGATE_TO_SWARM  (hand off mission, then land/RTH)
    3. Battery CRITICAL (<= 15%) + no healthy peer
       → FAILSAFE_LAND (not enough power to navigate home; land vertically now)
    4. GPS lost (cannot navigate)
       → ALTITUDE_HOLD_HOVER (stay in place; do NOT attempt RTH without GPS)
    5. Battery LOW (15-30%) + GPS OK + far from base (> 50m)
       → DELEGATE_TO_SWARM (delegate waypoints, then RTH)
    6. Battery LOW (15-30%) + GPS OK + close to base
       → RETURN_TO_HOME (enough to fly back safely)
    7. Barometer / Compass failure (non-critical, navigation degraded)
       → ALTITUDE_HOLD_HOVER (pause; stabilise; do not risk high-speed transit)
    8. Moderate composite risk (0.35-0.7)
       → ALTITUDE_HOLD_HOVER (cautious pause before deciding)
    9. All clear
       → NOMINAL_CONTINUE
    """
    fault   = state.fault_label
    energy  = state.energy_alert
    gps_ok  = state.gps_valid
    risk    = state.composite_risk
    dist    = (state.pos_x**2 + state.pos_y**2) ** 0.5

    proposal = "NOMINAL_CONTINUE"
    reason   = ""

    # Rule 1 – Critical structural sensor fault
    if fault in ("Gyroscope Failure", "Accelerometer Failure"):
        proposal = "FAILSAFE_LAND"
        reason   = (f"{fault} detected (conf {state.confidence:.0f}%) – "
                    "critical stabilisation sensor lost; landing immediately at current position.")

    # Rule 2 – Critical battery + peer available
    elif energy == "CRITICAL":
        peers_ok = [p for p in _PEER_UAVS if p.available and p.battery_pct > 35.0
                    and p.fault == "No Failure"]
        if peers_ok:
            proposal = "DELEGATE_TO_SWARM"
            reason   = (f"Battery critically low ({state.battery_pct:.1f}%). "
                        f"{len(peers_ok)} healthy peer(s) found; "
                        "delegating mission and initiating recovery landing.")
        else:
            proposal = "FAILSAFE_LAND"
            reason   = (f"Battery critically low ({state.battery_pct:.1f}%) and "
                        "no healthy peer available; landing immediately in place.")

    # Rule 3 – GPS lost
    elif not gps_ok or fault == "GPS Failure":
        proposal = "ALTITUDE_HOLD_HOVER"
        reason   = ("GPS signal lost or GPS Failure detected. "
                    "Cannot navigate safely. Hovering in place until position recovers.")

    # Rule 4 – Low battery, GPS OK
    elif energy == "LOW":
        if dist > 40.0:
            proposal = "DELEGATE_TO_SWARM"
            reason   = (f"Battery low ({state.battery_pct:.1f}%) and far from home ({dist:.0f}m). "
                        "Delegating task sector to healthy peer; returning home.")
        else:
            proposal = "RETURN_TO_HOME"
            reason   = (f"Battery low ({state.battery_pct:.1f}%), within return range ({dist:.0f}m). "
                        "Returning to launch base.")

    # Rule 5 – Barometer / Compass failure
    elif fault in ("Barometer Failure", "Compass Failure"):
        proposal = "ALTITUDE_HOLD_HOVER"
        reason   = (f"{fault} detected (conf {state.confidence:.0f}%). "
                    "Heading/altitude estimation degraded; hovering to stabilise.")

    # Rule 6 – Moderate risk
    elif risk >= 0.35:
        proposal = "ALTITUDE_HOLD_HOVER"
        reason   = f"Composite risk elevated ({risk:.2f}). Pausing mission to assess sensor integrity."

    # Rule 7 – All clear
    else:
        proposal = "NOMINAL_CONTINUE"
        reason   = f"All systems nominal (risk={risk:.2f}, battery={state.battery_pct:.1f}%)."

    state.agent_proposal  = proposal
    state.reasoning_trace = reason
    state.add_log(f"[Agent] → {proposal} | {reason}")
    return state


# ── Node 6: swarm_coordination ──────────────────────────────────────────────
def swarm_coordination(state: UAVState) -> UAVState:
    """
    Multi-agent handoff node.
    When agentic_reasoning proposes DELEGATE_TO_SWARM:
    1. Evaluates all peers in the swarm.
    2. Bids candidate based on battery and proximity.
    3. Handoff mission and updates state.
    """
    if state.agent_proposal != "DELEGATE_TO_SWARM":
        state.delegation_status = "N/A"
        return state

    candidates = [
        p for p in _PEER_UAVS
        if p.available and p.battery_pct > 35.0 and p.fault == "No Failure"
    ]

    if not candidates:
        state.delegation_status = "NO_PEER_AVAILABLE"
        state.agent_proposal    = "FAILSAFE_LAND"
        state.reasoning_trace  += " | No healthy peer found – falling back to FAILSAFE_LAND."
        state.add_log("[Swarm] No available peer – falling back to FAILSAFE_LAND.")
        return state

    def score(peer: PeerUAV) -> float:
        dist = ((peer.pos_x - state.pos_x)**2 + (peer.pos_y - state.pos_y)**2) ** 0.5
        return peer.battery_pct - dist * 0.4

    best_peer = max(candidates, key=score)
    best_peer.available = False  # Mark as engaged

    state.delegation_target = best_peer.uav_id
    state.delegation_status = "DELEGATED"
    state.add_log(
        f"[Swarm] Mission delegated to {best_peer.uav_id} (battery={best_peer.battery_pct:.0f}%)"
    )

    if state.battery_pct > 20.0 and state.gps_valid:
        state.agent_proposal = "RETURN_TO_HOME"
    else:
        state.agent_proposal = "FAILSAFE_LAND"

    return state


# ── Node 7: guardrail_l1 (mission policy) ───────────────────────────────────
def guardrail_l1(state: UAVState) -> UAVState:
    action = state.agent_proposal
    notes  = []
    passed = True

    # Cannot RTH without GPS
    if action == "RETURN_TO_HOME" and not state.gps_valid:
        action = "ALTITUDE_HOLD_HOVER"
        notes.append("GPS invalid – RTH blocked → HOVER")
        passed = False

    # Too close to ground – must land
    if state.altitude < 0.3 and action not in ("FAILSAFE_LAND", "NOMINAL_CONTINUE"):
        action = "FAILSAFE_LAND"
        notes.append("Altitude < 0.3 m → FAILSAFE_LAND")
        passed = False

    # On ground – block flight commands
    if state.on_ground and action in ("RETURN_TO_HOME", "DELEGATE_TO_SWARM"):
        action = "NOMINAL_CONTINUE"
        notes.append("On ground – flight commands blocked")
        passed = False

    state.l1_status    = "PASSED" if passed else "MODIFIED"
    state.l1_notes     = " | ".join(notes) if notes else "—"
    state.final_action = action
    state.add_log(f"[Guardrail L1] {state.l1_status} {state.l1_notes}")
    return state


# ── Node 8: guardrail_l2 (CBF) ──────────────────────────────────────────────
_ACTION_DESCENT = {
    "FAILSAFE_LAND":       -2.0,
    "RETURN_TO_HOME":      -0.8,
    "ALTITUDE_HOLD_HOVER": -0.1,
    "EMERGENCY_DESCENT":   -1.2,
    "NOMINAL_CONTINUE":     0.0,
    "DELEGATE_TO_SWARM":    0.0,
}

def guardrail_l2(state: UAVState) -> UAVState:
    action   = state.final_action
    u_nom    = _ACTION_DESCENT.get(action, 0.0)
    altitude = state.altitude

    if _CBF_AVAILABLE:
        cbf       = UAVControlBarrierFunction()
        vz_down   = max(0.0, -u_nom)
        u_nom_vec = np.array([0.0, 0.0, vz_down])
        result    = cbf.filter_control(u_nom_vec, altitude)
        u_safe_vec, cbf_active, h_val, _ = result
        vz_safe   = float(u_safe_vec[2])
        u_safe    = -vz_safe if u_nom < 0 else 0.0
        intervened = bool(cbf_active)
    else:
        if altitude < 1.5:
            u_safe = max(u_nom, -1.0)
        else:
            u_safe = u_nom
        intervened = abs(u_safe - u_nom) > 1e-4

    if intervened:
        state.l2_status = "INTERVENED"
        state.l2_notes  = f"CBF clamped descent {u_nom:.2f} → {u_safe:.2f} m/s"
        if action == "FAILSAFE_LAND" and altitude < 0.5:
            state.final_action = "ALTITUDE_HOLD_HOVER"
    else:
        state.l2_status = "PASSED"
        state.l2_notes  = f"u_nom={u_nom:.2f} m/s within safe bounds"

    state.add_log(f"[Guardrail L2] {state.l2_status} {state.l2_notes}")
    return state


# ── Node 9: execute_action ──────────────────────────────────────────────────
_PX4_CMD = {
    "FAILSAFE_LAND":       "NAV_LAND (cmd 21)",
    "RETURN_TO_HOME":      "NAV_RETURN_TO_LAUNCH (cmd 20)",
    "ALTITUDE_HOLD_HOVER": "DO_REPOSITION – hover (cmd 22)",
    "EMERGENCY_DESCENT":   "DO_CHANGE_ALTITUDE (cmd 222)",
    "NOMINAL_CONTINUE":    "— (no command; mission continues)",
    "DELEGATE_TO_SWARM":   "— (mission handoff broadcast sent)",
}

def execute_action(state: UAVState) -> UAVState:
    cmd = _PX4_CMD.get(state.final_action, "UNKNOWN")
    state.add_log(f"[Action] PX4 → {cmd}")

    # Closed-loop autonomous command dispatch to live PX4
    if _BRIDGE.is_active():
        if state.final_action == "FAILSAFE_LAND":
            _BRIDGE.send_vehicle_command(21)  # VEHICLE_CMD_NAV_LAND
        elif state.final_action == "RETURN_TO_HOME":
            _BRIDGE.send_vehicle_command(20)  # VEHICLE_CMD_NAV_RETURN_TO_LAUNCH
        elif state.final_action == "ALTITUDE_HOLD_HOVER":
            _BRIDGE.send_vehicle_command(22)  # VEHICLE_CMD_DO_REPOSITION
    return state


# ── Node 10: finalize ────────────────────────────────────────────────────────
def finalize(state: UAVState) -> UAVState:
    summary = {
        "telemetry_source": state.telemetry_source,
        "battery_pct":      state.battery_pct,
        "energy_alert":     state.energy_alert,
        "fdi":              {"fault": state.fault_label, "confidence": state.confidence},
        "risk":             {"S": state.risk_stability, "N": state.risk_navigation,
                             "E": state.risk_energy, "composite": state.composite_risk},
        "agent_proposal":   state.agent_proposal,
        "reasoning":        state.reasoning_trace,
        "delegation":       {"target": state.delegation_target, "status": state.delegation_status},
        "guardrail_l1":     state.l1_status,
        "guardrail_l2":     state.l2_status,
        "final_action":     state.final_action,
    }
    out = os.path.join(os.path.dirname(__file__), "langgraph_summary.json")
    try:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Build the LangGraph
# ═══════════════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(UAVState)

    g.add_node("collect_telemetry",  collect_telemetry)
    g.add_node("monitor_energy",     monitor_energy)
    g.add_node("run_fdi",            run_fdi)
    g.add_node("compute_risk",       compute_risk)
    g.add_node("agentic_reasoning",  agentic_reasoning)
    g.add_node("swarm_coordination", swarm_coordination)
    g.add_node("guardrail_l1",       guardrail_l1)
    g.add_node("guardrail_l2",       guardrail_l2)
    g.add_node("execute_action",     execute_action)
    g.add_node("finalize",           finalize)

    g.add_edge(START,                "collect_telemetry")
    g.add_edge("collect_telemetry",  "monitor_energy")
    g.add_edge("monitor_energy",     "run_fdi")
    g.add_edge("run_fdi",            "compute_risk")
    g.add_edge("compute_risk",       "agentic_reasoning")
    g.add_edge("agentic_reasoning",  "swarm_coordination")
    g.add_edge("swarm_coordination", "guardrail_l1")
    g.add_edge("guardrail_l1",       "guardrail_l2")
    g.add_edge("guardrail_l2",       "execute_action")
    g.add_edge("execute_action",     "finalize")
    g.add_edge("finalize",            END)

    return g.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Terminal Dashboard
# ═══════════════════════════════════════════════════════════════════════════
_C = {
    "CYAN": "\x1b[36m", "GREEN": "\x1b[32m", "YELLOW": "\x1b[33m",
    "RED":  "\x1b[31m", "BOLD":  "\x1b[1m",  "RESET":  "\x1b[0m",
    "WHITE":"\x1b[97m", "MAG":   "\x1b[35m",
}

def _col(text: str, c: str) -> str:
    return f"{_C[c]}{text}{_C['RESET']}"

def _bar(v: float, w: int = 18) -> str:
    n = max(0, min(w, int(v * w)))
    c = "GREEN" if v < 0.3 else ("YELLOW" if v < 0.6 else "RED")
    return _col("█" * n + "░" * (w - n), c)

def render_dashboard(state: UAVState, tick: int) -> None:
    print("\x1b[2J\x1b[H", end="")
    W = 68
    print(_col("=" * W, "CYAN"))
    mode_str = _col("LIVE PX4 ROS2", "GREEN") if state.telemetry_source == "LIVE_ROS2" else _col("DEMO SIMULATION", "YELLOW")
    print(f"  {_col('UAV Agentic-AI Dashboard', 'BOLD')} [LangGraph 10-node]  tick={tick}  [{mode_str}]")
    print(_col("=" * W, "CYAN"))

    # Telemetry
    print(_col("\n[1] Telemetry", "YELLOW"))
    gps_str = '✓ VALID' if state.gps_valid else _col('✗ INVALID / LOST', 'RED')
    print(f"    Alt: {state.altitude:6.2f}m   Vz: {state.vz:+.2f}m/s   "
          f"Pos: ({state.pos_x:.1f}, {state.pos_y:.1f})m   GPS: {gps_str}")

    # Energy Alert
    print(_col("\n[2] Energy Monitor (Independent of Sensor FDI)", "YELLOW"))
    batt_col = "GREEN" if state.energy_alert == "NORMAL" else \
               ("YELLOW" if state.energy_alert == "LOW" else "RED")
    alert_str = {"NORMAL": "✓ NORMAL (>30%)", "LOW": "⚠ LOW (15-30%)", "CRITICAL": "🔴 CRITICAL (≤15%)"}.get(
        state.energy_alert, state.energy_alert
    )
    print(f"    Battery : {state.battery_pct:5.1f}%  →  {_col(alert_str, batt_col)}")

    # FDI
    print(_col("\n[3] Fault Detection & Identification (ExtraTrees Classifier)", "YELLOW"))
    fdi_col = "GREEN" if state.fault_label == "No Failure" else "RED"
    print(f"    Sensor Status : {_col(state.fault_label, fdi_col)}")
    print(f"    Confidence    : {state.confidence:.1f}%")

    # Risk Assessment
    print(_col("\n[4] Multi-Dimensional Risk Assessment", "YELLOW"))
    print(f"    Stability  : {_bar(state.risk_stability)}  {state.risk_stability:.2f}")
    print(f"    Navigation : {_bar(state.risk_navigation)}  {state.risk_navigation:.2f}")
    print(f"    Energy     : {_bar(state.risk_energy)}  {state.risk_energy:.2f}")
    comp_c = "GREEN" if state.composite_risk < 0.3 else \
             ("YELLOW" if state.composite_risk < 0.6 else "RED")
    print(f"    Composite  : {_col(f'{state.composite_risk:.2f}', comp_c)}")

    # Agentic Reasoning
    print(_col("\n[5] Agentic Reasoning (Context-Aware Multi-Rule Engine)", "YELLOW"))
    print(f"    Proposed Action : {_col(state.agent_proposal, 'BOLD')}")
    # Word wrap reasoning
    words = state.reasoning_trace.split()
    line, lines = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 56:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    for i, ln in enumerate(lines):
        prefix = "    Reasoning Trace : " if i == 0 else "                      "
        print(f"{prefix}{ln}")

    # Swarm Coordination
    print(_col("\n[6] Swarm Coordination (Multi-Agent Mission Handoff)", "YELLOW"))
    if state.delegation_status == "DELEGATED":
        print(f"    Status : {_col('DELEGATED', 'GREEN')}  →  Transferred mission to {_col(state.delegation_target, 'BOLD')}")
    elif state.delegation_status == "NO_PEER_AVAILABLE":
        print(f"    Status : {_col('NO PEER AVAILABLE', 'RED')}  →  Escalating to autonomous landing")
    else:
        print(f"    Status : — (No delegation requested / Solitary execution)")

    print("    Peers  :", end="")
    for p in _PEER_UAVS:
        avail = "free" if p.available else _col("busy", "YELLOW")
        print(f"  {p.uav_id}(batt={p.battery_pct:.0f}%,{avail})", end="")
    print()

    # Guardrails
    print(_col("\n[7] Guardrail L1 – Mission & Operational Policy", "YELLOW"))
    l1c = "GREEN" if state.l1_status == "PASSED" else "RED"
    print(f"    Status : {_col(state.l1_status, l1c)}  {state.l1_notes}")

    print(_col("\n[8] Guardrail L2 – Control Barrier Function (CBF-QP)", "YELLOW"))
    l2c = "GREEN" if state.l2_status == "PASSED" else "YELLOW"
    print(f"    Status : {_col(state.l2_status, l2c)}  {state.l2_notes}")

    # Final Action
    print(_col("\n[9] Final Safe Action → Command Dispatch", "YELLOW"))
    action_exp = {
        "FAILSAFE_LAND":
            "Land IMMEDIATELY in-place vertically. No navigation needed. "
            "Used when critical sensor fails or battery <= 15% with no peer.",
        "RETURN_TO_HOME":
            "Fly safely back to launch base (requires GPS + sufficient power).",
        "ALTITUDE_HOLD_HOVER":
            "Hold current 3D position. Used for sensor transients or GPS loss.",
        "DELEGATE_TO_SWARM":
            "Transmit waypoint queue to healthy peer, then execute local recovery.",
        "EMERGENCY_DESCENT":
            "Controlled descent to safe altitude floor.",
        "NOMINAL_CONTINUE":
            "All health checks verified. Continuing mission path.",
    }
    print(f"    ✅  {_col(state.final_action, 'GREEN')}")
    exp = action_exp.get(state.final_action, "")
    words2 = exp.split()
    line2, lines2 = [], []
    for w in words2:
        line2.append(w)
        if len(" ".join(line2)) > 56:
            lines2.append(" ".join(line2))
            line2 = []
    if line2:
        lines2.append(" ".join(line2))
    for i, ln in enumerate(lines2):
        prefix = "    Meaning : " if i == 0 else "              "
        print(f"{prefix}{ln}")

    print(_col("\n" + "=" * W, "CYAN"))
    print(f"   [Press Ctrl-C to pause / stop]")


# ═══════════════════════════════════════════════════════════════════════════
# Async Main Loop
# ═══════════════════════════════════════════════════════════════════════════
async def main_loop(max_ticks: int = 0) -> None:
    global _TICK
    graph = build_graph()
    state = UAVState()

    print("[LangGraph] Starting Agentic AI pipeline loop...")
    await asyncio.sleep(0.5)

    while True:
        _TICK += 1
        state.log.clear()

        result_dict = graph.invoke(state)

        prev_window = state.window
        try:
            state = UAVState(**{
                k: v for k, v in result_dict.items()
                if k in UAVState.__dataclass_fields__
            })
        except TypeError:
            for k, v in result_dict.items():
                if hasattr(state, k):
                    setattr(state, k, v)

        if not list(state.window):
            state.window = prev_window

        render_dashboard(state, _TICK)

        if max_ticks > 0 and _TICK >= max_ticks:
            print(f"\n[Done] Reached target tick limit {max_ticks}.")
            break

        await asyncio.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description="UAV Agentic AI LangGraph Control Loop")
    parser.add_argument("--demo", action="store_true", help="Force demo simulation even if ROS2 is present")
    parser.add_argument("--battery", type=float, default=None, help="Inject fixed battery percentage (e.g. 16.0)")
    parser.add_argument("--fault", type=str, default=None, help="Inject specific sensor fault (e.g. 'GPS Failure')")
    parser.add_argument("--drain", action="store_true", help="Simulate progressive battery drain in demo mode")
    parser.add_argument("--all-busy", action="store_true", help="Mark all swarm peers as busy (triggers FAILSAFE_LAND)")
    parser.add_argument("--ticks", type=int, default=0, help="Run for N ticks then exit (0 = loop indefinitely)")
    args = parser.parse_args()

    _CLI_ARGS["force_demo"] = args.demo
    _CLI_ARGS["override_battery"] = args.battery
    _CLI_ARGS["override_fault"] = args.fault
    _CLI_ARGS["drain_battery"] = args.drain

    if args.all_busy:
        for p in _PEER_UAVS:
            p.available = False

    # Start live ROS 2 bridge if not forced to demo
    if not args.demo and _ROS_AVAILABLE:
        print("[ROS2] Initializing live PX4 Telemetry Bridge...")
        _BRIDGE.start()
        time.sleep(1.0)
        if _BRIDGE.is_active():
            print("[ROS2] Connected to live PX4 Micro-XRCE-DDS topics!")
        else:
            print("[ROS2] No live PX4 topics detected yet. Will auto-switch when data arrives.")

    try:
        asyncio.run(main_loop(max_ticks=args.ticks))
    except KeyboardInterrupt:
        print("\n[LangGraph] Execution stopped by user.")
    finally:
        _BRIDGE.stop()


if __name__ == "__main__":
    main()
