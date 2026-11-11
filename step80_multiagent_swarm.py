"""step80_multiagent_swarm.py  – Multi-Agent UAV Swarm with Autonomous LangGraph Agents

Real Multi-Agent Agentic AI Architecture
=========================================
1. Swarm of 4 Autonomous UAV Agents (UAV-1 Alpha, UAV-2 Bravo, UAV-3 Charlie, UAV-4 Delta).
2. EACH UAV executes an independent 10-node LangGraph state graph every tick:
     [Telemetry] → [Energy Monitor] → [FDI Inference] → [Risk Assessment] →
     [Agentic Reasoning] → [Swarm Coordination] → [Guardrail L1] →
     [Guardrail L2 CBF-QP] → [Execute Action] → [Finalize]
3. SwarmBus Coordination Protocol:
     - Dynamic Heartbeat exchange.
     - Task delegation & handoff protocol: Degraded UAV broadcasts HandoffRequest;
       peers evaluate battery, health, and distance; winning peer accepts the mission.
     - Inter-agent collision avoidance via Control Barrier Functions (CBF).
4. ROS 2 Integration:
     - Publishes live UAVStateReport msgs to /uav_swarm/uav_{id}/state_report.
     - If live PX4 SITL is running, UAV-1 can bind to live PX4 Micro-XRCE-DDS topics.
5. Live Multi-Agent ANSI Dashboard:
     - Side-by-side 4-agent status cards + swarm overview + inter-agent communication log.
6. Dynamic Fault Injection:
     - CLI flags to inject battery drop, sensor failure, or swarm degradation at any tick.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ── LangGraph ──────────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END, START
except ImportError:
    print("[ERROR] langgraph not installed. Run: pip install langgraph")
    sys.exit(1)

# ── Control Barrier Function (step76) ──────────────────────────────────────
try:
    from step76_control_barrier_function import UAVControlBarrierFunction
    _CBF_AVAILABLE = True
except ImportError:
    _CBF_AVAILABLE = False

# ── Sensor FDI Model (step70) ──────────────────────────────────────────────
try:
    import joblib
    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "step70_clean_fdi_model.joblib")
    _FEAT_PATH  = os.path.join(os.path.dirname(__file__), "step70_feature_names.joblib")
    _FDI_MODEL     = joblib.load(_MODEL_PATH)
    _FEATURE_NAMES = joblib.load(_FEAT_PATH)
    _FDI_AVAILABLE = True
except Exception:
    _FDI_AVAILABLE = False

# ── ROS 2 Support ──────────────────────────────────────────────────────────
_ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from uav_swarm_msgs.msg import UAVStateReport
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False

# ── MAVLink Multi-Vehicle Support (for QGroundControl live map) ─────────────
_MAVLINK_AVAILABLE = False
try:
    from pymavlink import mavutil
    _MAVLINK_AVAILABLE = True
except ImportError:
    _MAVLINK_AVAILABLE = False


def is_px4_sitl_running() -> bool:
    """Detect whether PX4 SITL (x500 instance 0) binary is running."""
    try:
        import subprocess
        # Must match exact executable name 'px4', not substring matches like 'python3 ...'
        out = subprocess.check_output(["pgrep", "-x", "px4"], stderr=subprocess.DEVNULL)
        return len(out.strip()) > 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Swarm Communication Bus (Inter-Agent Messaging)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SwarmMessage:
    msg_type:  str       # HEARTBEAT | HANDOFF_REQUEST | HANDOFF_BID | HANDOFF_ACCEPT
    sender_id: str
    data:      Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

# Global No-Fly Zones (NFZ): Center (x, y), radius r, and clearance alt
NFZ_ZONES = [
    {"name": "NFZ-Alpha (Hospital)", "x": 50.0, "y": 60.0, "radius": 18.0, "alt_ceil": 25.0},
    {"name": "NFZ-Bravo (Airport)", "x": -15.0, "y": 90.0, "radius": 22.0, "alt_ceil": 30.0},
]

class SwarmBus:
    """Decentralized message broker for dynamic peer-to-peer UAV swarm communications."""

    def __init__(self):
        self.lock = threading.RLock()
        self.peer_states: Dict[str, Dict[str, Any]] = {}
        self.event_log: deque = deque(maxlen=10)
        self.active_faults: Dict[str, str] = {}  # Injected via QGC/CLI: { "UAV-1": "GPS Failure" }
        self._ros_node = None
        self._publishers: Dict[str, Any] = {}
        self._mav_conns: Dict[str, Any] = {}
        self._px4_onboard_conn = None
        self.latest_px4_telemetry: Dict[str, Any] = {}
        self._last_px4_msg_time: float = 0.0
        self._qgc_listener_sock = None
        self.nfz_zones = list(NFZ_ZONES)
        # Task-completion events queued during a tick for immediate master reporting
        self._pending_task_completions: List[Dict[str, Any]] = []
        # Routes from degraded UAVs waiting to be picked up by assigned peers
        self._pending_handoff_routes: Dict[str, Dict[str, Any]] = {}

    def notify_task_completed(self, uav_id: str, wp_idx: int, tasks_done: int, total_tasks: int):
        """Called by node_collect_telemetry when a UAV finishes a waypoint.
        Queues an event so the main loop emits an immediate master report."""
        with self.lock:
            self._pending_task_completions.append({
                "uav_id": uav_id,
                "wp_idx": wp_idx,
                "tasks_done": tasks_done,
                "total_tasks": total_tasks,
            })

    def register_uav(self, uav_id: str):
        """Dynamically registers a new UAV agent for horizontal scaling."""
        with self.lock:
            if uav_id not in self.peer_states:
                self.peer_states[uav_id] = {
                    "pos": [0.0, 0.0, 10.0],
                    "vel": [0.0, 0.0, 0.0],
                    "battery_pct": 90.0,
                    "fault": "No Failure",
                    "available": True,
                    "task": "Dynamic Enroute",
                    "action": "NOMINAL_CONTINUE",
                }
                # Initialize MAVLink stream for QGC if available
                if _MAVLINK_AVAILABLE and uav_id not in self._mav_conns:
                    try:
                        sys_id = int(uav_id.split("-")[-1])
                        mav = mavutil.mavlink_connection("udpout:127.0.0.1:14550", source_system=sys_id, source_component=1)
                        self._mav_conns[uav_id] = mav
                        self._send_params(mav, sys_id)
                    except Exception:
                        pass
                # Initialize ROS2 publisher if ROS2 is active
                if self._ros_node and uav_id not in self._publishers and _ROS_AVAILABLE:
                    try:
                        topic = f"/uav_swarm/{uav_id.lower().replace('-', '_')}/state_report"
                        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
                        self._publishers[uav_id] = self._ros_node.create_publisher(UAVStateReport, topic, qos)
                    except Exception:
                        pass
                self.log_event(f"➕ [HORIZONTAL SCALING] {uav_id} registered into active swarm mesh")

    def init_ros(self):
        if not _ROS_AVAILABLE:
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._ros_node = Node("step80_swarm_bus")
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
            for uid in list(self.peer_states.keys()) or ("UAV-1", "UAV-2", "UAV-3", "UAV-4"):
                topic = f"/uav_swarm/{uid.lower().replace('-', '_')}/state_report"
                self._publishers[uid] = self._ros_node.create_publisher(UAVStateReport, topic, qos)
        except Exception as e:
            self.log_event(f"ROS2 publisher init error: {e}")

    def _send_params(self, mav, sys_id: int):
        """Send required MAVLink parameter responses so QGC does not timeout."""
        params = [
            (b"SYS_AUTOSTART\x00\x00\x00", 4001.0),
            (b"MAV_TYPE\x00\x00\x00\x00\x00\x00\x00\x00", 2.0),
            (b"COM_RC_IN_MODE\x00\x00", 1.0),
            (b"NAV_RCL_ACT\x00\x00\x00\x00\x00", 2.0),
            (b"SYS_ID\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", float(sys_id)),
        ]
        for idx, (p_id, val) in enumerate(params):
            try:
                mav.mav.param_value_send(
                    p_id,
                    float(val),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
                    len(params),
                    idx
                )
            except Exception:
                pass

    def init_mavlink(self, target_ip: str = "127.0.0.1", target_port: int = 14550):
        """Initializes MAVLink UDP broadcast streams for QGroundControl."""
        if not _MAVLINK_AVAILABLE:
            return
        try:
            sitl_active = is_px4_sitl_running()
            for i, uid in enumerate(("UAV-1", "UAV-2", "UAV-3", "UAV-4"), start=1):
                if sitl_active and uid == "UAV-1":
                    continue
                mav = mavutil.mavlink_connection(
                    f"udpout:{target_ip}:{target_port}",
                    source_system=i,
                    source_component=1
                )
                self._mav_conns[uid] = mav
                self._send_params(mav, i)
                for m_type in (
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                    mavutil.mavlink.MAV_MISSION_TYPE_FENCE,
                    mavutil.mavlink.MAV_MISSION_TYPE_RALLY,
                ):
                    try:
                        mav.mav.mission_count_send(255, 0, 0, m_type)
                    except Exception:
                        pass

            # If PX4 SITL is running, also connect to its onboard MAVLink port (14540)
            # so the SwarmBus can listen directly to PX4 STATUSTEXT and failure events
            if sitl_active:
                try:
                    self._px4_onboard_conn = mavutil.mavlink_connection("udpin:127.0.0.1:14540")
                    # Send a heartbeat back to PX4 so it knows a client is listening on 14540
                    self._px4_onboard_conn.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
                    self.log_event("🔗 Connected to live PX4 onboard MAVLink on port 14540")
                except Exception as e:
                    self.log_event(f"PX4 onboard bind warning: {e}")

            mode_str = "UAV-2..4 (UAV-1 via PX4 SITL)" if sitl_active else "UAV-1..4 (Standalone)"
            self.log_event(f"🎮 QGC MAVLink connected on UDP {target_port} for {mode_str}")
        except Exception as e:
            self.log_event(f"MAVLink init error: {e}")

    def is_connected_to_px4(self) -> bool:
        """Returns True if actively receiving telemetry from PX4 SITL + Gazebo."""
        return bool(self._px4_onboard_conn and (time.time() - self._last_px4_msg_time < 3.0))

    def poll_px4_telemetry(self) -> Dict[str, Any]:
        """Polls live PX4 onboard stream for failure injections and telemetry."""
        result: Dict[str, Any] = {}

        # Dynamically connect to PX4 onboard port 14540 if PX4 starts up while swarm is running
        if not self._px4_onboard_conn and is_px4_sitl_running() and _MAVLINK_AVAILABLE:
            try:
                self._px4_onboard_conn = mavutil.mavlink_connection("udpin:127.0.0.1:14540")
                self._px4_onboard_conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
                self.log_event("🔗 Connected to live PX4 SITL/Gazebo onboard MAVLink on port 14540")
            except Exception:
                pass

        if not self._px4_onboard_conn:
            return result

        try:
            # Send periodic heartbeat so PX4 keeps the stream active
            if not hasattr(self, "_last_onboard_hb") or time.time() - self._last_onboard_hb > 1.0:
                self._last_onboard_hb = time.time()
                try:
                    self._px4_onboard_conn.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
                except Exception:
                    pass

            while True:
                msg = self._px4_onboard_conn.recv_match(blocking=False)
                if not msg:
                    break
                self._last_px4_msg_time = time.time()
                mtype = msg.get_type()

                if mtype == "STATUSTEXT":
                    txt = getattr(msg, "text", "")
                    if isinstance(txt, bytes):
                        txt = txt.decode("utf-8", errors="ignore")
                    txt_lower = txt.lower()
                    if any(w in txt_lower for w in ("warn", "error", "fail", "inject")):
                        self.log_event(f"💬 PX4: {txt[:40]}")
                    if "gyro" in txt_lower or "unit: gyro" in txt_lower or "unit: (0)" in txt_lower:
                        fault = "Gyroscope Failure" if not any(ok in txt_lower for ok in ("ok", "recovered", "cleared")) else "No Failure"
                        with self.lock:
                            if fault == "No Failure":
                                self.active_faults.pop("UAV-1", None)
                            else:
                                self.active_faults["UAV-1"] = "Gyroscope Failure"
                        self.log_event(f"📡 PX4 reported: {fault} on UAV-1")
                    elif "accel" in txt_lower or "unit: accel" in txt_lower:
                        fault = "Accelerometer Failure" if not any(ok in txt_lower for ok in ("ok", "recovered", "cleared")) else "No Failure"
                        with self.lock:
                            if fault == "No Failure":
                                self.active_faults.pop("UAV-1", None)
                            else:
                                self.active_faults["UAV-1"] = "Accelerometer Failure"
                        self.log_event(f"📡 PX4 reported: {fault} on UAV-1")
                    elif "gps" in txt_lower or "navigation error" in txt_lower or "no valid position" in txt_lower or "blind land" in txt_lower:
                        fault = "GPS Failure" if not any(ok in txt_lower for ok in ("ok", "recovered", "cleared")) else "No Failure"
                        with self.lock:
                            if fault == "No Failure":
                                self.active_faults.pop("UAV-1", None)
                            else:
                                self.active_faults["UAV-1"] = "GPS Failure"
                        self.log_event(f"📡 PX4 reported: {fault} on UAV-1")
                    elif "mag" in txt_lower or "compass" in txt_lower:
                        fault = "Compass Failure" if not any(ok in txt_lower for ok in ("ok", "recovered", "cleared")) else "No Failure"
                        with self.lock:
                            if fault == "No Failure":
                                self.active_faults.pop("UAV-1", None)
                            else:
                                self.active_faults["UAV-1"] = "Compass Failure"
                        self.log_event(f"📡 PX4 reported: {fault} on UAV-1")
                    elif "baro" in txt_lower:
                        fault = "Barometer Failure" if not any(ok in txt_lower for ok in ("ok", "recovered", "cleared")) else "No Failure"
                        with self.lock:
                            if fault == "No Failure":
                                self.active_faults.pop("UAV-1", None)
                            else:
                                self.active_faults["UAV-1"] = "Barometer Failure"
                        self.log_event(f"📡 PX4 reported: {fault} on UAV-1")

                elif mtype == "SYS_STATUS":
                    if hasattr(msg, "battery_remaining") and msg.battery_remaining != -1:
                        batt = float(msg.battery_remaining)
                        result["battery_pct"] = batt
                        self.latest_px4_telemetry["battery_pct"] = batt
                    present = getattr(msg, "onboard_control_sensors_present", 0)
                    enabled = getattr(msg, "onboard_control_sensors_enabled", 0)
                    health = getattr(msg, "onboard_control_sensors_health", 0)

                    # When a sensor is failed/disabled in PX4, its health OR enabled bit is cleared
                    if present:
                        # GPS bit = 32 (1<<5)
                        if (present & 32) and (not (health & 32) or not (enabled & 32)):
                            with self.lock:
                                self.active_faults["UAV-1"] = "GPS Failure"
                            self.log_event("📡 PX4 SYS_STATUS: GPS Failure (disabled/unhealthy) on UAV-1")
                        elif (present & 32) and (health & 32) and (enabled & 32) and self.active_faults.get("UAV-1") == "GPS Failure":
                            with self.lock:
                                self.active_faults.pop("UAV-1", None)
                            self.log_event("📡 PX4 SYS_STATUS: GPS recovered to nominal on UAV-1")

                        # 3D Gyro bit = 1 (1<<0)
                        if (present & 1) and (not (health & 1) or not (enabled & 1)):
                            with self.lock:
                                self.active_faults["UAV-1"] = "Gyroscope Failure"
                            self.log_event("📡 PX4 SYS_STATUS: Gyroscope Failure on UAV-1")
                        elif (present & 1) and (health & 1) and (enabled & 1) and self.active_faults.get("UAV-1") == "Gyroscope Failure":
                            with self.lock:
                                self.active_faults.pop("UAV-1", None)

                        # 3D Accel bit = 2 (1<<1)
                        if (present & 2) and (not (health & 2) or not (enabled & 2)):
                            with self.lock:
                                self.active_faults["UAV-1"] = "Accelerometer Failure"
                            self.log_event("📡 PX4 SYS_STATUS: Accelerometer Failure on UAV-1")
                        elif (present & 2) and (health & 2) and (enabled & 2) and self.active_faults.get("UAV-1") == "Accelerometer Failure":
                            with self.lock:
                                self.active_faults.pop("UAV-1", None)

                        # 3D Mag bit = 4 (1<<2)
                        if (present & 4) and (not (health & 4) or not (enabled & 4)):
                            with self.lock:
                                self.active_faults["UAV-1"] = "Compass Failure"
                            self.log_event("📡 PX4 SYS_STATUS: Compass Failure on UAV-1")
                        elif (present & 4) and (health & 4) and (enabled & 4) and self.active_faults.get("UAV-1") == "Compass Failure":
                            with self.lock:
                                self.active_faults.pop("UAV-1", None)

                        # Barometer bit = 8 (1<<3)
                        if (present & 8) and (not (health & 8) or not (enabled & 8)):
                            with self.lock:
                                self.active_faults["UAV-1"] = "Barometer Failure"
                            self.log_event("📡 PX4 SYS_STATUS: Barometer Failure on UAV-1")
                        elif (present & 8) and (health & 8) and (enabled & 8) and self.active_faults.get("UAV-1") == "Barometer Failure":
                            with self.lock:
                                self.active_faults.pop("UAV-1", None)

                elif mtype == "GPS_RAW_INT":
                    fix_type = getattr(msg, "fix_type", 3)
                    satellites = getattr(msg, "satellites_visible", 10)
                    if fix_type < 2 or satellites < 4:
                        with self.lock:
                            if self.active_faults.get("UAV-1") != "GPS Failure":
                                self.active_faults["UAV-1"] = "GPS Failure"
                                self.log_event(f"📡 PX4 GPS_RAW_INT: GPS fix lost (fix_type={fix_type}, sats={satellites}) on UAV-1")
                    elif fix_type >= 3 and satellites >= 6 and self.active_faults.get("UAV-1") == "GPS Failure":
                        with self.lock:
                            self.active_faults.pop("UAV-1", None)
                        self.log_event("📡 PX4 GPS_RAW_INT: GPS 3D fix restored on UAV-1")

                elif mtype == "COMMAND_LONG" and getattr(msg, "command", 0) == 420:
                    unit = int(getattr(msg, "param1", -1))
                    ftype = int(getattr(msg, "param2", -1))
                    unit_map = {1: "Gyroscope Failure", 2: "Accelerometer Failure", 3: "Compass Failure", 4: "Barometer Failure", 5: "GPS Failure"}
                    with self.lock:
                        if ftype == 0:
                            self.active_faults.pop("UAV-1", None)
                            self.log_event("📡 PX4 VEHICLE_CMD_INJECT_FAILURE: Fault cleared on UAV-1")
                        else:
                            fname = unit_map.get(unit, "Sensor Fault")
                            self.active_faults["UAV-1"] = fname
                            self.log_event(f"📡 PX4 VEHICLE_CMD_INJECT_FAILURE: {fname} injected on UAV-1")

                elif mtype == "GLOBAL_POSITION_INT":
                    lat = float(msg.lat) / 1e7
                    lon = float(msg.lon) / 1e7
                    result["pos_z"] = max(0.0, float(msg.relative_alt) / 1000.0)
                    result["vx"] = float(msg.vx) / 100.0
                    result["vy"] = float(msg.vy) / 100.0
                    result["vz"] = float(msg.vz) / 100.0
                    result["pos_x"] = (lon - 8.545594) * 111320.0 * 0.6769
                    result["pos_y"] = (lat - 47.397742) * 111320.0
                    self.latest_px4_telemetry.update(result)

                elif mtype == "LOCAL_POSITION_NED":
                    result["pos_x"] = float(msg.x)
                    result["pos_y"] = float(msg.y)
                    result["pos_z"] = max(0.0, -float(msg.z))
                    result["vx"] = float(msg.vx)
                    result["vy"] = float(msg.vy)
                    result["vz"] = -float(msg.vz)
                    self.latest_px4_telemetry.update(result)
        except Exception:
            pass
        return result

    def log_event(self, text: str):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.event_log.append(f"[{ts}] {text}")

    def broadcast_heartbeat(self, uav_id: str, state_summary: Dict[str, Any]):
        with self.lock:
            self.peer_states[uav_id] = state_summary

        # Publish to ROS 2 topic if active
        if self._ros_node and uav_id in self._publishers:
            try:
                msg = UAVStateReport()
                msg.uav_id = int(uav_id.split("-")[-1])
                pos = state_summary.get("pos", [0.0, 0.0, 0.0])
                msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
                vel = state_summary.get("vel", [0.0, 0.0, 0.0])
                msg.velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
                msg.battery = float(state_summary.get("battery_pct", 100.0))
                msg.fault_status = str(state_summary.get("fault", "No Failure"))
                msg.current_task = str(state_summary.get("task", "NOMINAL"))
                msg.task_status = str(state_summary.get("action", "NOMINAL_CONTINUE"))
                self._publishers[uav_id].publish(msg)
            except Exception:
                pass

        # Publish to QGroundControl via MAVLink
        if uav_id in self._mav_conns:
            try:
                mav = self._mav_conns[uav_id]
                sys_id = int(uav_id.split("-")[-1])

                # Check and reply to QGC incoming requests (parameters, capabilities, mission/rally)
                try:
                    while True:
                        req = mav.recv_match(blocking=False)
                        if not req:
                            break
                        rtype = req.get_type()
                        if rtype in ("PARAM_REQUEST_LIST", "PARAM_REQUEST_READ"):
                            self._send_params(mav, sys_id)
                        elif rtype == "MISSION_REQUEST_LIST":
                            m_type = getattr(req, "mission_type", 0)
                            mav.mav.mission_count_send(
                                req.get_srcSystem() or 255,
                                req.get_srcComponent() or 0,
                                0,
                                m_type
                            )
                        elif rtype == "COMMAND_LONG":
                            mav.mav.command_ack_send(req.command, mavutil.mavlink.MAV_RESULT_ACCEPTED)
                            if req.command == 420:  # MAV_CMD_INJECT_FAILURE
                                unit = int(getattr(req, "param1", -1))
                                failure_type = int(getattr(req, "param2", -1))
                                fault_map = {
                                    4: "GPS Failure",
                                    0: "Gyroscope Failure",
                                    1: "Accelerometer Failure",
                                    2: "Compass Failure",
                                    3: "Barometer Failure",
                                }
                                if failure_type == 0:  # FAILURE_TYPE_OK (clear fault)
                                    with self.lock:
                                        self.active_faults.pop(uav_id, None)
                                    self.log_event(f"🔧 QGC cleared fault on {uav_id}")
                                    try:
                                        mav.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_INFO, f"🔧 [{uav_id}] Fault Cleared".encode()[:50])
                                    except Exception:
                                        pass
                                else:
                                    fname = fault_map.get(unit, "Sensor Fault")
                                    with self.lock:
                                        self.active_faults[uav_id] = fname
                                    self.log_event(f"🔴 QGC injected {fname} on {uav_id}")
                                    try:
                                        mav.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_WARNING, f"🔴 [{uav_id}] {fname} Injected".encode()[:50])
                                    except Exception:
                                        pass

                        elif rtype == "SERIAL_CONTROL":
                            # Handles QGroundControl MAVLink Console commands (e.g. 'failure gps off')
                            raw_data = bytes(req.data[:req.count])
                            cmd_str = raw_data.decode("utf-8", errors="ignore").strip()
                            cmd_lower = cmd_str.lower()

                            def _reply_console(txt: str):
                                full = f"\r\n[SWARM] {txt}\r\n"
                                b = full.encode("utf-8")
                                for k in range(0, len(b), 68):
                                    chunk = b[k:k+68]
                                    d_list = list(chunk) + [0] * (70 - len(chunk))
                                    try:
                                        mav.mav.serial_control_send(
                                            mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,
                                            mavutil.mavlink.SERIAL_CONTROL_FLAG_REPLY,
                                            0, 0,
                                            len(chunk),
                                            d_list
                                        )
                                    except Exception:
                                        pass

                            if not cmd_lower:
                                _reply_console(f"{uav_id} shell ready. Type 'failure help' for FDI commands.")
                            elif "help" in cmd_lower:
                                _reply_console("Commands:\r\n  failure gps off    - Inject GPS failure\r\n  failure gyro off   - Inject Gyro failure\r\n  failure accel off  - Inject Accel failure\r\n  failure ok         - Clear all faults\r\n  failure list       - List active faults")
                            elif "list" in cmd_lower or "status" in cmd_lower:
                                with self.lock:
                                    flist = ", ".join([f"{k}: {v}" for k, v in self.active_faults.items()]) or "All systems nominal (No active faults)"
                                _reply_console(f"Active Faults: {flist}")
                            elif "failure" in cmd_lower or "fault" in cmd_lower:
                                target_uid = uav_id
                                tokens = cmd_lower.split()
                                for tok in tokens:
                                    if tok.startswith("uav-") or (tok.startswith("uav") and len(tok) >= 4):
                                        target_uid = tok.upper()
                                        if not target_uid.startswith("UAV-"):
                                            target_uid = f"UAV-{target_uid[3:]}"

                                is_clear_cmd = any(w in cmd_lower for w in ("ok", "clear", "0", "recover", "reset", "none"))
                                fault_map = {
                                    "gps": "GPS Failure",
                                    "gyro": "Gyroscope Failure",
                                    "gyroscope": "Gyroscope Failure",
                                    "accel": "Accelerometer Failure",
                                    "accelerometer": "Accelerometer Failure",
                                    "baro": "Barometer Failure",
                                    "barometer": "Barometer Failure",
                                    "compass": "Compass Failure",
                                    "mag": "Compass Failure",
                                }
                                sensor_hit = None
                                for k_s, f_val in fault_map.items():
                                    if k_s in cmd_lower:
                                        sensor_hit = f_val
                                        break

                                if is_clear_cmd or "clear" in cmd_lower:
                                    with self.lock:
                                        self.active_faults.pop(target_uid, None)
                                    self.log_event(f"🔧 QGC MAVLink Console cleared fault on {target_uid}")
                                    _reply_console(f"SUCCESS: Fault cleared on {target_uid}. All sensors nominal.")
                                elif sensor_hit:
                                    with self.lock:
                                        self.active_faults[target_uid] = sensor_hit
                                    self.log_event(f"🔴 QGC MAVLink Console injected {sensor_hit} on {target_uid}")
                                    _reply_console(f"SUCCESS: Injected {sensor_hit} on {target_uid}! Autonomous safe action engaged.")
                                else:
                                    _reply_console(f"Unrecognized sensor in '{cmd_str}'. Try 'failure gps off' or 'failure help'.")
                            else:
                                _reply_console(f"Received '{cmd_str}' on {uav_id}. Type 'failure help' for fault injection commands.")
                except Exception:
                    pass

                now_ms = int(time.time() * 1000) & 0xFFFFFFFF
                now_us = int(time.time() * 1e6) & 0xFFFFFFFFFFFFFFFF
                pos = state_summary.get("pos", [0.0, 0.0, 0.0])
                vel = state_summary.get("vel", [0.0, 0.0, 0.0])
                batt_pct = int(max(0.0, min(100.0, state_summary.get("battery_pct", 100.0))))
                is_ground = pos[2] < 0.2

                is_leader = bool(state_summary.get("is_leader", False))
                # Distinct MAV_TYPE so QGC visually distinguishes the Master/Leader UAV on map
                veh_type = mavutil.mavlink.MAV_TYPE_HEXAROTOR if is_leader else mavutil.mavlink.MAV_TYPE_QUADROTOR

                # PX4 Custom Flight Mode encoding: (main_mode << 16) | (sub_mode << 24)
                # Main modes: 1=MANUAL, 2=ALTCTL, 3=POSCTL, 4=AUTO, 6=OFFBOARD
                # Sub modes for AUTO: 1=READY, 2=TAKEOFF, 3=LOITER, 4=MISSION, 5=RTL, 6=LAND
                act = state_summary.get("action", "NOMINAL_CONTINUE")
                if is_ground:
                    custom_mode = (3 << 16)  # POSCTL (Position Hold / Standby)
                elif act == "RETURN_TO_HOME":
                    custom_mode = (4 << 16) | (5 << 24)  # AUTO RTL
                elif act == "FAILSAFE_LAND":
                    custom_mode = (4 << 16) | (6 << 24)  # AUTO LAND
                elif act in ("ALTITUDE_HOLD_HOVER", "STANDBY_LOITER", "NFZ_REROUTE"):
                    custom_mode = (4 << 16) | (3 << 24)  # AUTO LOITER / HOLD
                else:
                    custom_mode = (3 << 16)               # POSCTL (Position Hold / Mission)

                # 1. Heartbeat
                base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                if not is_ground:
                    base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                mav.mav.heartbeat_send(
                    veh_type,
                    mavutil.mavlink.MAV_AUTOPILOT_PX4,
                    base_mode,
                    custom_mode,
                    mavutil.mavlink.MAV_STATE_ACTIVE if not is_ground else mavutil.mavlink.MAV_STATE_STANDBY
                )

                # Periodically send Leader / Master banner to QGroundControl
                last_notice = getattr(self, f"_last_notice_{uav_id}", 0)
                if (time.time() - last_notice) > 6.0:
                    setattr(self, f"_last_notice_{uav_id}", time.time())
                    try:
                        role_tag = f"👑 [MASTER] {uav_id} LEADER" if is_leader else f"[{uav_id}] {state_summary.get('task', 'SWARM')[:16]}"
                        mav.mav.statustext_send(
                            mavutil.mavlink.MAV_SEVERITY_NOTICE if is_leader else mavutil.mavlink.MAV_SEVERITY_INFO,
                            role_tag.encode("utf-8")[:50]
                        )
                    except Exception:
                        pass

                # 2. Battery & System Status
                v_mv = int(14000 + batt_pct * 25)
                mav.mav.sys_status_send(0, 0, 0, 250, v_mv, 1500, batt_pct, 0, 0, 0, 0, 0, 0)

                # 3. GPS Coordinates & Altitude (Zurich PX4 Origin)
                lat0 = 47.397742
                lon0 = 8.545594
                d_lat = pos[1] / 111320.0
                d_lon = pos[0] / (111320.0 * 0.6769)
                cur_lat = lat0 + d_lat
                cur_lon = lon0 + d_lon
                alt_m = max(0.0, pos[2])
                hdg_deg = float(np.degrees(np.arctan2(vel[1], vel[0])) % 360) if (abs(vel[0]) > 0.01 or abs(vel[1]) > 0.01) else 0.0

                mav.mav.global_position_int_send(
                    now_ms,
                    int(cur_lat * 1e7),
                    int(cur_lon * 1e7),
                    int((488.0 + alt_m) * 1000),
                    int(alt_m * 1000),
                    int(vel[0] * 100),
                    int(vel[1] * 100),
                    int(vel[2] * 100),
                    int(hdg_deg * 100)
                )

                # 4. Attitude
                mav.mav.attitude_send(
                    now_ms, 0.0, 0.0, float(np.radians(hdg_deg)), 0.0, 0.0, 0.0
                )
            except Exception:
                pass

    def request_handoff(self, sender_id: str, waypoints: List[str], reason: str) -> Optional[str]:
        """Solicits peers to take over a degraded UAV's mission."""
        with self.lock:
            candidates = []
            sender_pos = self.peer_states.get(sender_id, {}).get("pos", [0.0, 0.0, 0.0])

            for uid, info in self.peer_states.items():
                if uid == sender_id:
                    continue
                batt = info.get("battery_pct", 0.0)
                fault = info.get("fault", "No Failure")
                avail = info.get("available", True)
                pos = info.get("pos", [0.0, 0.0, 0.0])

                if avail and batt > 35.0 and fault == "No Failure":
                    dist = ((pos[0]-sender_pos[0])**2 + (pos[1]-sender_pos[1])**2)**0.5
                    # Score: Higher battery, closer proximity is best
                    score = batt - (dist * 0.4)
                    candidates.append((uid, score, batt, dist))

            if not candidates:
                self.log_event(f"⚠️ {sender_id} requested handoff: NO PEERS AVAILABLE!")
                return None

            candidates.sort(key=lambda x: x[1], reverse=True)
            winner_id, score, win_batt, win_dist = candidates[0]
            # Mark assignee as busy
            if winner_id in self.peer_states:
                self.peer_states[winner_id]["available"] = False
                self.peer_states[winner_id]["delegated_from"] = sender_id

            self.log_event(f"🤝 {sender_id} handed off {len(waypoints)} waypoints → {winner_id} (batt {win_batt:.0f}%, dist {win_dist:.1f}m)")
            return winner_id

    def evaluate_leader_election(self, agents: List[SwarmAgentState]) -> str:
        """
        Distributed Swarm Leader Election using fitness scoring.
        If current leader is degraded or in emergency landing, triggers re-election.
        """
        with self.lock:
            if not hasattr(self, "current_leader") or not self.current_leader:
                self.current_leader = "UAV-1"

            curr = next((a for a in agents if a.uav_id == self.current_leader), None)
            leader_healthy = (
                curr is not None
                and not curr.on_ground
                and curr.energy_alert == "NORMAL"
                and curr.fault_label == "No Failure"
                and curr.final_action not in ("FAILSAFE_LAND", "RETURN_TO_HOME")
                and curr.battery_pct > 25.0
            )

            if leader_healthy:
                return self.current_leader

            # Current leader is degraded! Trigger election among active candidates
            old_leader = self.current_leader
            candidates = []
            for a in agents:
                if a.on_ground or a.final_action in ("FAILSAFE_LAND", "RETURN_TO_HOME"):
                    continue
                # ANY fault disqualifies from leadership — a faulty UAV must never lead
                if a.fault_label != "No Failure":
                    continue
                if a.battery_pct <= 20.0:
                    continue

                # Leadership fitness score: battery (40%), low risk (30%), GPS (20%), clean sensors (10%)
                f_batt = a.battery_pct / 100.0
                f_risk = 1.0 - a.composite_risk
                f_gps  = 1.0 if a.gps_valid else 0.0
                f_sens = 1.0 if a.fault_label == "No Failure" else 0.5
                fitness = round(0.4 * f_batt + 0.3 * f_risk + 0.2 * f_gps + 0.1 * f_sens, 3)
                candidates.append((a.uav_id, fitness, a.battery_pct))

            if not candidates:
                self.log_event(f"⚠️ [LEADER ELECTION] All UAVs degraded! Swarm decentralized.")
                return ""

            # Pick highest fitness candidate
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            new_leader, win_fitness, win_batt = candidates[0]

            if new_leader != old_leader:
                self.current_leader = new_leader
                self.log_event(
                    f"👑 [LEADER ELECTION] {old_leader} degraded → {new_leader} elected SWARM LEADER (Fitness: {win_fitness:.2f}, Batt: {win_batt:.0f}%)"
                )
                # Update agent roles
                for a in agents:
                    if a.uav_id == new_leader:
                        a.is_leader = True
                        a.role = "👑 Swarm Leader"
                    elif a.uav_id == old_leader:
                        a.is_leader = False
                        a.role = "Degraded (Ex-Leader)"
                    else:
                        a.is_leader = False

            return self.current_leader

    def get_peer_info(self, uav_id: str) -> List[Dict[str, Any]]:
        with self.lock:
            return [info for uid, info in self.peer_states.items() if uid != uav_id]

    def _get_gazebo_world(self) -> str:
        """Discovers active Gazebo world name or defaults to 'default'."""
        if getattr(self, "_gz_world", None):
            return self._gz_world
        try:
            import subprocess
            top_out = subprocess.check_output(["gz", "topic", "-l"], stderr=subprocess.DEVNULL, timeout=2).decode("utf-8")
            for line in top_out.splitlines():
                if line.startswith("/world/") and line.endswith("/clock"):
                    self._gz_world = line.split("/")[2]
                    return self._gz_world
        except Exception:
            pass
        self._gz_world = "default"
        return self._gz_world

    def ensure_gazebo_swarm_models(self):
        """Automatically checks Gazebo and spawns x500_1, x500_2, x500_3 if missing.
        Does NOT require PX4 SITL detection — it probes Gazebo directly via gz CLI."""
        try:
            import subprocess
            out = subprocess.check_output(["gz", "model", "--list"], stderr=subprocess.DEVNULL, timeout=3).decode("utf-8")
        except Exception:
            return  # Gazebo not reachable — skip silently

        sdf_path = "/home/samiksha/PX4-Autopilot/Tools/simulation/gz/models/x500/model.sdf"
        if not os.path.exists(sdf_path):
            return

        world = self._get_gazebo_world()
        spawn_targets = [
            ("x500_1", 2.0,  1.5,  0.2),
            ("x500_2", -2.0, 2.5,  0.2),
            ("x500_3",  1.0, -1.5, 0.2),
        ]
        for m_name, sx, sy, sz in spawn_targets:
            if m_name not in out:
                cmd = [
                    "gz", "service", "-s", f"/world/{world}/create",
                    "--reqtype", "gz.msgs.EntityFactory",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "3000",
                    "--req",
                    f'name: "{m_name}", allow_renaming: false, '
                    f'sdf: "<sdf version=\\"1.6\\"><include><uri>file://{sdf_path}</uri>'
                    f'<pose>{sx} {sy} {sz} 0 0 0</pose></include></sdf>"'
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                    self.log_event(f"🛸 Auto-spawned {m_name} into Gazebo (pos {sx},{sy})")
                except Exception:
                    pass

    def sync_gazebo_swarm(self, agents: List["SwarmAgentState"]):
        """Synchronizes 3D poses of peer UAVs (x500_1, x500_2, x500_3) in Gazebo.
        Works regardless of PX4 SITL process detection — probes Gazebo directly."""
        gz_map = {"UAV-2": "x500_1", "UAV-3": "x500_2", "UAV-4": "x500_3"}
        uav1 = next((a for a in agents if a.uav_id == "UAV-1"), None)
        uav1_flying = bool(uav1 is not None and uav1.altitude > 0.5)
        world = self._get_gazebo_world()

        for a in agents:
            if a.uav_id in gz_map:
                m_name = gz_map[a.uav_id]
                alt = a.altitude if uav1_flying else 0.2
                cmd = [
                    "gz", "service", "-s", f"/world/{world}/set_pose",
                    "--reqtype", "gz.msgs.Pose",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "150",
                    "--req", f'name: "{m_name}", position: {{x: {a.pos_x:.2f}, y: {a.pos_y:.2f}, z: {alt:.2f}}}'
                ]
                try:
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass


_SWARM_BUS = SwarmBus()


# ═══════════════════════════════════════════════════════════════════════════
# Individual UAV State Dataclass
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SwarmAgentState:
    uav_id:           str   = "UAV-1"
    role:             str   = "Primary Survey"
    is_leader:        bool  = False
    active_task:      str   = "Survey Sector Alpha"
    waypoints_queue:  List[str] = field(default_factory=lambda: ["WP1", "WP2", "WP3", "WP4"])
    available:        bool  = True

    # Realistic Routes & Tasks Navigation
    route_coords:     List[Tuple[float, float, float]] = field(default_factory=list)
    current_wp_idx:   int   = 0
    tasks_completed:  int   = 0
    total_tasks:      int   = 0

    # Inter-Drone Mesh Network & Link Quality
    link_quality_pct: float = 100.0
    comms_latency_ms: float = 2.0

    # Telemetry
    pos_x:            float = 0.0
    pos_y:            float = 0.0
    altitude:         float = 10.0
    vx:               float = 0.0
    vy:               float = 0.0
    vz:               float = 0.0
    battery_pct:      float = 95.0
    gps_valid:        bool  = True
    on_ground:        bool  = False

    # FDI
    fault_label:      str   = "No Failure"
    confidence:       float = 98.0

    # Energy Alert
    energy_alert:     str   = "NORMAL"   # NORMAL | LOW | CRITICAL

    # Risk
    risk_stability:   float = 0.0
    risk_navigation:  float = 0.0
    risk_energy:      float = 0.0
    composite_risk:   float = 0.0

    # Agentic Reasoning
    agent_proposal:   str   = "NOMINAL_CONTINUE"
    reasoning_trace:  str   = "All systems nominal."

    # Swarm Handoff
    delegation_target: str  = ""
    delegation_status: str  = "N/A"

    # Guardrails
    l1_status:        str   = "PASSED"
    l1_notes:         str   = "—"
    l2_status:        str   = "PASSED"
    l2_notes:         str   = "—"

    # No-Fly Zone (NFZ) Geofencing
    nfz_violation:    bool  = False
    nfz_nearest:      str   = "—"
    nfz_dist:         float = 999.0

    # Final Action
    final_action:     str   = "NOMINAL_CONTINUE"
    hover_ticks:      int   = 0
    beacon_active:    bool  = False
    reroute_active:   bool  = False
    reroute_target:   List[float] = field(default_factory=list)
    log:              List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 10-Node LangGraph Agent Pipeline (Instantiated per UAV)
# ═══════════════════════════════════════════════════════════════════════════

def node_collect_telemetry(state: SwarmAgentState) -> SwarmAgentState:
    # If UAV-1 and live PX4/Gazebo telemetry is streaming, take physical readings directly
    if state.uav_id == "UAV-1" and _SWARM_BUS.is_connected_to_px4() and _SWARM_BUS.latest_px4_telemetry:
        tel = _SWARM_BUS.latest_px4_telemetry
        if "pos_x" in tel:
            state.pos_x = float(tel["pos_x"])
        if "pos_y" in tel:
            state.pos_y = float(tel["pos_y"])
        if "pos_z" in tel:
            state.altitude = float(tel["pos_z"])
        if "vx" in tel:
            state.vx = float(tel["vx"])
        if "vy" in tel:
            state.vy = float(tel["vy"])
        if "vz" in tel:
            state.vz = float(tel["vz"])
        if "battery_pct" in tel:
            state.battery_pct = float(tel["battery_pct"])
    else:
        # Realistic Waypoint Navigation & Movement Simulation
        if state.on_ground:
            state.vx = 0.0
            state.vy = 0.0
            state.vz = 0.0
            state.altitude = 0.0
        elif state.final_action in ("NOMINAL_CONTINUE", "NFZ_REROUTE"):
            if state.route_coords:
                wp = state.route_coords[state.current_wp_idx]

                # Waypoint reached check (< 3.5 meters to real assigned waypoint)
                real_dx = wp[0] - state.pos_x
                real_dy = wp[1] - state.pos_y
                real_dist = (real_dx**2 + real_dy**2)**0.5

                if real_dist < 3.5:
                    if state.tasks_completed < state.total_tasks:
                        state.tasks_completed += 1
                    prev_idx = state.current_wp_idx
                    state.current_wp_idx = (state.current_wp_idx + 1) % len(state.route_coords)
                    state.reroute_active = False
                    state.reroute_target = []
                    _SWARM_BUS.log_event(
                        f"🎯 [{state.uav_id}] Completed WP-{prev_idx+1} ({state.tasks_completed}/{state.total_tasks} Tasks) → Next WP-{state.current_wp_idx+1}"
                    )
                    _SWARM_BUS.notify_task_completed(
                        uav_id=state.uav_id,
                        wp_idx=prev_idx + 1,
                        tasks_done=state.tasks_completed,
                        total_tasks=state.total_tasks,
                    )
                    wp = state.route_coords[state.current_wp_idx]

                # If currently tracking a dynamic bypass and reached within 3m, clear it to resume path to wp
                if getattr(state, "reroute_target", None) and len(state.reroute_target) == 2:
                    b_dist = ((state.reroute_target[0] - state.pos_x)**2 + (state.reroute_target[1] - state.pos_y)**2)**0.5
                    if b_dist < 3.0:
                        state.reroute_target = []
                        state.reroute_active = False

                target_x, target_y = wp[0], wp[1]
                cur_dx = target_x - state.pos_x
                cur_dy = target_y - state.pos_y
                seg_sq = cur_dx**2 + cur_dy**2

                # ── Proactive Predictive Line-Segment NFZ Avoidance ──
                if seg_sq > 0.1:
                    path_obstructed = False
                    for nfz in _SWARM_BUS.nfz_zones:
                        cx, cy, r_nfz = nfz["x"], nfz["y"], nfz["radius"]
                        buffer_r = r_nfz + 10.0  # 10-meter clearance envelope

                        t_proj = max(0.0, min(1.0, ((cx - state.pos_x)*cur_dx + (cy - state.pos_y)*cur_dy) / seg_sq))
                        qx = state.pos_x + t_proj * cur_dx
                        qy = state.pos_y + t_proj * cur_dy
                        dist_path = ((qx - cx)**2 + (qy - cy)**2)**0.5
                        dist_drone = ((state.pos_x - cx)**2 + (state.pos_y - cy)**2)**0.5

                        if dist_path < buffer_r or dist_drone < buffer_r:
                            path_obstructed = True
                            state.reroute_active = True
                            state.final_action = "NFZ_REROUTE"
                            state.agent_proposal = "NFZ_REROUTE"

                            # If active bypass waypoint is already set and > 3m away, track it stably
                            if getattr(state, "reroute_target", None) and len(state.reroute_target) == 2:
                                rem_b_dist = ((state.reroute_target[0] - state.pos_x)**2 + (state.reroute_target[1] - state.pos_y)**2)**0.5
                                if rem_b_dist >= 3.0:
                                    target_x, target_y = state.reroute_target[0], state.reroute_target[1]
                                    break

                            # Normal vector away from center
                            nx = (state.pos_x - cx) if dist_drone > 0.1 else (qx - cx)
                            ny = (state.pos_y - cy) if dist_drone > 0.1 else (qy - cy)
                            n_len = max(0.1, (nx**2 + ny**2)**0.5)
                            nx /= n_len
                            ny /= n_len

                            # Tangent vector: choose direction pointing toward destination wp
                            tx, ty = -ny, nx
                            if (tx * cur_dx + ty * cur_dy) < 0:
                                tx, ty = -tx, -ty

                            # Dynamic Bypass Waypoint: pushed 14m outside the zone
                            bypass_x = cx + nx * (r_nfz + 14.0) + tx * 10.0
                            bypass_y = cy + ny * (r_nfz + 14.0) + ty * 10.0
                            target_x, target_y = bypass_x, bypass_y
                            state.reroute_target = [round(bypass_x, 1), round(bypass_y, 1)]
                            state.reasoning_trace = (
                                f"⚡ 2-Layer Guardrail: Path to WP blocked by {nfz['name']}! "
                                f"Proactively rerouting around safe perimeter (+14m clearance)."
                            )
                            break

                    if not path_obstructed:
                        state.reroute_active = False
                        state.reroute_target = []

                dx = target_x - state.pos_x
                dy = target_y - state.pos_y
                dist_2d = max(0.1, (dx**2 + dy**2)**0.5)

                # Cruise speed towards active target (either bypass or waypoint)
                cruise_spd = 2.2
                state.vx = (dx / dist_2d) * cruise_spd
                state.vy = (dy / dist_2d) * cruise_spd

                state.pos_x += state.vx * 0.2
                state.pos_y += state.vy * 0.2
                state.battery_pct = max(0.0, state.battery_pct - 0.04)

        elif state.final_action == "RETURN_TO_HOME":
            target_x, target_y = 0.0, 0.0
            state.reroute_active = False
            state.reroute_target = []

            # Check if RTL straight line (pos -> (0,0)) intersects any NFZ
            cur_dx = target_x - state.pos_x
            cur_dy = target_y - state.pos_y
            seg_sq = cur_dx**2 + cur_dy**2

            if seg_sq > 0.1:
                for nfz in _SWARM_BUS.nfz_zones:
                    cx, cy, r_nfz = nfz["x"], nfz["y"], nfz["radius"]
                    buffer_r = r_nfz + 10.0

                    t_proj = max(0.0, min(1.0, ((cx - state.pos_x)*cur_dx + (cy - state.pos_y)*cur_dy) / seg_sq))
                    qx = state.pos_x + t_proj * cur_dx
                    qy = state.pos_y + t_proj * cur_dy
                    dist_path = ((qx - cx)**2 + (qy - cy)**2)**0.5
                    dist_drone = ((state.pos_x - cx)**2 + (state.pos_y - cy)**2)**0.5

                    if dist_path < buffer_r or dist_drone < buffer_r:
                        state.reroute_active = True
                        nx = (state.pos_x - cx) if dist_drone > 0.1 else (qx - cx)
                        ny = (state.pos_y - cy) if dist_drone > 0.1 else (qy - cy)
                        n_len = max(0.1, (nx**2 + ny**2)**0.5)
                        nx /= n_len
                        ny /= n_len
                        tx, ty = -ny, nx
                        if (tx * cur_dx + ty * cur_dy) < 0:
                            tx, ty = -tx, -ty
                        bypass_x = cx + nx * (r_nfz + 14.0) + tx * 8.0
                        bypass_y = cy + ny * (r_nfz + 14.0) + ty * 8.0
                        target_x, target_y = bypass_x, bypass_y
                        state.reroute_target = [round(bypass_x, 1), round(bypass_y, 1)]
                        break

            dist = max(0.1, ((target_x - state.pos_x)**2 + (target_y - state.pos_y)**2)**0.5)
            if dist > 1.5:
                state.vx = ((target_x - state.pos_x) / dist) * 1.8
                state.vy = ((target_y - state.pos_y) / dist) * 1.8
                state.pos_x += state.vx * 0.2
                state.pos_y += state.vy * 0.2
            else:
                if target_x == 0.0 and target_y == 0.0:
                    state.vx = 0.0
                    state.vy = 0.0
                    state.altitude = max(0.0, state.altitude - 0.4)
                    if state.altitude <= 0.2:
                        state.on_ground = True
                        state.altitude = 0.0
            state.battery_pct = max(0.0, state.battery_pct - 0.03)

        elif state.final_action == "FAILSAFE_LAND":
            state.vx *= 0.5
            state.vy *= 0.5
            state.altitude = max(0.0, state.altitude - 0.5)
            state.battery_pct = max(0.0, state.battery_pct - 0.02)
            if state.altitude <= 0.2:
                state.on_ground = True
                state.altitude = 0.0
                state.vx = 0.0
                state.vy = 0.0

        elif state.final_action == "ALTITUDE_HOLD_HOVER":
            state.vx = 0.0
            state.vy = 0.0
            state.battery_pct = max(0.0, state.battery_pct - 0.03)
            state.hover_ticks += 1

            # 30-Second Hover Failsafe Timeout (120 ticks at 4Hz)
            if state.hover_ticks >= 120 and not state.on_ground:
                state.final_action = "FAILSAFE_LAND"
                state.agent_proposal = "FAILSAFE_LAND"
                state.reasoning_trace = "⏱️ GPS Hover Timeout Exceeded (30s) — Executing Controlled Failsafe Land to preserve hardware."
                _SWARM_BUS.log_event(f"⏱️ [{state.uav_id}] GPS Hover Timeout Exceeded (30s) → Auto Failsafe Landing")
        else:
            # If drone was hovering and fault was cleared, announce recovery
            if getattr(state, "hover_ticks", 0) > 0:
                if state.fault_label == "No Failure":
                    _SWARM_BUS.log_event(f"🛰️ [{state.uav_id}] GPS Signal Restored → Resumed Navigation / RTL Corridor")
                state.hover_ticks = 0

        # ── HARD CBF SAFETY GEOFENCE CLAMP (Layer 2) ──
        # Guaranteed physical wall: executes AFTER any position update for all airborne UAVs
        if not state.on_ground:
            for nfz in _SWARM_BUS.nfz_zones:
                d_c = ((state.pos_x - nfz["x"])**2 + (state.pos_y - nfz["y"])**2)**0.5
                hard_wall = nfz["radius"] + 5.0  # 5-meter absolute zero-tolerance boundary
                if d_c < hard_wall:
                    norm_x = (state.pos_x - nfz["x"]) / max(0.001, d_c)
                    norm_y = (state.pos_y - nfz["y"]) / max(0.001, d_c)
                    state.pos_x = nfz["x"] + norm_x * hard_wall
                    state.pos_y = nfz["y"] + norm_y * hard_wall
                    state.vx = norm_x * 2.5
                    state.vy = norm_y * 2.5
                    state.final_action = "NFZ_REROUTE"
                    state.reroute_active = True

        # Field Emergency Beacon: When landed on critical battery, activate ELT beacon
        if state.on_ground and state.energy_alert == "CRITICAL":
            if not getattr(state, "beacon_active", False):
                state.beacon_active = True
                state.role = "📍 Beacon Active (Awaiting Retrieval)"
                _SWARM_BUS.log_event(f"📍 [FIELD BEACON] {state.uav_id} safe touchdown at ({state.pos_x:.1f}, {state.pos_y:.1f}) | ELT Beacon Active | Ground Retrieval Dispatched")

    # NFZ proximity check against all registered no-fly zones
    state.nfz_violation = False
    state.nfz_nearest = "—"
    state.nfz_dist = 999.0
    for nfz in _SWARM_BUS.nfz_zones:
        dx = state.pos_x - nfz["x"]
        dy = state.pos_y - nfz["y"]
        dist = (dx**2 + dy**2)**0.5
        margin = dist - nfz["radius"]
        if margin < state.nfz_dist:
            state.nfz_dist = round(margin, 1)
            state.nfz_nearest = nfz["name"]
        if margin < 5.0:  # Within 5m of NFZ boundary → VIOLATION WARNING
            state.nfz_violation = True

    # Mesh Link Quality and Latency to Swarm Leader
    lead_id = getattr(_SWARM_BUS, "current_leader", "UAV-1") or "UAV-1"
    if state.uav_id == lead_id:
        state.link_quality_pct = 100.0
        state.comms_latency_ms = 0.5
    elif state.on_ground:
        state.link_quality_pct = 0.0
        state.comms_latency_ms = 999.0
    else:
        lead_pos = _SWARM_BUS.peer_states.get(lead_id, {}).get("pos", [0.0, 0.0, 10.0])
        dist_lead = ((state.pos_x - lead_pos[0])**2 + (state.pos_y - lead_pos[1])**2)**0.5
        state.link_quality_pct = round(max(25.0, min(100.0, 100.0 - (dist_lead / 160.0) * 55.0)), 1)
        state.comms_latency_ms = round(1.8 + (dist_lead / 28.0), 1)

    return state


def node_monitor_energy(state: SwarmAgentState) -> SwarmAgentState:
    b = state.battery_pct
    if b <= 15.0:
        state.energy_alert = "CRITICAL"
    elif b <= 30.0:
        state.energy_alert = "LOW"
    else:
        state.energy_alert = "NORMAL"
    return state


def node_run_fdi(state: SwarmAgentState) -> SwarmAgentState:
    # Sensor FDI model check (already loaded from ExtraTrees)
    # Fault label is preserved if injected or computed
    return state


def node_compute_risk(state: SwarmAgentState) -> SwarmAgentState:
    risk_map = {
        "No Failure": 0.0, "Barometer Failure": 0.35, "Compass Failure": 0.30,
        "Accelerometer Failure": 0.75, "Gyroscope Failure": 0.85, "GPS Failure": 0.60
    }
    base = risk_map.get(state.fault_label, 0.0)
    conf = state.confidence / 100.0
    state.risk_stability = round(base * conf * 1.1, 2)
    state.risk_navigation = round(base * conf if state.gps_valid else 0.7, 2)

    b = state.battery_pct
    if b <= 15.0:
        state.risk_energy = round(min(1.0, (15.0 - b) / 15.0 + 0.5), 2)
    elif b <= 30.0:
        state.risk_energy = round((30.0 - b) / 30.0 * 0.7, 2)
    else:
        state.risk_energy = 0.0

    state.composite_risk = max(state.risk_stability, state.risk_navigation, state.risk_energy)
    return state


def node_agentic_reasoning(state: SwarmAgentState) -> SwarmAgentState:
    fault = state.fault_label
    energy = state.energy_alert
    gps_ok = state.gps_valid
    dist_home = (state.pos_x**2 + state.pos_y**2)**0.5

    # If already landed
    if state.on_ground:
        state.agent_proposal = "LANDED_SAFE"
        state.reasoning_trace = "Landed on ground safely. Motors disarmed."
        return state

    # Rule 1: Critical stabilization sensor lost
    if fault in ("Gyroscope Failure", "Accelerometer Failure"):
        if state.waypoints_queue:
            state.agent_proposal = "DELEGATE_TO_SWARM"
            state.reasoning_trace = f"🔴 {fault} detected! Delegating {len(state.waypoints_queue)} WPs to swarm peer and descending safely."
        else:
            state.agent_proposal = "FAILSAFE_LAND"
            state.reasoning_trace = f"🔴 {fault} detected. Critical attitude control degraded → Land immediately vertically."

    # Rule 2: Battery CRITICAL (<= 15%)
    elif energy == "CRITICAL":
        if state.waypoints_queue:
            state.agent_proposal = "DELEGATE_TO_SWARM"
            state.reasoning_trace = f"Battery critical ({state.battery_pct:.1f}%). Requesting swarm peer to take remaining {len(state.waypoints_queue)} WPs, then land."
        else:
            state.agent_proposal = "FAILSAFE_LAND"
            state.reasoning_trace = f"Battery critical ({state.battery_pct:.1f}%) and mission done → Failsafe descent."

    # Rule 3: GPS Lost
    elif not gps_ok or fault == "GPS Failure":
        if state.waypoints_queue:
            state.agent_proposal = "DELEGATE_TO_SWARM"
            state.reasoning_trace = f"🔴 GPS Failure! Cannot navigate safely. Delegating {len(state.waypoints_queue)} WPs to peer and holding altitude."
        else:
            state.agent_proposal = "ALTITUDE_HOLD_HOVER"
            state.reasoning_trace = "🔴 GPS signal lost or degraded. Holding position in place."

    # Rule 4: Battery LOW (15-30%)
    elif energy == "LOW":
        if dist_home > 40.0 and state.waypoints_queue:
            state.agent_proposal = "DELEGATE_TO_SWARM"
            state.reasoning_trace = f"Battery low ({state.battery_pct:.1f}%) & far from base ({dist_home:.0f}m). Delegating task sector to peer; returning home."
        else:
            state.agent_proposal = "RETURN_TO_HOME"
            state.reasoning_trace = f"Battery low ({state.battery_pct:.1f}%), within return radius ({dist_home:.0f}m). Returning to home base."

    # Rule 5: Non-critical sensor fault
    elif fault in ("Barometer Failure", "Compass Failure"):
        state.agent_proposal = "ALTITUDE_HOLD_HOVER"
        state.reasoning_trace = f"{fault} detected. Pausing trajectory to stabilize attitude and heading."

    # Rule 6: Moderate risk
    elif state.composite_risk >= 0.4:
        state.agent_proposal = "ALTITUDE_HOLD_HOVER"
        state.reasoning_trace = f"Risk elevated ({state.composite_risk:.2f}). Pausing for assessment."

    # Rule 6.5: No-Fly Zone violation — reroute or delegate
    elif state.nfz_violation:
        if state.nfz_dist < 0.0:
            # INSIDE the NFZ! Emergency reroute + delegate task
            if state.waypoints_queue:
                state.agent_proposal = "DELEGATE_TO_SWARM"
                state.reasoning_trace = (
                    f"⛔ INSIDE {state.nfz_nearest} (margin {state.nfz_dist:.1f}m)! "
                    f"Delegating {len(state.waypoints_queue)} WPs to swarm peer and retreating."
                )
            else:
                state.agent_proposal = "RETURN_TO_HOME"
                state.reasoning_trace = f"⛔ INSIDE {state.nfz_nearest}! No waypoints left, returning to home."
            # Reverse velocity to flee NFZ
            nfz_match = next((n for n in _SWARM_BUS.nfz_zones if n["name"] == state.nfz_nearest), None)
            if nfz_match:
                dx = state.pos_x - nfz_match["x"]
                dy = state.pos_y - nfz_match["y"]
                norm = max(0.01, (dx**2 + dy**2)**0.5)
                state.vx = (dx / norm) * 1.5  # Flee outward at 1.5 m/s
                state.vy = (dy / norm) * 1.5
        else:
            # Approaching NFZ boundary (<5m) — reroute around it
            state.agent_proposal = "NFZ_REROUTE"
            nfz_match = next((n for n in _SWARM_BUS.nfz_zones if n["name"] == state.nfz_nearest), None)
            if nfz_match:
                # Steer tangentially (perpendicular to NFZ center vector)
                dx = state.pos_x - nfz_match["x"]
                dy = state.pos_y - nfz_match["y"]
                norm = max(0.01, (dx**2 + dy**2)**0.5)
                # Tangent vector (rotate 90°) * 2.2 m/s + outward repulsive buffer * 1.2 m/s
                state.vx = (-dy / norm) * 2.2 + (dx / norm) * 1.2
                state.vy = (dx / norm) * 2.2 + (dy / norm) * 1.2
            state.reasoning_trace = (
                f"⚠️ Approaching {state.nfz_nearest} (margin {state.nfz_dist:.1f}m). "
                f"CBF activated — rerouting tangentially around NFZ boundary."
            )

    # Rule 7: Nominal
    else:
        state.agent_proposal = "NOMINAL_CONTINUE"
        state.reasoning_trace = f"All parameters nominal (batt {state.battery_pct:.0f}%, risk {state.composite_risk:.2f})."

    return state


def node_swarm_coordination(state: SwarmAgentState) -> SwarmAgentState:
    if state.agent_proposal == "DELEGATE_TO_SWARM":
        # Capture remaining route waypoints BEFORE clearing
        remaining_route = []
        if state.route_coords:
            remaining_route = list(state.route_coords[state.current_wp_idx:])

        assigned_peer = _SWARM_BUS.request_handoff(
            sender_id=state.uav_id,
            waypoints=list(state.waypoints_queue),
            reason=f"Degradation ({state.energy_alert} batt / {state.fault_label})"
        )
        if assigned_peer:
            state.delegation_target = assigned_peer
            state.delegation_status = "DELEGATED"
            # Store remaining route in SwarmBus for the peer to pick up
            with _SWARM_BUS.lock:
                if not hasattr(_SWARM_BUS, '_pending_handoff_routes'):
                    _SWARM_BUS._pending_handoff_routes = {}
                _SWARM_BUS._pending_handoff_routes[assigned_peer] = {
                    "from": state.uav_id,
                    "remaining_route": remaining_route,
                    "task_name": state.active_task,
                    "wp_names": list(state.waypoints_queue),
                }
            state.waypoints_queue.clear()
            state.active_task = f"Degraded – Transferred to {assigned_peer}"
            # After delegating, recovery action:
            if state.battery_pct > 20.0 and state.gps_valid and state.fault_label not in ("Gyroscope Failure", "Accelerometer Failure"):
                state.agent_proposal = "RETURN_TO_HOME"
            else:
                state.agent_proposal = "FAILSAFE_LAND"
        else:
            state.delegation_status = "NO_PEER_AVAILABLE"
            state.agent_proposal = "FAILSAFE_LAND"
            state.reasoning_trace += " | Swarm exhausted: falling back to emergency land."
    else:
        state.delegation_status = "N/A"

    return state


def node_guardrail_l1(state: SwarmAgentState) -> SwarmAgentState:
    if state.on_ground:
        state.final_action = "LANDED_SAFE"
        state.l1_status = "PASSED"
        state.l1_notes = "Grounded / Disarmed"
        return state

    action = state.agent_proposal
    notes = []
    passed = True

    if action == "RETURN_TO_HOME" and not state.gps_valid:
        action = "ALTITUDE_HOLD_HOVER"
        notes.append("No GPS: RTH blocked → HOVER")
        passed = False

    if action == "NFZ_REROUTE" and not state.gps_valid:
        action = "ALTITUDE_HOLD_HOVER"
        notes.append("No GPS: NFZ reroute blocked → HOVER")
        passed = False

    if state.altitude < 0.3 and action not in ("FAILSAFE_LAND", "LANDED_SAFE", "NOMINAL_CONTINUE"):
        action = "FAILSAFE_LAND"
        notes.append("Alt < 0.3m → LAND")
        passed = False

    state.l1_status = "PASSED" if passed else "MODIFIED"
    state.l1_notes = " | ".join(notes) if notes else "Policy OK"
    state.final_action = action
    return state


def node_guardrail_l2(state: SwarmAgentState) -> SwarmAgentState:
    action = state.final_action
    if state.on_ground or action == "LANDED_SAFE":
        state.l2_status = "PASSED"
        state.l2_notes = "Zero thrust (Landed)"
        return state
    u_nom = -2.0 if action == "FAILSAFE_LAND" else (-0.8 if action == "RETURN_TO_HOME" else 0.0)

    if _CBF_AVAILABLE:
        cbf = UAVControlBarrierFunction()
        vz_down = max(0.0, -u_nom)
        # Gather peer positions for swarm separation constraint
        peer_positions = []
        for info in _SWARM_BUS.get_peer_info(state.uav_id):
            peer_positions.append(info.get("pos", [0.0, 0.0, 0.0]))
        current_pos = [state.pos_x, state.pos_y, state.altitude]
        # Gather NFZ zones for geofence constraint
        nfz_for_cbf = [{"x": z["x"], "y": z["y"], "radius": z["radius"]} for z in _SWARM_BUS.nfz_zones]

        u_safe_vec, cbf_active, _, cbf_info = cbf.filter_control(
            np.array([state.vx, state.vy, vz_down]),
            state.altitude,
            peer_positions=peer_positions if peer_positions else None,
            current_pos=current_pos,
            nfz_zones=nfz_for_cbf if nfz_for_cbf else None,
        )
        # Apply CBF-corrected velocity back to the agent
        if cbf_active:
            state.vx = float(u_safe_vec[0])
            state.vy = float(u_safe_vec[1])
        intervened = bool(cbf_active)
        cbf_detail = "NFZ+Swarm+Ground" if state.nfz_violation else "Ground+Swarm"
    else:
        intervened = (state.altitude < 1.0 and u_nom < -0.5)
        cbf_detail = "Fallback"

    if intervened:
        state.l2_status = "INTERVENED"
        state.l2_notes = f"CBF modified vel ({cbf_detail})"
    else:
        state.l2_status = "PASSED"
        state.l2_notes = f"Safe barrier ({cbf_detail})"

    return state


def node_execute_action(state: SwarmAgentState) -> SwarmAgentState:
    return state


def node_finalize(state: SwarmAgentState) -> SwarmAgentState:
    # Broadcast heartbeat to the SwarmBus
    _SWARM_BUS.broadcast_heartbeat(
        uav_id=state.uav_id,
        state_summary={
            "pos": [state.pos_x, state.pos_y, state.altitude],
            "vel": [state.vx, state.vy, state.vz],
            "battery_pct": state.battery_pct,
            "fault": state.fault_label,
            "available": state.available,
            "task": state.active_task,
            "action": state.final_action,
            "is_leader": state.is_leader,
            "role": state.role,
            "current_wp_idx": state.current_wp_idx,
            "tasks_completed": state.tasks_completed,
            "total_tasks": state.total_tasks,
            "link_quality_pct": state.link_quality_pct,
            "comms_latency_ms": state.comms_latency_ms,
        }
    )
    return state


def build_uav_graph():
    g = StateGraph(SwarmAgentState)
    g.add_node("telemetry",   node_collect_telemetry)
    g.add_node("energy",      node_monitor_energy)
    g.add_node("fdi",         node_run_fdi)
    g.add_node("risk",        node_compute_risk)
    g.add_node("reasoning",   node_agentic_reasoning)
    g.add_node("swarm",       node_swarm_coordination)
    g.add_node("guardrail_l1",node_guardrail_l1)
    g.add_node("guardrail_l2",node_guardrail_l2)
    g.add_node("execute",     node_execute_action)
    g.add_node("finalize",    node_finalize)

    g.add_edge(START,         "telemetry")
    g.add_edge("telemetry",   "energy")
    g.add_edge("energy",      "fdi")
    g.add_edge("fdi",         "risk")
    g.add_edge("risk",        "reasoning")
    g.add_edge("reasoning",   "swarm")
    g.add_edge("swarm",       "guardrail_l1")
    g.add_edge("guardrail_l1","guardrail_l2")
    g.add_edge("guardrail_l2","execute")
    g.add_edge("execute",     "finalize")
    g.add_edge("finalize",    END)
    return g.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Swarm Orchestrator & Live Terminal Dashboard
# ═══════════════════════════════════════════════════════════════════════════
_COLORS = {
    "CYAN": "\x1b[36m", "GREEN": "\x1b[32m", "YELLOW": "\x1b[33m",
    "RED": "\x1b[31m", "BOLD": "\x1b[1m", "RESET": "\x1b[0m",
    "WHITE": "\x1b[97m", "MAGENTA": "\x1b[35m", "BLUE": "\x1b[34m",
}

def col(text: str, c: str) -> str:
    return f"{_COLORS[c]}{text}{_COLORS['RESET']}"

def mini_bar(pct: float, width: int = 10) -> str:
    filled = max(0, min(width, int(pct / 100.0 * width)))
    color = "GREEN" if pct > 30 else ("YELLOW" if pct > 15 else "RED")
    return col("█" * filled + "░" * (width - filled), color)


def export_master_system_report(agents: List[SwarmAgentState], tick: int, report_path: str = "/home/samiksha/uav_fdi_data/master_system_report.json", task_event: dict = None):
    """
    Aggregates full swarm mission and telemetry data from the Master UAV and exports
    to a structured JSON file so external systems and operators have total transparency.
    When task_event is provided, the report is event-triggered (task completion);
    otherwise it is a periodic telemetry snapshot.
    """
    lead_id = getattr(_SWARM_BUS, "current_leader", "UAV-1") or "UAV-1"
    master_agent = next((a for a in agents if a.uav_id == lead_id), None)
    if not master_agent and agents:
        master_agent = agents[0]
        lead_id = master_agent.uav_id

    total_tasks = sum(a.total_tasks for a in agents)
    completed_tasks = sum(min(a.total_tasks, a.tasks_completed) for a in agents)
    progress_pct = min(100.0, round((completed_tasks / max(1, total_tasks)) * 100.0, 1))

    online_agents = sum(1 for a in agents if not a.on_ground)
    fault_agents = [a for a in agents if a.fault_label != "No Failure" or a.energy_alert != "NORMAL"]

    mesh_matrix = {}
    for a in agents:
        mesh_matrix[a.uav_id] = {
            "link_quality_pct": a.link_quality_pct,
            "latency_ms": a.comms_latency_ms,
            "status": "ONLINE" if not a.on_ground else "GROUNDED_DISARMED",
            "rssi_dbm": int(-30 - (100.0 - a.link_quality_pct) * 0.6)
        }

    report_data = {
        "report_type": "AUTONOMOUS_SWARM_SYSTEM_TELEMETRY",
        "timestamp_epoch": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tick": tick,
        "swarm_master": {
            "uav_id": lead_id,
            "role": master_agent.role if master_agent else "Swarm Leader",
            "position": [round(master_agent.pos_x, 2), round(master_agent.pos_y, 2), round(master_agent.altitude, 2)] if master_agent else [0, 0, 0],
            "battery_pct": round(master_agent.battery_pct, 1) if master_agent else 100.0,
            "health": master_agent.fault_label if master_agent else "No Failure",
        },
        "swarm_health_summary": {
            "total_agents": len(agents),
            "online_agents": online_agents,
            "grounded_agents": len(agents) - online_agents,
            "active_faults_count": len(fault_agents),
            "active_faults": {a.uav_id: a.fault_label for a in fault_agents},
            "system_status": "NOMINAL" if len(fault_agents) == 0 and online_agents == len(agents) else "DEGRADED",
            "simulation_backend": "PX4_SITL_GAZEBO" if _SWARM_BUS.is_connected_to_px4() else "STANDALONE_SIM"
        },
        "swarm_mission_progress": {
            "total_tasks_completed": completed_tasks,
            "total_tasks_assigned": total_tasks,
            "progress_pct": progress_pct,
            "mission_status": "IN_PROGRESS" if progress_pct < 100.0 else "COMPLETED"
        },
        "mesh_connectivity_matrix": mesh_matrix,
        "agents_telemetry": [
            {
                "uav_id": a.uav_id,
                "role": a.role,
                "is_leader": a.is_leader,
                "position": {"x": round(a.pos_x, 2), "y": round(a.pos_y, 2), "z": round(a.altitude, 2)},
                "velocity": {"vx": round(a.vx, 2), "vy": round(a.vy, 2), "vz": round(a.vz, 2)},
                "battery_pct": round(a.battery_pct, 1),
                "energy_alert": a.energy_alert,
                "fdi_fault": a.fault_label,
                "decision_proposal": a.agent_proposal,
                "action": a.final_action,
                "current_waypoint_idx": a.current_wp_idx,
                "tasks_completed": a.tasks_completed,
                "total_tasks": a.total_tasks,
                "active_task": a.active_task,
                "nfz_distance_m": round(a.nfz_dist, 1),
                "nfz_violation": a.nfz_violation,
                "reroute_active": getattr(a, "reroute_active", False),
                "reroute_target": getattr(a, "reroute_target", []),
                "link_quality_pct": a.link_quality_pct,
                "comms_latency_ms": a.comms_latency_ms
            }
            for a in agents
        ],
        "active_nfz_geofences": _SWARM_BUS.nfz_zones,
        "report_trigger": "TASK_COMPLETION_EVENT" if task_event else "PERIODIC_SNAPSHOT",
        "task_completion_event": task_event if task_event else None,
    }

    try:
        tmp_path = report_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(report_data, f, indent=2)
        os.replace(tmp_path, report_path)
    except Exception:
        pass

    # ── Non-disruptive Flight Telemetry CSV Logging ──
    try:
        csv_path = "/home/samiksha/uav_fdi_data/flight_telemetry_log.csv"
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            if not file_exists:
                writer.writerow([
                    "tick", "timestamp_iso", "uav_id", "role", "is_leader",
                    "pos_x", "pos_y", "altitude", "vx", "vy", "vz",
                    "battery_pct", "fault_label", "final_action",
                    "current_wp_idx", "tasks_completed", "total_tasks",
                    "nfz_dist", "nfz_violation", "link_quality_pct", "comms_latency_ms"
                ])
            for a in agents:
                writer.writerow([
                    tick, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    a.uav_id, a.role, int(a.is_leader),
                    round(a.pos_x, 2), round(a.pos_y, 2), round(a.altitude, 2),
                    round(a.vx, 2), round(a.vy, 2), round(a.vz, 2),
                    round(a.battery_pct, 1), a.fault_label, a.final_action,
                    a.current_wp_idx, a.tasks_completed, a.total_tasks,
                    round(a.nfz_dist, 1), int(a.nfz_violation),
                    round(a.link_quality_pct, 1), round(a.comms_latency_ms, 1)
                ])
    except Exception:
        pass




def render_swarm_dashboard(agents: List[SwarmAgentState], tick: int):
    N = len(agents)
    print("\x1b[2J\x1b[H", end="")
    CW = 26  # Column width per UAV
    W = max(110, N * (CW + 5) + 6)  # Dynamically adapt width to N agents
    print(col("╔" + "═" * (W - 2) + "╗", "CYAN"))
    title = f"  🛸 AUTONOMOUS UAV SWARM — AGENTIC AI & FAULT-TOLERANT CONTROL (LangGraph × {N})  "
    print(col("║", "CYAN") + col(f"{title:^{W-2}}", "BOLD") + col("║", "CYAN"))
    print(col("╠" + "═" * (W - 2) + "╣", "CYAN"))

    # Global Swarm Status Bar
    active_count = sum(1 for a in agents if not a.on_ground)
    fault_count  = sum(1 for a in agents if a.fault_label != "No Failure" or a.energy_alert != "NORMAL")
    nfz_count    = sum(1 for a in agents if a.nfz_violation)
    avg_batt     = sum(a.battery_pct for a in agents) / max(1, N)

    lead_name = _SWARM_BUS.current_leader if hasattr(_SWARM_BUS, "current_leader") and _SWARM_BUS.current_leader else "DECENTRALIZED"
    lead_badge = col(f"👑 LEADER: {lead_name}", "YELLOW")
    health_badge = col(f"HEALTH: {active_count}/{N} Online", "GREEN" if active_count == N else "YELLOW")
    fault_badge  = col(f"FAULTS: {fault_count}", "RED" if fault_count > 0 else "GREEN")
    sim_mode = col("🎮 PX4+GAZEBO: LIVE", "GREEN") if _SWARM_BUS.is_connected_to_px4() else col("⚙️ MODE: STANDALONE", "CYAN")
    batt_badge   = f"BATT: {avg_batt:.1f}%"
    tick_badge   = f"TICK: {tick:04d}"

    status_line = f"  {lead_badge}   │   {health_badge}   │   {fault_badge}   │   {sim_mode}   │   {batt_badge}   │   {tick_badge}"
    print(col("║", "CYAN") + f"{status_line:<{W+24}}" + col("║", "CYAN"))
    print(col("╠" + "═" * (W - 2) + "╣", "CYAN"))

    # ── Dynamic N-agent column headers ──
    def header_card(a: SwarmAgentState) -> str:
        if a.is_leader:
            gz_tag = " (GZ)" if (a.uav_id == "UAV-1" and _SWARM_BUS.is_connected_to_px4()) else ""
            return col(f" [{a.uav_id}] 👑 LEADER{gz_tag} ", "YELLOW")
        elif "STEP DOWN" in a.role or "Ex-Leader" in a.role:
            return col(f" [{a.uav_id}] ⚠️ STEP DOWN ", "RED")
        else:
            return col(f" [{a.uav_id}] {a.role[:13]} ", "BOLD")

    h_line = "  " + "  │  ".join([f"{header_card(a):<{CW+8}}" for a in agents])
    print(col("║", "CYAN") + f"{h_line:<{W+N*8}}" + col("║", "CYAN"))
    print(col("╟" + "─" * (W - 2) + "╢", "CYAN"))

    # ── Helper: plain row ──
    def row(items: List[str]) -> None:
        line_content = "  " + "  │  ".join([f"{it:<{CW}}" for it in items])
        print(col("║", "CYAN") + f"{line_content:<{W-2}}" + col("║", "CYAN"))

    # ── Helper: colored row ──
    def color_row(items: List[str], extra: int = 9) -> None:
        line_content = "  " + "  │  ".join([f"{it:<{CW+extra}}" for it in items])
        print(col("║", "CYAN") + f"{line_content:<{W + N*extra}}" + col("║", "CYAN"))

    # Row 1: Position & Velocity
    row([f"📍 ({a.pos_x:5.1f},{a.pos_y:5.1f}) z:{a.altitude:4.1f}m" for a in agents])

    # Row 2: Velocity Vector
    row([f"🔀 v=({a.vx:+.1f},{a.vy:+.1f},{a.vz:+.1f}) m/s" for a in agents])

    # Row 3: Battery Gauge
    row([f"🔋 {mini_bar(a.battery_pct, 6)} {a.battery_pct:5.1f}%" for a in agents])

    # Row 4: FDI Status (colored)
    def fdi_color(a: SwarmAgentState) -> str:
        if a.fault_label == "No Failure":
            return col(f"🟢 FDI: {a.fault_label}", "GREEN")
        return col(f"🔴 FDI: {a.fault_label[:16]}", "RED")
    color_row([fdi_color(a) for a in agents])

    # Row 5: Energy Alert (colored)
    def energy_color(a: SwarmAgentState) -> str:
        c = "GREEN" if a.energy_alert == "NORMAL" else ("YELLOW" if a.energy_alert == "LOW" else "RED")
        icon = "🟢" if a.energy_alert == "NORMAL" else ("🟡" if a.energy_alert == "LOW" else "🔴")
        return col(f"{icon} Energy: {a.energy_alert}", c)
    color_row([energy_color(a) for a in agents])

    # Row 6: Risk Breakdown
    row([f"⚡ Risk: {a.composite_risk:.2f} (S:{a.risk_stability:.1f} N:{a.risk_navigation:.1f} E:{a.risk_energy:.1f})" for a in agents])

    # Row 7: NFZ Status (colored)
    def nfz_color(a: SwarmAgentState) -> str:
        if a.nfz_violation:
            return col(f"⛔ NFZ: {a.nfz_nearest[:12]} {a.nfz_dist:+.0f}m", "RED")
        elif a.nfz_dist < 20.0:
            return col(f"⚠️ NFZ: {a.nfz_nearest[:12]} {a.nfz_dist:+.0f}m", "YELLOW")
        return col(f"✅ NFZ: CLEAR ({a.nfz_dist:+.0f}m)", "GREEN")
    color_row([nfz_color(a) for a in agents])

    # Row 8: Agent Decision (colored)
    def dec_color(a: SwarmAgentState) -> str:
        if a.agent_proposal in ("NOMINAL_CONTINUE", "LANDED_SAFE"):
            c = "GREEN"
        elif a.agent_proposal in ("RETURN_TO_HOME", "ALTITUDE_HOLD_HOVER", "NFZ_REROUTE"):
            c = "YELLOW"
        else:
            c = "RED"
        return col(f"🧠 {a.agent_proposal[:19]}", c)
    color_row([dec_color(a) for a in agents])

    # Row 9: L1 Guardrail
    row([f"L1: {a.l1_status} {a.l1_notes[:15]}" for a in agents])

    # Row 10: CBF Safety Filter (L2)
    def cbf_color(a: SwarmAgentState) -> str:
        if a.l2_status == "INTERVENED":
            return col(f"🛡️ CBF: INTERVENED", "YELLOW")
        return col(f"🛡️ CBF: {a.l2_notes[:16]}", "GREEN")
    color_row([cbf_color(a) for a in agents])

    # Row 11: Final Action (colored)
    def act_color(a: SwarmAgentState) -> str:
        if a.final_action == "LANDED_SAFE":
            return col("🏁 LANDED_SAFE", "MAGENTA")
        c = "GREEN" if a.final_action in ("NOMINAL_CONTINUE", "NFZ_REROUTE") else ("YELLOW" if a.final_action in ("RETURN_TO_HOME", "ALTITUDE_HOLD_HOVER") else "RED")
        return col(f"▶ {a.final_action[:18]}", c)
    color_row([act_color(a) for a in agents])

    # Row 12: Delegation Info
    def deleg_info(a: SwarmAgentState) -> str:
        if a.delegation_status == "DELEGATED":
            return f"🤝 → {a.delegation_target}"
        elif a.delegation_status == "NO_PEER_AVAILABLE":
            return "❌ No peer avail"
        return "— N/A"
    row([deleg_info(a) for a in agents])

    # Row 13: Active Task
    row([f"📋 {a.active_task[:22]}" for a in agents])

    # Row 14: Target Waypoint
    row([f"🎯 Target: WP-{a.current_wp_idx+1}" if a.route_coords else "🎯 Target: —" for a in agents])

    # Row 15: Tasks Completed / Total
    def task_str(a: SwarmAgentState) -> str:
        tot = max(1, a.total_tasks)
        pct = int((a.tasks_completed / tot) * 100)
        return f"📊 Tasks: {a.tasks_completed}/{a.total_tasks} ({pct}%)"
    row([task_str(a) for a in agents])

    # Row 16: Mesh Link Quality to Master
    def link_badge(a: SwarmAgentState) -> str:
        if a.is_leader:
            return col("📶 Mesh: LOCAL MASTER (0ms)", "GREEN")
        elif a.on_ground:
            return col("📶 Mesh: DISCONNECTED", "RED")
        c = "GREEN" if a.link_quality_pct > 75 else ("YELLOW" if a.link_quality_pct > 40 else "RED")
        return col(f"📶 Mesh: {a.link_quality_pct:.0f}% ({a.comms_latency_ms:.1f}ms)", c)
    color_row([link_badge(a) for a in agents])

    # Row 17: Reasoning Trace (truncated)
    print(col("╟" + "─" * (W - 2) + "╢", "CYAN"))
    print(col("║", "CYAN") + col("  🧠 Agent Reasoning Traces:".ljust(W-2), "MAGENTA") + col("║", "CYAN"))
    for a in agents:
        trace_line = f"    [{a.uav_id}] {a.reasoning_trace[:W-14]}"
        c = "RED" if any(w in a.reasoning_trace.lower() for w in ("critical", "fail", "nfz", "⛔")) else ("YELLOW" if any(w in a.reasoning_trace.lower() for w in ("low", "degraded", "hover", "reroute")) else "GREEN")
        print(col("║", "CYAN") + col(f"{trace_line:<{W-2}}", c) + col("║", "CYAN"))

    # NFZ Zone Legend
    if _SWARM_BUS.nfz_zones:
        print(col("╟" + "─" * (W - 2) + "╢", "CYAN"))
        nfz_legend = "  🚫 NFZ Zones: " + " | ".join([f"{z['name']} ({z['x']:.0f},{z['y']:.0f}) r={z['radius']:.0f}m" for z in _SWARM_BUS.nfz_zones])
        print(col("║", "CYAN") + col(f"{nfz_legend:<{W-2}}", "RED") + col("║", "CYAN"))

    # SwarmBus Event Log Box
    print(col("╠" + "═" * (W - 2) + "╣", "CYAN"))
    print(col("║", "CYAN") + col("  📡 Inter-Agent SwarmBus Decentralized Event & Handoff Log:".ljust(W-2), "YELLOW") + col("║", "CYAN"))
    print(col("╟" + "─" * (W - 2) + "╢", "CYAN"))
    with _SWARM_BUS.lock:
        if not _SWARM_BUS.event_log:
            print(col("║", "CYAN") + "    (Swarm nominal — exchanging periodic heartbeats on UDP 14550)".ljust(W-2) + col("║", "CYAN"))
        else:
            for ev in list(_SWARM_BUS.event_log)[-5:]:
                print(col("║", "CYAN") + f"    {ev:<{W-6}}" + col("║", "CYAN"))

    # ── Dedicated Master UAV System & Mission Telemetry Report ──
    print(col("╠" + "═" * (W - 2) + "╣", "CYAN"))
    print(col("║", "CYAN") + col("  👑 MASTER UAV SYSTEM & MISSION TELEMETRY REPORT".ljust(W-2), "YELLOW") + col("║", "CYAN"))
    print(col("╟" + "─" * (W - 2) + "╢", "CYAN"))

    tot_tasks = sum(a.total_tasks for a in agents)
    done_tasks = sum(a.tasks_completed for a in agents)
    progress_pct = (done_tasks / max(1, tot_tasks)) * 100.0

    master_summary = f"    Master: {lead_name}  │  Swarm Progress: {mini_bar(progress_pct, 12)} {progress_pct:5.1f}% ({done_tasks}/{tot_tasks} Tasks)  │  Mesh Links: {active_count}/{N} Connected"
    print(col("║", "CYAN") + f"{master_summary:<{W+8}}" + col("║", "CYAN"))

    mesh_links = []
    for a in agents:
        if a.uav_id != lead_name:
            st = f"{a.link_quality_pct:.0f}% ({a.comms_latency_ms:.1f}ms)" if not a.on_ground else "OFFLINE"
            mesh_links.append(f"{lead_name}↔{a.uav_id}: {st}")
    if mesh_links:
        mesh_line = "    Mesh Links: " + "  │  ".join(mesh_links)
        print(col("║", "CYAN") + f"{mesh_line:<{W-2}}" + col("║", "CYAN"))

    report_notice = "    📄 Live Master System Telemetry Exported → /home/samiksha/uav_fdi_data/master_system_report.json"
    print(col("║", "CYAN") + col(f"{report_notice:<{W-2}}", "WHITE") + col("║", "CYAN"))

    print(col("╚" + "═" * (W - 2) + "╝", "CYAN"))
    print(col(f"  [Agents: {N} | Terminal 4 Commands: 'fault UAV-2 GPS', 'clear UAV-2', 'clear all', 'batt UAV-2 14' | Ctrl+C]", "BOLD"))


# ═══════════════════════════════════════════════════════════════════════════
# Main Swarm Execution Loop
# ═══════════════════════════════════════════════════════════════════════════
async def run_swarm(
    ticks: int = 0,
    inject_uav: str = "UAV-1",
    inject_batt: Optional[float] = None,
    inject_fault: Optional[str] = None,
    inject_tick: int = 1,
    add_uav_specs: Optional[List[str]] = None,
    remove_uav_specs: Optional[List[str]] = None,
    initial_uavs: int = 4,
):
    _SWARM_BUS.init_ros()
    _SWARM_BUS.init_mavlink()

    # Clean, verified flight corridors — ALL waypoints strictly outside No-Fly Zones
    # NFZ-Alpha (Hospital) center: (50.0, 60.0, r=18m) — UAV-2 flies perimeter corridor with >10m clearance
    # NFZ-Bravo (Airport) center: (-15.0, 90.0, r=22m) — UAV-3 flies perimeter corridor with >15m clearance
    uav1_route = [(0.0, 0.0, 12.0), (30.0, 15.0, 12.0), (70.0, 15.0, 12.0), (85.0, -10.0, 12.0), (45.0, -20.0, 12.0), (0.0, 0.0, 12.0)]
    uav2_route = [(20.0, 10.0, 14.0), (50.0, 25.0, 14.0), (85.0, 60.0, 14.0), (50.0, 95.0, 14.0), (15.0, 60.0, 14.0), (20.0, 10.0, 14.0)]
    uav3_route = [(-20.0, 30.0, 11.0), (-50.0, 65.0, 11.0), (-50.0, 115.0, 11.0), (20.0, 115.0, 11.0), (20.0, 65.0, 11.0), (-20.0, 30.0, 11.0)]
    uav4_route = [(10.0, -15.0, 10.0), (35.0, -25.0, 10.0), (50.0, -10.0, 10.0), (25.0, 5.0, 10.0), (10.0, -15.0, 10.0)]

    agents = [
        SwarmAgentState(
            uav_id="UAV-1", role="👑 Swarm Leader", is_leader=True, pos_x=0.0, pos_y=0.0, altitude=12.0,
            vx=1.5, vy=1.0, battery_pct=95.0, active_task="Survey Sector Alpha",
            route_coords=uav1_route, total_tasks=len(uav1_route),
            waypoints_queue=[f"WP-1.{i+1}" for i in range(len(uav1_route))]
        ),
        SwarmAgentState(
            uav_id="UAV-2", role="Relay / Escort", pos_x=20.0, pos_y=10.0, altitude=14.0,
            vx=1.2, vy=1.1, battery_pct=92.0, active_task="Comms Relay & Recon",
            route_coords=uav2_route, total_tasks=len(uav2_route),
            waypoints_queue=[f"WP-2.{i+1}" for i in range(len(uav2_route))]
        ),
        SwarmAgentState(
            uav_id="UAV-3", role="Survey Bravo", pos_x=-20.0, pos_y=30.0, altitude=11.0,
            vx=-0.8, vy=1.3, battery_pct=88.0, active_task="Survey Sector Bravo",
            route_coords=uav3_route, total_tasks=len(uav3_route),
            waypoints_queue=[f"WP-3.{i+1}" for i in range(len(uav3_route))]
        ),
        SwarmAgentState(
            uav_id="UAV-4", role="Swarm Reserve", pos_x=10.0, pos_y=-15.0, altitude=10.0,
            vx=1.3, vy=-0.5, battery_pct=94.0, active_task="Perimeter Patrol",
            route_coords=uav4_route, total_tasks=len(uav4_route),
            waypoints_queue=[f"WP-4.{i+1}" for i in range(len(uav4_route))]
        ),
    ]

    # Support initial horizontal scaling beyond 4 agents
    if initial_uavs > 4:
        for idx in range(5, initial_uavs + 1):
            uid = f"UAV-{idx}"
            scale_route = [
                (15.0 * (idx - 3), -12.0 * (idx - 3), 12.5),
                (15.0 * (idx - 3) + 25.0, -12.0 * (idx - 3) + 15.0, 12.5),
                (15.0 * (idx - 3) + 10.0, -12.0 * (idx - 3) + 30.0, 12.5),
                (15.0 * (idx - 3), -12.0 * (idx - 3), 12.5),
            ]
            agents.append(
                SwarmAgentState(
                    uav_id=uid,
                    role=f"Scale Sector {idx}",
                    pos_x=15.0 * (idx - 3),
                    pos_y=-12.0 * (idx - 3),
                    altitude=12.5,
                    vx=0.8, vy=-0.5,
                    battery_pct=94.0,
                    active_task=f"Sector {chr(64 + idx)}",
                    route_coords=scale_route,
                    total_tasks=len(scale_route),
                    waypoints_queue=[f"WP-{idx}A", f"WP-{idx}B"]
                )
            )

    # Register all agents in the decentralized swarm bus
    for a in agents:
        _SWARM_BUS.register_uav(a.uav_id)

    graphs = {a.uav_id: build_uav_graph() for a in agents}
    current_tick = 0

    print(f"[Swarm] Initializing {len(agents)} autonomous LangGraph UAV agents...")
    _SWARM_BUS.ensure_gazebo_swarm_models()
    await asyncio.sleep(0.5)

    while True:
        current_tick += 1
        if current_tick % 25 == 1:
            _SWARM_BUS.ensure_gazebo_swarm_models()

        # ── 1. Live Interactive Commands from Terminal Stdin or /tmp/swarm_cmd ──
        pending_commands = []
        try:
            import select
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                raw = sys.stdin.readline().strip()
                if raw:
                    pending_commands.append(raw)
        except Exception:
            pass

        for cmd_path in ("/tmp/swarm_cmd", os.path.join(os.path.dirname(__file__), "swarm_cmd.txt")):
            if os.path.exists(cmd_path):
                try:
                    with open(cmd_path, "r") as f:
                        content = f.read().strip()
                    os.remove(cmd_path)
                    if content:
                        for line in content.splitlines():
                            if line.strip():
                                pending_commands.append(line.strip())
                except Exception:
                    pass

        def _norm_id(tok: str) -> str:
            t = tok.upper().strip()
            if t.isdigit(): return f"UAV-{t}"
            if t.startswith("UAV") and not t.startswith("UAV-"): return f"UAV-{t[3:]}"
            return t

        for raw_cmd in pending_commands:
            try:
                tokens = raw_cmd.split()
                if not tokens: continue

                # Syntax normalizer: if command is 'UAV-2 clear' or '2 clear' or 'UAV-2 batt 14'
                if (tokens[0].isdigit() or tokens[0].lower().startswith("uav")) and len(tokens) >= 2:
                    sub = tokens[1].lower()
                    if sub in ("clear", "ok", "reset"):
                        tokens = ["clear", tokens[0]]
                    elif sub in ("batt", "battery") and len(tokens) >= 3:
                        tokens = ["batt", tokens[0], tokens[2]]
                    elif sub in ("gps", "gyro", "gyroscope", "accel", "accelerometer", "baro", "barometer", "compass", "mag"):
                        tokens = ["fault", tokens[0], tokens[1]]

                verb = tokens[0].lower()

                if verb in ("takeoff", "arm"):
                    for a in agents:
                        a.on_ground = False
                        a.altitude = max(a.altitude, 12.0)
                        a.final_action = "NOMINAL_CONTINUE"
                        a.agent_proposal = "NOMINAL_CONTINUE"
                    _SWARM_BUS.log_event("🚀 Swarm commanded to Takeoff to operational altitude (12m)")

                elif verb in ("land", "disarm"):
                    for a in agents:
                        a.final_action = "FAILSAFE_LAND"
                    _SWARM_BUS.log_event("🛬 Swarm commanded to Land safely")

                elif verb in ("rtl", "home", "return"):
                    target_id = _norm_id(tokens[1]) if len(tokens) >= 2 and tokens[1].lower() not in ("all", "*", "swarm") else "ALL"
                    for a in agents:
                        if target_id == "ALL" or a.uav_id == target_id:
                            a.final_action = "RETURN_TO_HOME"
                            a.agent_proposal = "RETURN_TO_HOME"
                            a.active_task = "Return To Home (RTL)"
                            a.role = "RTL Inbound"
                            a.waypoints_queue.clear()
                    _SWARM_BUS.log_event(f"🏠 [RTL] Return To Home commanded for {target_id} -> Vectoring back to Home Base (0,0)")

                elif verb in ("add", "+") and len(tokens) >= 2:
                    for new_tok in tokens[1:]:
                        new_id = _norm_id(new_tok)
                        if not any(a.uav_id == new_id for a in agents):
                            idx = len(agents) + 1
                            scale_route = [
                                (10.0 * (idx - 2), -15.0 * (idx - 2), 13.0),
                                (10.0 * (idx - 2) + 20.0, -15.0 * (idx - 2) + 20.0, 13.0),
                                (10.0 * (idx - 2), -15.0 * (idx - 2), 13.0)
                            ]
                            new_uav = SwarmAgentState(
                                uav_id=new_id,
                                role=f"Dynamic Join {idx}",
                                pos_x=10.0 * (idx - 2),
                                pos_y=-15.0 * (idx - 2),
                                altitude=13.0,
                                vx=0.8, vy=-0.6,
                                battery_pct=95.0,
                                active_task="Dynamic Survey",
                                route_coords=scale_route,
                                total_tasks=len(scale_route),
                                waypoints_queue=[f"WP-D{idx}-1", f"WP-D{idx}-2"]
                            )
                            _SWARM_BUS.register_uav(new_id)
                            graphs[new_id] = build_uav_graph()
                            agents.append(new_uav)
                            _SWARM_BUS.log_event(f"➕ [HORIZONTAL SCALING] {new_id} added via live operator command (Swarm size: {len(agents)})")

                elif verb in ("remove", "-", "rm") and len(tokens) >= 2:
                    for rem_tok in tokens[1:]:
                        rem_id = _norm_id(rem_tok)
                        target = next((a for a in agents if a.uav_id == rem_id), None)
                        if target:
                            if target.waypoints_queue:
                                _SWARM_BUS.request_handoff(rem_id, list(target.waypoints_queue), "Operator Removed / Recalled")
                            agents.remove(target)
                            graphs.pop(rem_id, None)
                            with _SWARM_BUS.lock:
                                _SWARM_BUS.peer_states.pop(rem_id, None)
                                _SWARM_BUS.active_faults.pop(rem_id, None)
                            _SWARM_BUS.log_event(f"➖ [HORIZONTAL SCALING] {rem_id} removed via live operator command (Remaining: {len(agents)})")

                elif verb in ("fault", "inject", "f"):
                    # formats: fault 1 gps, fault UAV-2 GPS, or fault gps 2
                    target_id = "UAV-1"
                    sensor_name = "GPS Failure"
                    for tok in tokens[1:]:
                        tok_low = tok.lower()
                        if tok.isdigit() or tok_low.startswith("uav"):
                            target_id = _norm_id(tok)
                        elif tok_low in ("gps", "nav"):
                            sensor_name = "GPS Failure"
                        elif tok_low in ("gyro", "gyroscope"):
                            sensor_name = "Gyroscope Failure"
                        elif tok_low in ("accel", "accelerometer"):
                            sensor_name = "Accelerometer Failure"
                        elif tok_low in ("baro", "barometer"):
                            sensor_name = "Barometer Failure"
                        elif tok_low in ("compass", "mag"):
                            sensor_name = "Compass Failure"
                    with _SWARM_BUS.lock:
                        _SWARM_BUS.active_faults[target_id] = sensor_name
                    target = next((a for a in agents if a.uav_id == target_id), None)
                    if target:
                        target.fault_label = sensor_name
                        if sensor_name == "GPS Failure":
                            target.gps_valid = False
                    _SWARM_BUS.log_event(f"🔴 Live injected {sensor_name} on {target_id}")

                elif verb in ("clear", "ok", "reset"):
                    # formats: clear 1, clear UAV-2, or just 'clear' / 'clear all'
                    is_all = (len(tokens) == 1) or (len(tokens) >= 2 and tokens[1].lower() in ("all", "*", "swarm"))
                    if not is_all:
                        target_id = _norm_id(tokens[1])
                        with _SWARM_BUS.lock:
                            _SWARM_BUS.active_faults.pop(target_id, None)
                        target = next((a for a in agents if a.uav_id == target_id), None)
                        if target:
                            target.fault_label = "No Failure"
                            target.gps_valid = True
                            target.battery_pct = max(target.battery_pct, 92.0)
                            target.energy_alert = "NORMAL"
                            target.on_ground = False
                            target.altitude = max(target.altitude, 12.0)
                            target.final_action = "NOMINAL_CONTINUE"
                            target.agent_proposal = "NOMINAL_CONTINUE"
                            target.available = True
                            target.delegation_status = "N/A"
                            target.delegation_target = ""
                            target.beacon_active = False
                            target.hover_ticks = 0
                            if not target.waypoints_queue and target.route_coords:
                                target.waypoints_queue = [f"WP-{target.uav_id}.{i+1}" for i in range(len(target.route_coords))]
                        _SWARM_BUS.log_event(f"🔧 Restored {target_id} to 100% nominal (Battery: 92%, Faults Cleared, Resumed Route)")
                    else:
                        with _SWARM_BUS.lock:
                            _SWARM_BUS.active_faults.clear()
                        for a in agents:
                            a.fault_label = "No Failure"
                            a.gps_valid = True
                            a.battery_pct = max(a.battery_pct, 92.0)
                            a.energy_alert = "NORMAL"
                            a.on_ground = False
                            a.altitude = max(a.altitude, 12.0)
                            a.final_action = "NOMINAL_CONTINUE"
                            a.agent_proposal = "NOMINAL_CONTINUE"
                            a.available = True
                            a.delegation_status = "N/A"
                            a.delegation_target = ""
                            a.beacon_active = False
                            a.hover_ticks = 0
                            if not a.waypoints_queue and a.route_coords:
                                a.waypoints_queue = [f"WP-{a.uav_id}.{i+1}" for i in range(len(a.route_coords))]
                        _SWARM_BUS.log_event("🔧 Restored ALL UAVs to nominal health & 92% battery (All Missions Resumed)")

                elif verb in ("batt", "battery", "b") and len(tokens) >= 3:
                    target_id = _norm_id(tokens[1])
                    b_val = float(tokens[2])
                    target = next((a for a in agents if a.uav_id == target_id), None)
                    if target:
                        target.battery_pct = b_val
                        _SWARM_BUS.log_event(f"🔋 Set {target_id} battery to {b_val:.0f}%")

                elif verb in ("idle", "loiter") and len(tokens) >= 2:
                    target_id = _norm_id(tokens[1])
                    target = next((a for a in agents if a.uav_id == target_id), None)
                    if target:
                        target.role = "Swarm Reserve"
                        target.active_task = "Reserve Loiter"
                        target.waypoints_queue.clear()
                        target.final_action = "STANDBY_LOITER"
                        _SWARM_BUS.log_event(f"💤 {target_id} placed in standby reserve loiter")

                elif verb == "add" and len(tokens) >= 2:
                    for tok in tokens[1:]:
                        new_id = _norm_id(tok)
                        if not any(a.uav_id == new_id for a in agents):
                            idx = len(agents) + 1
                            scale_route = [
                                (10.0 * (idx - 1), -10.0 * (idx - 1), 12.0),
                                (20.0 * (idx - 1), -5.0 * (idx - 1), 14.0),
                                (10.0 * (idx - 1), -10.0 * (idx - 1), 12.0),
                            ]
                            new_uav = SwarmAgentState(
                                uav_id=new_id,
                                role=f"Escort & Survey",
                                pos_x=10.0 * (idx - 2),
                                pos_y=-15.0 * (idx - 2),
                                altitude=13.0,
                                vx=0.4, vy=-0.3,
                                battery_pct=95.0,
                                active_task="Dynamic Enroute",
                                route_coords=scale_route,
                                total_tasks=len(scale_route),
                                waypoints_queue=[f"WP-{idx}A", f"WP-{idx}B"]
                            )
                            _SWARM_BUS.register_uav(new_id)
                            graphs[new_id] = build_uav_graph()
                            agents.append(new_uav)
                            _SWARM_BUS.log_event(f"➕ [HORIZONTAL SCALING] {new_id} dynamically added to swarm (Total: {len(agents)} UAVs)")

                elif verb in ("remove", "delete", "rm") and len(tokens) >= 2:
                    for tok in tokens[1:]:
                        rem_id = _norm_id(tok)
                        target = next((a for a in agents if a.uav_id == rem_id), None)
                        if target and len(agents) > 1:
                            target.on_ground = True
                            target.final_action = "FAILSAFE_LAND"
                            target.role = "Decommissioned"
                            target.available = False
                            _SWARM_BUS.log_event(f"➖ [HORIZONTAL SCALING] {rem_id} decommissioned & landed safely")
            except Exception as e:
                _SWARM_BUS.log_event(f"Command error: {e}")

        # ── 2. Dynamic Horizontal Scaling: Add one or more agents at scheduled ticks ──
        if add_uav_specs:
            for spec in list(add_uav_specs):
                parts = spec.split(":")
                new_id = parts[0].strip().upper()
                at_t = int(parts[1].strip()) if len(parts) > 1 else 1
                if current_tick == at_t and not any(a.uav_id == new_id for a in agents):
                    idx = len(agents) + 1
                    new_uav = SwarmAgentState(
                        uav_id=new_id,
                        role=f"Dynamic Join {idx}",
                        pos_x=10.0 * (idx - 2),
                        pos_y=-15.0 * (idx - 2),
                        altitude=13.0,
                        vx=0.4, vy=-0.3,
                        battery_pct=95.0,
                        active_task="Dynamic Survey",
                        waypoints_queue=[f"WP-D{idx}-1", f"WP-D{idx}-2"]
                    )
                    _SWARM_BUS.register_uav(new_id)
                    graphs[new_id] = build_uav_graph()
                    agents.append(new_uav)
                    _SWARM_BUS.log_event(f"➕ [HORIZONTAL SCALING] {new_id} joined active swarm mesh seamlessly (Swarm size: {len(agents)})")

        # ── 3. Dynamic Removal: Safely decommission/recall agents at scheduled ticks ──
        if remove_uav_specs:
            for spec in list(remove_uav_specs):
                parts = spec.split(":")
                rem_id = parts[0].strip().upper()
                at_t = int(parts[1].strip()) if len(parts) > 1 else 1
                if current_tick == at_t:
                    target = next((a for a in agents if a.uav_id == rem_id), None)
                    if target:
                        if target.waypoints_queue:
                            _SWARM_BUS.request_handoff(rem_id, list(target.waypoints_queue), "Scheduled Removal / Recalled")
                        agents.remove(target)
                        graphs.pop(rem_id, None)
                        with _SWARM_BUS.lock:
                            _SWARM_BUS.peer_states.pop(rem_id, None)
                            _SWARM_BUS.active_faults.pop(rem_id, None)
                        _SWARM_BUS.log_event(f"➖ [HORIZONTAL SCALING] {rem_id} safely decommissioned at tick {current_tick} (Remaining: {len(agents)})")

        # ── 4. CLI Fault Injection ──
        if current_tick >= inject_tick:
            for a in agents:
                if a.uav_id == inject_uav:
                    if inject_batt is not None:
                        a.battery_pct = float(inject_batt)
                    if inject_fault:
                        a.fault_label = inject_fault
                        if inject_fault == "GPS Failure":
                            a.gps_valid = False

        # ── 5. Poll live PX4 telemetry ──
        px4_data = _SWARM_BUS.poll_px4_telemetry()
        if px4_data:
            for a in agents:
                if a.uav_id == "UAV-1":
                    if "pos_x" in px4_data:
                        a.pos_x = float(px4_data["pos_x"])
                    if "pos_y" in px4_data:
                        a.pos_y = float(px4_data["pos_y"])
                    if "pos_z" in px4_data:
                        a.altitude = float(px4_data["pos_z"])
                    if "vx" in px4_data:
                        a.vx = float(px4_data["vx"])
                    if "vy" in px4_data:
                        a.vy = float(px4_data["vy"])
                    if "vz" in px4_data:
                        a.vz = float(px4_data["vz"])
                    if "battery_pct" in px4_data:
                        a.battery_pct = float(px4_data["battery_pct"])

        # ── 6. Check live fault injection from QGroundControl, Terminal, or PX4 ──
        with _SWARM_BUS.lock:
            for a in agents:
                if a.uav_id in _SWARM_BUS.active_faults:
                    qgc_fault = _SWARM_BUS.active_faults[a.uav_id]
                    a.fault_label = qgc_fault
                    if qgc_fault == "GPS Failure":
                        a.gps_valid = False
                    else:
                        a.gps_valid = True
                elif not inject_fault or a.uav_id != inject_uav or current_tick < inject_tick:
                    if a.fault_label != "No Failure":
                        a.fault_label = "No Failure"
                        a.gps_valid = True

        # ── 7. Execute each UAV's LangGraph pipeline ──
        for i, a in enumerate(agents):
            res_dict = graphs[a.uav_id].invoke(a)
            # Update state with LangGraph outputs
            for k, v in res_dict.items():
                if hasattr(a, k):
                    setattr(a, k, v)

            # If task was delegated to a peer, transfer the actual route to the peer
            if a.delegation_status == "DELEGATED" and a.delegation_target:
                for peer in agents:
                    if peer.uav_id == a.delegation_target and "from" not in peer.active_task:
                        # Pick up the stored route from SwarmBus
                        handoff_data = None
                        with _SWARM_BUS.lock:
                            handoff_routes = getattr(_SWARM_BUS, '_pending_handoff_routes', {})
                            handoff_data = handoff_routes.pop(peer.uav_id, None)

                        if handoff_data and handoff_data.get("remaining_route"):
                            delegated_wps = handoff_data["remaining_route"]
                            from_uav = handoff_data["from"]
                            # Append delegated route to peer's existing route
                            # Insert after current waypoint so peer finishes current target first
                            insert_pos = peer.current_wp_idx + 1
                            for idx_d, wp_d in enumerate(delegated_wps):
                                peer.route_coords.insert(insert_pos + idx_d, wp_d)
                            peer.total_tasks += len(delegated_wps)
                            # Add waypoint names
                            for idx_d in range(len(delegated_wps)):
                                peer.waypoints_queue.append(f"WP-{from_uav[-1]}.D{idx_d+1}")
                            _SWARM_BUS.log_event(
                                f"📦 [{peer.uav_id}] Received {len(delegated_wps)} waypoints from {from_uav} — appended to route"
                            )

                        peer.active_task = f"Sector A (from {a.uav_id})"
                        peer.role = "Task Reallocated"
                        peer.available = False

        # ── 7.5. Immediate Master Report on Task Completion ──
        with _SWARM_BUS.lock:
            pending_completions = list(_SWARM_BUS._pending_task_completions)
            _SWARM_BUS._pending_task_completions.clear()

        if pending_completions:
            lead_id        = getattr(_SWARM_BUS, "current_leader", "UAV-1") or "UAV-1"
            tot_tasks_all  = sum(a.total_tasks    for a in agents)
            done_tasks_all = sum(a.tasks_completed for a in agents)
            swarm_pct      = round((done_tasks_all / max(1, tot_tasks_all)) * 100.0, 1)
            online_cnt     = sum(1 for a in agents if not a.on_ground)
            fault_cnt      = sum(1 for a in agents if a.fault_label != "No Failure")
            nfz_cnt        = sum(1 for a in agents if a.nfz_violation)
            sys_status     = "NOMINAL" if fault_cnt == 0 and nfz_cnt == 0 else "DEGRADED"
            ts_iso         = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            for evt in pending_completions:
                uid        = evt["uav_id"]
                wp_idx     = evt["wp_idx"]
                tasks_done = evt["tasks_done"]
                total_t    = evt["total_tasks"]
                agent_pct  = round((tasks_done / max(1, total_t)) * 100.0, 1)

                # Find completing agent for live telemetry
                ag       = next((a for a in agents if a.uav_id == uid), None)
                ag_pos   = f"({ag.pos_x:.1f}, {ag.pos_y:.1f}, {ag.altitude:.1f}m)" if ag else "N/A"
                ag_batt  = f"{ag.battery_pct:.1f}%" if ag else "N/A"
                ag_fault = ag.fault_label if ag else "No Failure"
                ag_action= ag.final_action if ag else "N/A"
                ag_nfz   = f"VIOLATION — {ag.nfz_nearest} ({ag.nfz_dist:.1f}m)" if (ag and ag.nfz_violation) else "Clear"
                ag_link  = f"{ag.link_quality_pct:.0f}% quality / {ag.comms_latency_ms:.1f}ms latency" if ag else "N/A"

                # Progress bar (20 chars wide)
                pbar_fill = int(swarm_pct / 100 * 20)
                pbar      = "█" * pbar_fill + "░" * (20 - pbar_fill)

                W = 70  # inner content width
                def _r(lbl, val):
                    row = f"  {lbl:<28}{val}"
                    return f"║ {row:<{W}} ║"
                def _div():
                    return f"╠{'═'*(W+2)}╣"
                def _sep():
                    return f"║ {'·'*W} ║"
                hdr = f"╔{'═'*(W+2)}╗"
                ftr = f"╚{'═'*(W+2)}╝"
                blank = f"║ {' '*W} ║"

                lines = [
                    "",
                    hdr,
                    f"║  {'👑  MASTER UAV — OPERATIONAL TASK REPORT':<{W}} ║",
                    f"║  {'Reported by: ' + lead_id + '   ·   System: ' + sys_status + '   ·   ' + ts_iso:<{W}} ║",
                    f"║  {'Simulation Tick: ' + str(current_tick) + '   ·   Backend: ' + ('PX4_SITL+GAZEBO' if _SWARM_BUS.is_connected_to_px4() else 'STANDALONE_SIM'):<{W}} ║",
                    _div(),
                    f"║  {'📍  COMPLETING AGENT — LIVE TELEMETRY':<{W}} ║",
                    _sep(),
                    _r("Agent ID:", uid),
                    _r("Waypoint Completed:", f"WP-{wp_idx}"),
                    _r("Agent Task Progress:", f"{tasks_done} / {total_t}  ({agent_pct}%)"),
                    _r("Current Position (X,Y,Z):", ag_pos),
                    _r("Battery Level:", ag_batt),
                    _r("Fault / Health Status:", ag_fault),
                    _r("Current Flight Action:", ag_action),
                    _r("NFZ Geofence Status:", ag_nfz),
                    _r("Mesh Link Quality:", ag_link),
                    _div(),
                    f"║  {'📊  SWARM MISSION SUMMARY':<{W}} ║",
                    _sep(),
                    _r("Swarm Leader:", lead_id),
                    _r("UAVs Online / Total:", f"{online_cnt} / {len(agents)}"),
                    _r("Active Fault Count:", f"{fault_cnt} UAV(s) with sensor fault"),
                    _r("NFZ Violations:", f"{nfz_cnt} active violation(s)"),
                    _r("Tasks Completed / Total:", f"{done_tasks_all} / {tot_tasks_all}"),
                    _r("Overall Progress:", f"[{pbar}] {swarm_pct}%"),
                    _div(),
                    f"║  {'📋  PER-AGENT STATUS TABLE':<{W}} ║",
                    _sep(),
                ]
                for a in agents:
                    icon   = "👑" if a.uav_id == lead_id else ("⚠" if a.fault_label != "No Failure" else "✈")
                    marker = " ◄ COMPLETED NOW" if a.uav_id == uid else ""
                    bfill  = int(a.battery_pct / 100 * 8)
                    bbr    = "█" * bfill + "░" * (8 - bfill)
                    row = (
                        f"{icon} {a.uav_id:<6}  "
                        f"WP {a.current_wp_idx+1:>2}/{len(a.route_coords):<2}  "
                        f"Tasks {a.tasks_completed}/{a.total_tasks}  "
                        f"[{bbr}]{a.battery_pct:5.1f}%  "
                        f"Alt {a.altitude:5.1f}m  "
                        f"{a.fault_label:<20}{marker}"
                    )
                    lines.append(f"║   {row:<{W-1}} ║")

                lines += [blank, ftr, ""]
                print("\033[93m" + "\n".join(lines) + "\033[0m")

                # ── Immediate JSON report export with embedded task event ──
                export_master_system_report(agents, current_tick,
                    task_event={
                        "event_type": "TASK_COMPLETION",
                        "completing_agent": uid,
                        "waypoint_completed": wp_idx,
                        "agent_tasks_done": tasks_done,
                        "agent_total_tasks": total_t,
                        "agent_progress_pct": agent_pct,
                        "swarm_progress_pct": swarm_pct,
                        "timestamp_iso": ts_iso,
                        "reported_by_master": lead_id,
                    }
                )

                # ── QGC STATUSTEXT for this specific completion ──
                if _MAVLINK_AVAILABLE and lead_id in _SWARM_BUS._mav_conns:
                    mav_m   = _SWARM_BUS._mav_conns[lead_id]
                    qgc_txt = f"[MASTER] {uid} WP-{wp_idx} OK {tasks_done}/{total_t} | Swarm {swarm_pct}%"
                    try:
                        mav_m.mav.statustext_send(
                            mavutil.mavlink.MAV_SEVERITY_INFO,
                            qgc_txt.encode("utf-8")[:50],
                        )
                    except Exception:
                        pass

        # ── 8. Distributed Swarm Leader Re-Election ──
        _SWARM_BUS.evaluate_leader_election(agents)

        # ── 9. Render combined swarm dashboard ──
        render_swarm_dashboard(agents, current_tick)

        # ── 10. Sync 3D poses to Gazebo world so all quadrotors fly in Gazebo ──
        _SWARM_BUS.sync_gazebo_swarm(agents)

        # ── 11. Master System Report Export (written to disk every 2 ticks) ──
        if current_tick % 2 == 0 or current_tick == 1:
            export_master_system_report(agents, current_tick)

        # ── 12. Master UAV periodic STATUSTEXT update to QGroundControl ──
        lead_id = getattr(_SWARM_BUS, "current_leader", "UAV-1") or "UAV-1"
        if _MAVLINK_AVAILABLE and lead_id in _SWARM_BUS._mav_conns and current_tick % 16 == 0:
            mav_m = _SWARM_BUS._mav_conns[lead_id]
            tot_tasks = sum(a.total_tasks for a in agents)
            done_tasks = sum(a.tasks_completed for a in agents)
            p_pct = int((done_tasks / max(1, tot_tasks)) * 100)
            active_cnt = sum(1 for a in agents if not a.on_ground)
            msg_txt = f"👑 [MASTER] Progress: {p_pct}% ({done_tasks}/{tot_tasks} Tasks) | {active_cnt}/{len(agents)} Online"
            try:
                mav_m.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_NOTICE, msg_txt.encode("utf-8")[:50])
            except Exception:
                pass


        if ticks > 0 and current_tick >= ticks:
            print(f"\n[Done] Completed {ticks} swarm simulation ticks.")
            break

        await asyncio.sleep(0.25)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Autonomous UAV Swarm (LangGraph + ROS2)")
    parser.add_argument("--ticks", type=int, default=0, help="Number of ticks to run (0 = continuous)")
    parser.add_argument("--inject-uav", type=str, default="UAV-1", help="Target UAV for fault injection (UAV-1..4)")
    parser.add_argument("--inject-battery", type=float, default=None, help="Inject battery percentage (e.g. 14.0)")
    parser.add_argument("--inject-fault", type=str, default=None, help="Inject sensor fault (e.g. 'GPS Failure', 'Gyroscope Failure')")
    parser.add_argument("--inject-at-tick", type=int, default=3, help="Tick at which to trigger fault injection")
    parser.add_argument("--initial-uavs", type=int, default=4, help="Initial number of UAV agents in swarm (default 4)")
    parser.add_argument("--add-uav-at-tick", nargs="*", default=[], help="Dynamically add UAVs at tick, e.g. --add-uav-at-tick UAV-5:5 UAV-6:5")
    parser.add_argument("--remove-uav-at-tick", nargs="*", default=[], help="Dynamically remove UAVs at tick, e.g. --remove-uav-at-tick UAV-2:15")
    args = parser.parse_args()

    try:
        asyncio.run(run_swarm(
            ticks=args.ticks,
            inject_uav=args.inject_uav,
            inject_batt=args.inject_battery,
            inject_fault=args.inject_fault,
            inject_tick=args.inject_at_tick,
            add_uav_specs=args.add_uav_at_tick,
            remove_uav_specs=args.remove_uav_at_tick,
            initial_uavs=args.initial_uavs,
        ))
    except KeyboardInterrupt:
        print("\n[Swarm] Orchestrator terminated by user.")


if __name__ == "__main__":
    main()
