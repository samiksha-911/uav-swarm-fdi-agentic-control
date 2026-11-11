# SOURCE CODE AND COMPONENTS REFERENCE GUIDE
## Complete File Inventory, Code Anatomy & Architecture Directory

---

## 1. Directory Tree & File Inventory

```text
/home/samiksha/uav_fdi_data/
├── step80_multiagent_swarm.py       # Core Multi-Agent Swarm Orchestrator & LangGraph Engine
├── mission_control.py               # Unified Web Ground Control Station (Tornado REST + WS)
├── plot_results.py                  # Scientific Publication 300-DPI Evaluation Figure Generator
├── verify_pipeline.py               # Automated Unit & Integration Testing Suite
├── run_mission_control.sh           # One-Click Single-Command Web GCS Launcher
├── launch_all.sh                    # Multi-Terminal Diagnostic Launcher (4 Gnome-Terminal Tabs)
├── inject.sh                        # CLI Quick Action & Fault Injection Helper
├── flight_telemetry_log.csv         # Streaming Real-Time Telemetry CSV (Generated During Flight)
├── master_system_report.json        # Atomic JSON Snapshot of Swarm Health & Progress
├── swarm_cmd.txt                    # Asynchronous Swarm Command Exchange Buffer
├── uav_fdi_extratrees_model.joblib  # Pre-trained ExtraTrees Sensor FDI Classifier Artifact
├── variable_features.csv            # Engineered Sensor Feature Schema & Definitions
│
├── web_gui/
│   └── index.html                   # Single Page Application: Three.js 3D Theater + Radar UI
│
├── plots/
│   ├── swarm_3d_flight_trajectories.png # 3D Flight Corridor & NFZ Avoidance Visualization
│   ├── swarm_fdi_safety_timeline.png    # Temporal Telemetry & Safety Transition Timeline
│   └── swarm_mesh_performance.png       # Mesh Link Quality vs. Inter-Agent Distance Plot
│
└── fdi_env/                         # Python Virtual Environment (Python 3.10, PyTorch/Sklearn/Tornado)
```

---

## 2. Core Source Code Components

### 2.1. `step80_multiagent_swarm.py` — Swarm Orchestrator (~2,345 Lines)
The central simulation and execution kernel. Key classes and methods:

| Component / Function | Line Range | Architectural Purpose |
|---|---|---|
| `SwarmAgentState` (Dataclass) | Lines ~864–920 | Complete state vector per UAV: 3D position, velocity, battery, FDI label, risk, CBF action, waypoints. |
| `SwarmBus` (Singleton) | Lines ~110–860 | Decentralized shared memory bus: mesh topology, MAVLink connections, Gazebo sync, consensus log. |
| `evaluate_leader_election()` | Lines ~728–790 | Decentralized fitness scoring algorithm ($0.4B + 0.3(1-R) + 0.2G + 0.1S$) for master failover. |
| `ensure_gazebo_swarm_models()` | Lines ~795–830 | Automated discovery of Gazebo world and live multi-quadcopter model instantiation. |
| `sync_gazebo_swarm()` | Lines ~831–860 | Pushes 3D Cartesian poses of all swarm agents directly to Gazebo physics topics. |
| `node_collect_telemetry()` | Lines ~950–1075 | LangGraph Node 1: Updates kinematics, checks waypoints, updates distance to NFZ geofences. |
| `node_monitor_energy()` | Lines ~1076–1085 | LangGraph Node 2: Computes battery health alerts (`NORMAL`, `LOW`, `CRITICAL`). |
| `node_run_fdi()` | Lines ~1086–1092 | LangGraph Node 3: Executes ExtraTrees sensor classifier and isolates sensor anomalies. |
| `node_compute_risk()` | Lines ~1093–1113 | LangGraph Node 4: Formulates composite operational risk from stability, navigation, and energy. |
| `node_agentic_reasoning()` | Lines ~1114–1195 | LangGraph Node 5: Rule/LLM reasoning engine deciding action proposals (`NOMINAL`, `DELEGATE`, `FAILSAFE_LAND`). |
| `node_guardrail_l1()` & `l2()` | Lines ~1240–1320 | LangGraph Nodes 7–8: Control Barrier Function safety filter and tangential velocity projection. |
| `export_master_system_report()` | Lines ~1400–1520 | Atomically writes JSON telemetry snapshot and appends row to CSV flight log. |
| `run_swarm()` (Async Loop) | Lines ~1800–2300 | Main operational event loop: processes operator commands, ticks LangGraph, synchronizes display. |

### 2.2. `mission_control.py` — Web Mission Control Backend (~290 Lines)
Asynchronous web and telemetry server built with Tornado:

| Class / Handler | Route | Method | Description |
|---|---|---|---|
| `MainHandler` | `/` | GET | Serves `web_gui/index.html` Single Page Application. |
| `TelemetryAPIHandler` | `/api/telemetry` | GET | Returns swarm process liveness (`RUNNING`/`IDLE`) and latest JSON telemetry. |
| `StartMissionAPIHandler` | `/api/start` | POST | Launches `step80_multiagent_swarm.py` subprocess with fleet size and fault options. |
| `StopMissionAPIHandler` | `/api/stop` | POST | Sends `land` command to swarm and cleanly terminates the simulation subprocess. |
| `CommandAPIHandler` | `/api/command` | POST | Writes operator flight commands (`fault`, `batt`, `rtl`, `clear`, `add`) to `/tmp/swarm_cmd`. |
| `GeneratePlotsAPIHandler` | `/api/generate_plots` | POST | Executes `plot_results.py` in background and returns paths to 300-DPI PNGs. |
| `DownloadFileHandler` | `/api/download/(json\|csv)` | GET | Streams raw `master_system_report.json` or `flight_telemetry_log.csv` as download. |
| `TelemetryWebSocket` | `/ws/telemetry` | WS | Real-time bi-directional telemetry broadcast socket. |
| `broadcast_telemetry_tick()` | — | Callback | Native Tornado `PeriodicCallback` pushing telemetry to all clients every 250ms. |

### 2.3. `web_gui/index.html` — Interactive 3D Theater & Radar Dashboard (~1,370 Lines)
Client-side interface combining Three.js 3D WebGL and 2D Canvas:
- **3D Spatial Theater**:
  - `initThreeScene()`: Instantiates WebGL renderer, perspective camera, aerospace coordinate grid, dynamic lighting.
  - `homeGroup`: Prominent metallic Home Base Helipad at `(0, 0, 0)`, white/gold 'H' landing cross, 4 dedicated touchdown parking slots, 50m vertical sky light column, and animated expanding radar waves.
  - `create3DDroneMesh()`: 3D Quadcopter model with carbon fiber cross arms, 4 motor nacelles, rotating translucent propeller discs, high-intensity LED beacons, altitude drop-laser, and ground projection shadow.
  - `build3DRoutesAndWaypoints()`: 3D planned flight corridors at designated patrol altitudes with rotating octahedron crystals.
  - `rtlVectorGroup`: Dynamic 3D dashed amber return trajectories pointing from each drone straight to Home Base.
  - `OrbitControls` with presets: 3D Orbit, Focus Home, Top-Down, Track Master, Focus Fault.
- **2D Radar View**:
  - Top-down tactical radar display with range rings, NFZ boundaries, breadcrumb flight trails, and velocity arrows.
- **AI Decision Pipeline Banner**:
  - Real-time 5-stage badge updating dynamically: `1. Telemetry` $\to$ `2. FDI` $\to$ `3. Risk` $\to$ `4. CBF` $\to$ `5. Swarm Handoff`.

---

## 3. Utility Scripts and Launchers

1. **`run_mission_control.sh`**:
   - Checks port availability, starts `mission_control.py`, and automatically launches the default web browser to `http://localhost:8080` (or `8081`).
2. **`launch_all.sh`**:
   - Diagnostic script opening 4 organized Gnome-Terminal tabs:
     - Tab 1: QGroundControl AppImage.
     - Tab 2: Core Multi-Agent Swarm Orchestrator (`step80_multiagent_swarm.py`).
     - Tab 3: Interactive CLI Fault Injection Console.
     - Tab 4: Live CSV Telemetry Stream (`tail -f flight_telemetry_log.csv`).
3. **`plot_results.py`**:
   - Publication graphing engine using Matplotlib. Parses `flight_telemetry_log.csv` and renders 3 pub-quality 300-DPI PNGs into the `plots/` directory.

---
*Component Reference Guide maintained for codebase audits and architecture reviews.*
