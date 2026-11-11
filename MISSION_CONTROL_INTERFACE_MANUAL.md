# MISSION CONTROL INTERFACE MANUAL & COMMAND REFERENCE
## Operator Handbook for the Autonomous 3D Multi-Agent Swarm Ground Control Center

---

## 1. Quick Launch & Setup

To launch the entire mission control suite in one go without touching complex terminals:

```bash
cd /home/samiksha/uav_fdi_data
./run_mission_control.sh
```

This starts the asynchronous Tornado backend and automatically opens your web browser to:
**`http://localhost:8080`** (or **`http://localhost:8081`** if 8080 is occupied).

---

## 2. Interface Layout Walkthrough

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🛸 AUTONOMOUS UAV SWARM — 3D MISSION CONTROL     [Backend: STANDALONE_SIM] [Swarm: NOMINAL] │
├──────────────────────┬──────────────────────────────────────────────┬───────────────────────┤
│ LEFT PANEL:          │ CENTER PANEL:                                │ RIGHT PANEL:          │
│                      │ ┌──────────────────────────────────────────┐ │                       │
│ 1. Mission Config    │ │  3D SPATIAL THEATER / 2D RADAR ARENA     │ │ Live Telemetry Cards: │
│    - Fleet Size      │ │  - 3D Drones with spinning rotors        │ │                       │
│    - Profile         │ │  - Highlighted Home Base Helipad & Beacon│ │ 👑 UAV-1 (Master)     │
│    - Auto Scenarios  │ │  - 3D No-Fly Zone Cylinders (Red/Orange) │ │    Batt: 92% | Alt:12m│
│                      │ │  - 3D Planned Routes & Waypoint Crystals │ │    FDI: No Failure    │
│ 2. Flight Commands   │ │  - Yellow Dashed Inbound RTL Corridors   │ │                       │
│    - Takeoff / Land  │ └──────────────────────────────────────────┘ │ ✈️ UAV-2               │
│    - Return to Home  │ ┌──────────────────────────────────────────┐ │    Batt: 88% | Alt:14m│
│                      │ │ 🧠 AGENTIC AI PIPELINE: 1->2->3->4->5     │ │    FDI: No Failure    │
│ 3. Fault Injection   │ └──────────────────────────────────────────┘ │                       │
│    - Target UAV Drop │ ┌──────────────────────────────────────────┐ │ ✈️ UAV-3               │
│    - Fault Type Drop │ │ 👑 Master Telemetry & Progress (63%)     │ │    Batt: 85% | Alt:11m│
│    - Inject / Clear  │ └──────────────────────────────────────────┘ │                       │
│                      │ ┌──────────────────────────────────────────┐ │ ✈️ UAV-4               │
│ 4. Analytics & Export│ │ 📝 Live Operational Event Log Stream     │ │    Batt: 90% | Alt:10m│
│    - 300-DPI Plots   │ └──────────────────────────────────────────┘ │    [RTL] [Reset]      │
│    - JSON / CSV logs │                                              │                       │
└──────────────────────┴──────────────────────────────────────────────┴───────────────────────┘
```

---

## 3. 3D Spatial Theater Controls

| Action | Mouse / Touch Gesture | Result in 3D Scene |
|---|---|---|
| **Rotate / Orbit** | **Left-Click + Drag** | Freely rotates the 3D airspace around the center target. |
| **Pan Camera** | **Right-Click + Drag** | Shifts the camera viewing target across the X/Z ground plane. |
| **Zoom In / Out** | **Mouse Scroll Wheel** | Zooms into individual drone quadcopter mechanics or zooms out to fleet view. |
| **View Switcher** | **Click `[2D Radar]` / `[3D Theater]`** | Toggles between realistic 3D WebGL space and 2D tactical radar. |

### Camera Presets (Bottom-Right Buttons):
- **`[3D Orbit]`**: Resets camera to standard isometric 45° perspective.
- **`[Focus Home]`**: Zooms directly onto the glowing Home Base Helipad and RTL recovery slots.
- **`[Top-Down]`**: Bird’s-eye 90° overhead satellite perspective.
- **`[Track Master]`**: Centers camera on the currently elected Swarm Leader.
- **`[Focus Fault]`**: Instantly pans and focuses on any faulted or degraded UAV for emergency inspection.
- **`[Reset]`**: Restores initial default viewing angle.

---

## 4. Complete Operator Command Glossary

You can issue commands via the dashboard UI buttons or through the backend `/api/command` endpoint:

| Command Syntax | Parameters | Immediate Swarm Effect |
|---|---|---|
| `takeoff` | None | All online UAVs spin motors and climb to designated cruise altitudes (10–14m). |
| `land` | None | Swarm immediately halts forward motion and conducts vertical failsafe descent. |
| **`rtl`** or **`home`** | Optional: `[UAV-ID]` or `all` | **Return To Home**: Commands drone(s) to vector straight back to Home Base `(0, 0, 0)` along highlighted inbound recovery corridors and touch down in assigned landing slots. |
| `fault <target> <type>` | `target`: `UAV-1`..`4`<br>`type`: `GPS`, `Gyroscope`, `Accelerometer`, `Barometer`, `Compass` | Injects sensor anomaly. Target isolates fault via ExtraTrees, flags red alert, reallocates tasks, and descends safely. |
| `batt <target> <pct>` | `target`: `UAV-1`..`4`<br>`pct`: e.g. `14.0`, `8.0` | Overrides battery level. Low battery triggers leader election; critical battery triggers emergency landing. |
| `clear <target>` | `target`: `UAV-1`..`4` | Restores specified drone to 100% nominal health, resets faults, and resumes patrol. |
| `clear all` | None | Restores entire fleet to nominal state, 92% battery, and clears all alerts. |
| `add <target>` | `target`: `UAV-5`, `6`, etc. | **Horizontal Scaling**: Dynamically spawns new drone agent into live swarm with assigned patrol corridor. |
| `remove <target>` | `target`: e.g. `UAV-2` | Safely recalls specified UAV, hands off its remaining waypoints to peers, and decommissions it. |
| `loiter <target>` | `target`: e.g. `UAV-3` | Sets drone to standby reserve loitering pattern. |

---

## 5. Step-by-Step Test Scenarios

### Scenario 1: Nominal Mission Launch
1. On Left Panel, select **Fleet Size: 4 UAVs**, **Automated Fault: Nominal Flight**.
2. Click **`INITIATE MISSION`**.
3. *Expected Result*: Status badge turns green (`Swarm: NOMINAL`), 4 3D quadcopters appear at Home Helipad with spinning rotors, climb to patrol altitudes, and start patrolling their colored waypoint routes.

### Scenario 2: Return to Home (RTL) Demonstration
1. While swarm is flying, click the yellow button: **`RETURN TO HOME (RTL ALL)`**.
2. *Expected Result*: 
   - Yellow dashed inbound RTL corridors brightly illuminate from each drone directly to the Home Base Helipad.
   - All drones halt forward waypoint patrol, turn towards `(0, 0, 0)`, and fly along the RTL vectors.
   - Each UAV descends smoothly into its designated touchdown slot on the Helipad.
   - Action badges update to `RETURN_TO_HOME`.

### Scenario 3: Sensor Fault Injection (Gyroscope Loss on UAV-2)
1. In Fault Injection Console, select **Target: UAV-2**, **Fault: Gyroscope Failure**.
2. Click **`Inject Fault`**.
3. *Expected Result*:
   - UAV-2 beacon in 3D flashes alarm red, and its ground projection turns red.
   - AI Decision Pipeline turns red: `2. FDI: 🔴 UAV-2 Gyroscope Failure` $\to$ `4. CBF: 🛡️ FAILSAFE_LAND`.
   - UAV-2 safely lands vertically. Its remaining waypoints are autonomously reallocated to peers.

### Scenario 4: Master Battery Depletion & Leader Re-Election
1. In Fault Injection Console, select **Target: UAV-1**, **Fault: Low Battery (14% - Re-elect)**.
2. Click **`Inject Fault`**.
3. *Expected Result*:
   - UAV-1 battery bar drops to 14% (red).
   - Distributed consensus triggers within 250ms.
   - Highest-scoring healthy drone (e.g., UAV-2 or UAV-4) is elected new Swarm Master (`👑 Master: UAV-2`).
   - The golden leader aura dynamically transfers to the new leader quadcopter in 3D.

---
*Manual Version: 2.4.0 — Unified Autonomous UAV Swarm Ground Control Center.*
