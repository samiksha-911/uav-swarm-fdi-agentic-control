# PROJECT QUICKSTART, VERIFICATION & VIVA CHEATSHEET
## One-Stop Verification Guide for Autonomous Multi-Agent UAV Swarms

---

## 1. Quick Launch Summary

| Objective | Command | Notes |
|---|---|---|
| **Launch Unified 3D Web Ground Control** | `./run_mission_control.sh` | Starts web server & opens browser to `http://localhost:8080` (or `8081`). |
| **Run Diagnostic Multi-Terminal Mode** | `./launch_all.sh` | Opens 4 tabs: QGC, Swarm Kernel, Inject CLI, and Live CSV log. |
| **Generate 300-DPI Scientific Figures** | `fdi_env/bin/python3 plot_results.py` | Produces 3 high-res figures in `plots/` directory. |
| **Execute Pipeline Verification Tests** | `fdi_env/bin/python3 verify_pipeline.py` | Automated unit/integration test suite. |

---

## 2. Five-Minute Live Demo Checklist for Evaluators

```text
[ ] STEP 1: Start Mission Control
    Execute: ./run_mission_control.sh
    Verify: Browser opens to 3D Tactical Theater with dark aerospace grid.

[ ] STEP 2: Highlight Home Base
    Look at center coordinates (0, 0, 0):
    Verify: Metallic helipad with 'H' cross, 4 colored touchdown slots, 
            expanding circular radar pulse waves, and 50m vertical sky beacon.

[ ] STEP 3: Launch Autonomous Swarm
    On Left Panel: Select Fleet Size: 4, Scenario: Nominal -> Click "INITIATE MISSION".
    Verify: 4 quadcopters spin propellers, climb to 10-14m, and fly planned 3D routes.

[ ] STEP 4: Demonstrate Return to Home (RTL)
    Click yellow button: "RETURN TO HOME (RTL ALL)".
    Verify: Yellow dashed inbound vectors appear from each drone to Home Base,
            and drones turn and fly back to their assigned landing slots.

[ ] STEP 5: Inject Sensor Failure
    Select Target: UAV-2, Fault: Gyroscope Failure -> Click "Inject Fault".
    Verify: Red alarm flashes on UAV-2, AI Decision Pipeline switches to FDI/CBF Land,
            and remaining waypoints are reallocated to peers.

[ ] STEP 6: Inject Battery Depletion & Leader Election
    Select Target: UAV-1, Fault: Low Battery (14%) -> Click "Inject Fault".
    Verify: Distributed consensus triggers, Swarm Master transfers to UAV-2,
            and golden crown icon moves in 3D.

[ ] STEP 7: Generate Scientific Evaluation Plots
    Click blue button: "View Scientific Evaluation Plots".
    Verify: Modal displays 300-DPI trajectory, safety timeline, and mesh latency graphs.
```

---

## 3. Top Viva & Defense Questions with Technical Answers

### Q1: Why use ExtraTrees instead of a standard Random Forest or Deep Neural Network?
**Answer**: ExtraTrees (Extremely Randomized Trees) randomizes cut-point splits in addition to feature subsets. This provides lower variance, exceptional resistance to high-frequency sensor noise, and a microsecond-level inference footprint ($< 12\text{ms}$) that can run on an edge companion computer (e.g., Raspberry Pi 4 or Jetson Nano) without requiring power-hungry GPUs.

### Q2: What is the mathematical advantage of Control Barrier Functions over Artificial Potential Fields?
**Answer**: Artificial Potential Fields (APF) model obstacles as repulsive energy fields, but frequently suffer from local minima traps near concave boundaries or opposing goals. Control Barrier Functions (CBFs) define a forward-invariant safe set $\mathcal{C} = \{x \mid h(x) \ge 0\}$. As long as $\dot{h} + \alpha(h) \ge 0$, safety is mathematically guaranteed at every continuous time step with zero boundary breaches.

### Q3: How is leader failure handled without a central ground station?
**Answer**: The swarm uses a decentralized consensus protocol over the `SwarmBus`. Each peer evaluates all online agents via a multi-objective fitness function:
$$F_i = 0.4 \cdot B_i + 0.3 \cdot (1 - R_i) + 0.2 \cdot G_i + 0.1 \cdot S_i$$
The healthy agent with the highest composite fitness score is autonomously designated Master within one simulation tick ($< 250\text{ms}$), guaranteeing zero interruption to MAVLink reporting or mission continuity.

---
*Verification Cheatsheet prepared for project demonstrations and technical vivas.*
