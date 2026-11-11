# ALGORITHMS AND MATHEMATICAL FORMULATIONS SPECIFICATION
## Comprehensive Algorithmic Inventory of the Autonomous Multi-Agent UAV Swarm System

---

This document outlines the formal algorithms, pseudo-code, mathematical formulations, and execution flows implemented across the codebase (`step80_multiagent_swarm.py`, `mission_control.py`, and `web_gui/index.html`).

---

## Algorithm 1: ExtraTrees Ensemble Fault Detection and Isolation (FDI)

### Objective
Isolate sensor anomalies (GPS loss, Gyroscope drift, Accelerometer clipping, Barometer spikes, Compass deviation, and Battery sag) in under 15 milliseconds without requiring computationally prohibitive deep neural networks.

### Mathematical Formulation
Given a continuous multi-sensor telemetry stream $s(t) = [\omega(t), a(t), p_{baro}(t), \psi(t), G(t)]^T$:
1. A sliding temporal buffer $\mathcal{W}_\tau = \{s(t - \tau), \dots, s(t)\}$ of length $\tau = 5$ samples extracts a statistical feature vector $f(t) \in \mathbb{R}^{D}$:
   $$f(t) = \left[ \mu(\mathcal{W}_\tau), \sigma(\mathcal{W}_\tau), \max(\mathcal{W}_\tau) - \min(\mathcal{W}_\tau), \Delta s(t) \right]^T$$
2. The ExtraTrees (Extremely Randomized Trees) ensemble builds $M$ unpruned decision trees $\{T_1, \dots, T_M\}$. At each internal node, $K$ features are randomly selected, and random cut-points are chosen independently of the target labels.
3. Node impurity is evaluated via Gini Index:
   $$I_G(p) = 1 - \sum_{c=1}^{C} p_c^2$$
4. Majority ensemble soft-voting assigns class probability:
   $$\hat{y}(t) = \arg\max_{c \in \mathcal{C}} \frac{1}{M} \sum_{m=1}^M \mathbb{P}_m(y = c \mid f(t))$$
   Where $\mathcal{C} = \{\text{Nominal}, \text{GPS\_Fail}, \text{Gyro\_Fail}, \text{Accel\_Fail}, \text{Baro\_Fail}, \text{Compass\_Fail}, \text{Batt\_Low}\}$.

```text
Algorithm 1: ExtraTrees Sensor FDI Classification
Input  : Raw sensor telemetry stream s(t)
Output : Fault label L_fdi in C, Confidence score conf in [0, 100]

1: Extract sliding temporal window W_tau from telemetry history buffer
2: Compute statistical feature vector f(t) = [mean(W_tau), std(W_tau), range(W_tau), delta(s)]
3: For each tree T_m in ExtraTrees ensemble (m = 1 to M):
4:     Compute leaf class distribution P_m(c | f(t))
5: End For
6: P_avg(c) = (1 / M) * sum_{m=1}^M P_m(c | f(t))
7: L_fdi = argmax_{c in C} P_avg(c)
8: conf  = max_{c in C} P_avg(c) * 100
9: Return L_fdi, conf
```

---

## Algorithm 2: LangGraph Finite-State Computational Pipeline

### Objective
Execute an asynchronous, cyclic, directed acyclic computation per drone agent every tick, ensuring deterministic order of operations from telemetry collection to control execution.

```mermaid
graph LR
    START --> Telemetry --> Energy --> FDI --> Risk --> Reasoning --> SwarmCoord --> CBF_L1 --> CBF_L2 --> Execute --> Finalize --> END
```

### Pseudo-code
```text
Algorithm 2: Per-UAV LangGraph Execution Cycle
Input  : Current SwarmAgentState S_i
Output : Updated SwarmAgentState S_i'

1: Node 1 (Telemetry)   : S_i = node_collect_telemetry(S_i)    // Integrates kinematics, checks NFZ distance
2: Node 2 (Energy)      : S_i = node_monitor_energy(S_i)       // Updates alert: NORMAL, LOW (<=30%), CRITICAL (<=15%)
3: Node 3 (FDI)         : S_i = node_run_fdi(S_i)              // Executes Algorithm 1
4: Node 4 (Risk)        : S_i = node_compute_risk(S_i)         // Formulates composite operational risk score
5: Node 5 (Reasoning)   : S_i = node_agentic_reasoning(S_i)    // Proposes NOMINAL, DELEGATE, or FAILSAFE action
6: Node 6 (SwarmCoord)  : S_i = node_swarm_coordination(S_i)   // Coordinates peer state & delegation
7: Node 7 (CBF L1)      : S_i = node_guardrail_l1(S_i)         // Checks safety barrier h(x) >= 0
8: Node 8 (CBF L2)      : S_i = node_guardrail_l2(S_i)         // Computes tangential boundary override v*
9: Node 9 (Execute)     : S_i = node_execute_action(S_i)       // Applies safe velocity to quadrotor kinematics
10: Node 10 (Finalize)  : S_i = node_finalize(S_i)             // Broadcasts heartbeat to decentralized SwarmBus
11: Return S_i
```

---

## Algorithm 3: Dual-Layer Control Barrier Functions (CBF) with Tangential Override

### Objective
Guarantee mathematically forward-invariant safety around No-Fly Zones (NFZs) such that the probability of boundary penetration is strictly zero ($P(\text{Incursion}) = 0$).

### Mathematical Formulation
For each circular No-Fly Zone $k$ centered at $p_{c,k} = [x_{c,k}, y_{c,k}]^T$ with radius $R_k$ and buffer margin $D_{buf} = 5.0\text{m}$:
1. Define the barrier certificate $h_k(p) \in C^1$:
   $$h_k(p) = \|p_{xy} - p_{c,k}\|^2 - (R_k + D_{buf})^2$$
   Safe set: $\mathcal{C}_k = \{p \in \mathbb{R}^3 \mid h_k(p) \ge 0\}$.
2. Time derivative under velocity control $v = [\dot{x}, \dot{y}]^T$:
   $$\dot{h}_k(p, v) = \nabla h_k(p)^T v = 2(p_{xy} - p_{c,k})^T v$$
3. CBF condition with linear class-$\mathcal{K}$ parameter $\gamma > 0$:
   $$\dot{h}_k(p, v) + \gamma h_k(p) \ge 0$$
4. If nominal desired velocity $v_{des}$ violates this condition ($\dot{h}_k + \gamma h_k < 0$), the safety filter overrides the control via orthogonal boundary decomposition:
   $$n = \frac{p_{xy} - p_{c,k}}{\|p_{xy} - p_{c,k}\|}, \quad t = [-n_y, n_x]^T$$
   $$v_{safe} = (v_{des} \cdot t) t + v_{repulsive} n$$
   Where $v_{repulsive} = \max\left(0, -\nabla h_k^T v_{des} + \gamma h_k(p)\right)$.

```text
Algorithm 3: Control Barrier Function (CBF) Safety Filter
Input  : Current position p_xy, Desired velocity v_des, NFZ set { (p_c,k, R_k) }
Output : Certified safe velocity v_safe, Action flag action

1: v_safe = v_des
2: action = "NOMINAL_CONTINUE"
3: For each NFZ k in { 1, ..., K }:
4:     diff = p_xy - p_c,k
5:     dist = norm(diff)
6:     h_val = dist^2 - (R_k + D_buf)^2
7:     If h_val < 0 or dist < (R_k + D_buf):
8:         // Barrier violated or in safety boundary margin
9:         grad_h = 2 * diff
10:        h_dot = dot(grad_h, v_des)
11:        If h_dot + gamma * h_val < 0:
12:            // Inward trajectory detected: Apply Layer 2 Tangential Projection
13:            n = diff / dist
14:            t_tangent = [-n.y, n.x]
15:            // Preserve tangential momentum while repelling radially
16:            v_safe = dot(v_des, t_tangent) * t_tangent + (1.5 * n)
17:            action = "NFZ_REROUTE"
18:        End If
19:    End If
20: End For
21: Return v_safe, action
```

---

## Algorithm 4: Distributed Fitness-Based Leader Re-Election

### Objective
Maintain uninterrupted swarm command, MAVLink telemetry, and task oversight without a single-point-of-failure centralized master station.

### Mathematical Formulation
The election triggers immediately when:
- Current leader suffers an FDI sensor failure ($L_{fdi} \ne \text{Nominal}$), OR
- Current leader battery drops to warning levels ($B \le 20\%$), OR
- Current leader enters failsafe landing.

Each agent $i$ evaluates all active online peers $j \in \mathcal{A}_{online}$ using the composite fitness function:
$$F_j = w_b \cdot \frac{B_j}{100} + w_r \cdot (1 - R_j) + w_g \cdot \mathbb{I}(G_j) + w_s \cdot S_j$$
Where:
- $w_b = 0.40$ (Battery Weight)
- $w_r = 0.30$ (Risk Inversion Weight)
- $w_g = 0.20$ (GPS Navigation Validity Weight)
- $w_s = 0.10$ (Sensor Integrity Weight)
- $\mathbb{I}(G_j) = 1$ if GPS fix valid, else $0$.
- $S_j = 1.0$ if no sensor faults, else $0.5$.

$$\text{Elected Leader} = \arg\max_{j \in \mathcal{A}_{online}} F_j$$

```text
Algorithm 4: Distributed Leader Election Consensus
Input  : SwarmBus peer states { S_1, ..., S_N }, Current Leader ID L_curr
Output : New Leader ID L_new

1: If S_{L_curr}.fault != "No Failure" or S_{L_curr}.battery <= 20.0 or S_{L_curr}.on_ground:
2:     candidates = [ S_i for S_i in swarm if not S_i.on_ground ]
3:     If candidates is empty: Return None
4:     For each agent a in candidates:
5:         f_batt = a.battery / 100.0
6:         f_risk = 1.0 - a.composite_risk
7:         f_gps  = 1.0 if a.gps_valid else 0.0
8:         f_sens = 1.0 if a.fault == "No Failure" else 0.5
9:         score[a.id] = (0.4 * f_batt) + (0.3 * f_risk) + (0.2 * f_gps) + (0.1 * f_sens)
10:    End For
11:    L_new = argmax_{id} score[id]
12:    If L_new != L_curr:
13:        Broadcast LeaderChangeEvent(old=L_curr, new=L_new, score=score[L_new])
14:        Update MAVLink master stream to L_new
15:    End If
16:    Return L_new
17: End If
18: Return L_curr
```

---

## Algorithm 5: Autonomous Waypoint Handoff & Task Delegation

### Objective
Ensure 100% mission task completion even when individual drones fail or exhaust energy mid-flight.

```text
Algorithm 5: Decentralized Waypoint Task Delegation
Input  : Faulted Agent S_fault, SwarmBus registry
Output : Target Peer S_peer assigned orphan waypoints

1: If S_fault.fault != "No Failure" or S_fault.energy == "CRITICAL":
2:     orphan_wps = S_fault.waypoints_queue
3:     If orphan_wps is empty: Return
4:     
5:     // Search for most suitable peer: Online, Healthy, Lowest Workload
6:     eligible_peers = [ a for a in swarm if a.id != S_fault.id and not a.on_ground and a.fault == "No Failure" ]
7:     If eligible_peers is empty:
8:         LogWarning("No eligible peers available. Orphan tasks dropped.")
9:         Return
10:    
11:    // Sort peers by remaining capacity (fewer tasks assigned + higher battery)
12:    best_peer = argmin_{a in eligible_peers} ( len(a.waypoints_queue) - (a.battery / 50.0) )
13:    
14:    // Handoff transfer
15:    best_peer.waypoints_queue.extend(orphan_wps)
16:    best_peer.role = "Task Reallocated (Sector Exp)"
17:    best_peer.total_tasks += len(orphan_wps)
18:    
19:    S_fault.waypoints_queue.clear()
20:    S_fault.delegation_status = "DELEGATED"
21:    S_fault.delegation_target = best_peer.id
22:    S_fault.final_action = "FAILSAFE_LAND"
23:    
24:    LogEvent("Handoff: " + S_fault.id + " -> " + best_peer.id + " (" + len(orphan_wps) + " tasks)")
25: End If
```

---

## Algorithm 6: Return-To-Home (RTL) Spatial Convergence

### Objective
Vector drones from anywhere in the 3D airspace directly back to the Home Base Helipad $(0, 0, 0)$ and conduct safe touchdown sequencing into dedicated landing slots.

### Mathematical Formulation
For drone $i$ with current position $p_i = [x_i, y_i, z_i]^T$ and assigned slot coordinate $p_{slot, i} = [x_{s,i}, y_{s,i}, 0]^T$:
1. Horizontal separation vector:
   $$d_{xy} = \sqrt{(x_i - x_{s,i})^2 + (y_i - y_{s,i})^2}$$
2. Horizontal guidance phase ($d_{xy} > 1.5\text{m}$):
   $$\dot{x}_i = -\frac{x_i - x_{s,i}}{d_{xy}} \cdot v_{cruise}, \quad \dot{y}_i = -\frac{y_i - y_{s,i}}{d_{xy}} \cdot v_{cruise}$$
   Altitude maintained at cruise altitude ($z_i \ge 10.0\text{m}$) to clear any ground obstacles.
3. Vertical descent phase ($d_{xy} \le 1.5\text{m}$):
   $$\dot{x}_i = 0, \quad \dot{y}_i = 0, \quad \dot{z}_i = -v_{descent}$$
4. Touchdown threshold ($z_i \le 0.2\text{m}$):
   $$z_i = 0, \quad \text{on\_ground} = \text{True}, \quad \text{Motors Disarmed}$$

---
*Algorithmic Specification Document Version: 2.4.0 — Maintained for academic citations and implementation verification.*
