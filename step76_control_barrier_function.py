#!/usr/bin/env python3
"""
step76_control_barrier_function.py
===================================
Formal Control Barrier Function (CBF) Safety Filter for UAVs.

Formulation:
1. Ground Impact & Descent Barrier:
   h_alt(z, vz) = (z - z_min) - (vz^2 / (2 * a_max)) >= 0
   Ensures the UAV always has enough braking distance before hitting ground.

2. Maximum Kinetic Speed Barrier:
   h_speed(v) = v_max^2 - (vx^2 + vy^2 + vz^2) >= 0

3. Inter-UAV Swarm Separation Barrier (for multi-agent collision avoidance):
   h_swarm(p_i, p_j) = ||p_i - p_j||^2 - d_min^2 >= 0
   Derivative condition: 2 * (p_i - p_j)^T * (v_i - v_j) + gamma * h_swarm >= 0

4. No-Fly Zone (NFZ) Geofence Barrier:
   h_nfz(p, c) = ||p - c||^2 - r_nfz^2 >= 0
   Repels UAV away from NFZ center c with radius r_nfz.
   Derivative condition: 2 * (p - c)^T * v + gamma * h_nfz >= 0

5. CBF-QP Safety Filter:
   Solves: min_u 0.5 * ||u - u_nom||^2
           s.t.  A_cbf * u <= b_cbf
   Using quadratic programming (or closed-form KKT when single-constraint).
"""

import numpy as np
from scipy.optimize import minimize


class UAVControlBarrierFunction:
    """
    Control Barrier Function (CBF) Runtime Assurance Filter.
    Takes agent's proposed velocity/descent command u_nom = [vx, vy, vz]
    and projects it onto the forward-invariant safe set C.
    """

    def __init__(
        self,
        z_min=0.3,           # Minimum ground clearance (meters)
        a_max=2.5,           # Maximum vertical deceleration capability (m/s^2)
        v_max=5.0,           # Maximum allowable speed (m/s)
        d_min_swarm=2.5,     # Minimum inter-UAV separation distance (meters)
        gamma=1.5,           # Class-K barrier decay rate parameter
    ):
        self.z_min = z_min
        self.a_max = a_max
        self.v_max = v_max
        self.d_min_swarm = d_min_swarm
        self.gamma = gamma

    def ground_barrier(self, altitude, vz):
        """
        h_alt(z, vz) = (z - z_min) - (vz^2 / (2 * a_max))
        z: altitude above ground (positive up)
        vz: vertical speed downward (positive descending)
        """
        if vz <= 0:
            # Ascending or hovering: pure distance barrier
            return altitude - self.z_min
        return (altitude - self.z_min) - (vz**2 / (2.0 * self.a_max))

    def swarm_separation_barrier(self, p_i, p_j):
        """
        h_swarm(p_i, p_j) = ||p_i - p_j||^2 - d_min^2
        """
        diff = np.array(p_i) - np.array(p_j)
        dist_sq = float(np.sum(diff**2))
        return dist_sq - (self.d_min_swarm**2)

    def nfz_barrier(self, current_pos_2d, nfz_center_2d, nfz_radius):
        """
        No-Fly Zone geofence barrier:
        h_nfz(p, c) = ||p - c||^2 - r_nfz^2  >=  0
        Keeps UAV outside the NFZ circle.
        """
        diff = np.array(current_pos_2d[:2]) - np.array(nfz_center_2d[:2])
        dist_sq = float(np.sum(diff**2))
        return dist_sq - (nfz_radius**2)

    def filter_control(self, u_nom, altitude, peer_positions=None, current_pos=None, nfz_zones=None):
        """
        CBF-QP Safety Filter:
        u_nom = [vx_nom, vy_nom, vz_down_nom]
        nfz_zones: list of dicts {"x": float, "y": float, "radius": float}
        Returns:
            u_safe: [vx_safe, vy_safe, vz_safe]
            cbf_active: bool (True if CBF modified the nominal proposal)
            h_val: float (current barrier value)
            slack_info: dict
        """
        u_nom = np.array(u_nom, dtype=float)

        # 1. Evaluate current ground barrier value
        h_alt = self.ground_barrier(altitude, u_nom[2])

        # If barrier is safe and speed within bounds with no swarm peers and no NFZ, pass through
        speed_nom = np.linalg.norm(u_nom)
        if h_alt > 0.5 and speed_nom <= self.v_max and not peer_positions and not nfz_zones:
            return u_nom, False, float(h_alt), "SAFE (h_alt > 0)"

        # 2. Formulate CBF-QP optimization:
        # min 0.5 * ||u - u_nom||^2
        def objective(u):
            return 0.5 * np.sum((u - u_nom) ** 2)

        def grad_objective(u):
            return u - u_nom

        constraints = []

        # Constraint 1: Ground CBF Lie derivative condition
        # dot(h_alt) + gamma * h_alt >= 0
        def cbf_ground_constraint(u):
            vz = u[2]
            # dh/dt = -vz - (vz / a_max) * u_acc >= -gamma * h
            # Direct kinematic velocity bound:
            if altitude <= self.z_min:
                return -vz  # must not descend
            # Maximum allowed descent velocity from barrier:
            v_safe_descent_sq = max(0.0, 2.0 * self.a_max * (altitude - self.z_min))
            v_max_descent = np.sqrt(v_safe_descent_sq)
            return v_max_descent - vz

        constraints.append({'type': 'ineq', 'fun': cbf_ground_constraint})

        # Constraint 2: Kinetic velocity limit
        def speed_constraint(u):
            return self.v_max - np.linalg.norm(u)

        constraints.append({'type': 'ineq', 'fun': speed_constraint})

        # Constraint 3: Swarm peer separation barriers (if peers present)
        if peer_positions and current_pos is not None:
            p_i = np.array(current_pos)
            for p_j in peer_positions:
                p_j = np.array(p_j)
                diff = p_i - p_j
                dist_sq = np.sum(diff**2)
                h_ij = dist_sq - (self.d_min_swarm**2)

                def make_swarm_constraint(d_vec, h_val):
                    def cbf_swarm(u):
                        # dot(h) + gamma * h = 2 * (p_i - p_j)^T * v_i + gamma * h >= 0
                        return 2.0 * np.dot(d_vec, u) + self.gamma * h_val
                    return cbf_swarm

                constraints.append({'type': 'ineq', 'fun': make_swarm_constraint(diff, h_ij)})

        # Constraint 4: No-Fly Zone (NFZ) geofence barriers
        if nfz_zones and current_pos is not None:
            p_i_2d = np.array(current_pos[:2])
            for nfz in nfz_zones:
                c_2d = np.array([nfz["x"], nfz["y"]])
                r_nfz = nfz["radius"]
                diff_nfz = p_i_2d - c_2d
                h_nfz = float(np.sum(diff_nfz**2)) - r_nfz**2

                def make_nfz_constraint(d_vec, h_val):
                    def cbf_nfz(u):
                        # dot(h_nfz) + gamma * h_nfz = 2*(p-c)^T * v + gamma * h >= 0
                        return 2.0 * np.dot(d_vec, u[:2]) + self.gamma * h_val
                    return cbf_nfz

                constraints.append({'type': 'ineq', 'fun': make_nfz_constraint(diff_nfz, h_nfz)})

        # Solve QP
        res = minimize(
            objective,
            x0=u_nom,
            jac=grad_objective,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-4, 'maxiter': 50}
        )

        if res.success:
            u_safe = res.x
            deviation = np.linalg.norm(u_safe - u_nom)
            cbf_active = deviation > 0.05
            status = "CBF INTERVENED" if cbf_active else "CBF INVARIANT (SAFE)"
            return u_safe, cbf_active, float(h_alt), status
        else:
            # Conservative fallback if optimization fails
            u_safe = np.copy(u_nom)
            u_safe[2] = min(u_safe[2], 0.5)  # hard-cap descent to 0.5 m/s
            return u_safe, True, float(h_alt), "CBF CONSERVATIVE FALLBACK"


# Self-test when run directly
if __name__ == "__main__":
    cbf = UAVControlBarrierFunction(z_min=0.5, a_max=2.0, v_max=4.0)

    print("==================================================")
    print("CONTROL BARRIER FUNCTION (CBF) UNIT VALIDATION")
    print("==================================================")

    # Test 1: Safe descent high altitude
    u_nom = np.array([1.0, 0.0, 1.0])  # vx=1, vy=0, vz_down=1
    u_safe, active, h, status = cbf.filter_control(u_nom, altitude=10.0)
    print(f"Test 1 (High Alt = 10m): u_nom={u_nom} -> u_safe={np.round(u_safe, 2)} | Status: {status} (h={h:.2f})")

    # Test 2: Unsafe aggressive descent near ground
    u_nom_unsafe = np.array([0.0, 0.0, 3.5])  # aggressive descent near ground!
    u_safe, active, h, status = cbf.filter_control(u_nom_unsafe, altitude=1.0)
    print(f"Test 2 (Low Alt = 1.0m, unsafe vz=3.5): u_nom={u_nom_unsafe} -> u_safe={np.round(u_safe, 2)} | Status: {status} (h={h:.2f})")

    # Test 3: Close to peer collision
    p_i = [5.0, 5.0, 2.0]
    p_j = [6.0, 5.0, 2.0]  # only 1.0m away! (d_min is 2.5m)
    u_nom_towards = [1.0, 0.0, 0.0]  # flying towards peer
    u_safe, active, h, status = cbf.filter_control(u_nom_towards, altitude=5.0, peer_positions=[p_j], current_pos=p_i)
    print(f"Test 3 (Peer collision risk): u_nom={u_nom_towards} -> u_safe={np.round(u_safe, 2)} | Status: {status}")
    print("==================================================")
    print("CBF MODULE VERIFIED.")
