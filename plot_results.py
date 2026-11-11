#!/usr/bin/env python3
"""
=============================================================================
plot_results.py — Scientific Visualization & Evaluation for UAV Swarm
=============================================================================
Analyzes 'flight_telemetry_log.csv' and outputs publication-quality figures:
  1. swarm_3d_flight_trajectories.png  (3D paths, Waypoints, NFZ Geofences)
  2. swarm_fdi_safety_timeline.png     (Battery, FDI Fault Detection, Actions)
  3. swarm_mesh_performance.png       (Tasks Completed, Mesh Link & Latency)
=============================================================================
Usage:
  python plot_results.py
=============================================================================
"""

import os
import sys

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless saving
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_CSV = "/home/samiksha/uav_fdi_data/flight_telemetry_log.csv"
OUTPUT_DIR = "/home/samiksha/uav_fdi_data/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})

UAV_COLORS = {
    "UAV-1": "#D4AF37",  # Gold / Swarm Leader
    "UAV-2": "#E63946",  # Red / Fault Drone
    "UAV-3": "#1D3557",  # Deep Navy / Survey
    "UAV-4": "#2A9D8F",  # Teal / Patrol
}

NFZ_ZONES = [
    {"name": "NFZ-Alpha (Hospital)", "x": 50.0, "y": 60.0, "radius": 18.0, "alt_ceil": 25.0, "color": "red"},
    {"name": "NFZ-Bravo (Airport)",  "x": -15.0, "y": 90.0, "radius": 22.0, "alt_ceil": 30.0, "color": "orange"},
]


def load_telemetry():
    if not os.path.exists(DATA_CSV) or os.path.getsize(DATA_CSV) == 0:
        print(f"⚠️ Telemetry log not found at {DATA_CSV}.")
        print("   Generating synthetic sample run to showcase visualization...")
        return generate_sample_data()

    try:
        df = pd.read_csv(DATA_CSV)
        if len(df) < 5:
            print("⚠️ Insufficient rows in CSV. Generating synthetic sample...")
            return generate_sample_data()
        return df
    except Exception as e:
        print(f"⚠️ Error reading CSV ({e}). Using sample data...")
        return generate_sample_data()


def generate_sample_data():
    """Generates realistic synthetic multi-agent flight data if CSV is fresh."""
    ticks = np.arange(1, 120)
    rows = []
    
    # UAV-1 nominal survey
    for t in ticks:
        angle = t * 0.05
        rows.append({
            "tick": t, "uav_id": "UAV-1", "role": "👑 Swarm Leader", "is_leader": 1,
            "pos_x": 30 * np.cos(angle), "pos_y": 20 * np.sin(angle), "altitude": 12.0 + np.sin(t*0.1),
            "vx": -1.5 * np.sin(angle), "vy": 1.0 * np.cos(angle), "vz": 0.0,
            "battery_pct": max(20.0, 100.0 - t * 0.08), "fault_label": "No Failure",
            "final_action": "NOMINAL_CONTINUE", "current_wp_idx": (t // 20) % 6,
            "tasks_completed": min(6, t // 18), "total_tasks": 6,
            "nfz_dist": 45.0, "nfz_violation": 0, "link_quality_pct": 100.0, "comms_latency_ms": 0.5
        })

    # UAV-2 GPS fault at tick 45
    for t in ticks:
        if t < 45:
            px = 20.0 + t * 0.8
            py = 10.0 + t * 0.6
            alt = 14.0
            vx, vy = 1.4, 1.2
            fault = "No Failure"
            action = "NOMINAL_CONTINUE"
        else:
            px, py = 56.0, 37.0  # Holding position
            alt = 13.5
            vx, vy = 0.0, 0.0
            fault = "GPS Failure"
            action = "ALTITUDE_HOLD_HOVER"
            
        rows.append({
            "tick": t, "uav_id": "UAV-2", "role": "Relay / Escort", "is_leader": 0,
            "pos_x": px, "pos_y": py, "altitude": alt,
            "vx": vx, "vy": vy, "vz": 0.0,
            "battery_pct": max(15.0, 95.0 - t * 0.07), "fault_label": fault,
            "final_action": action, "current_wp_idx": 2,
            "tasks_completed": 2 if t >= 45 else min(2, t // 20), "total_tasks": 6,
            "nfz_dist": max(5.0, 40.0 - t * 0.4), "nfz_violation": 1 if t >= 45 else 0,
            "link_quality_pct": max(60.0, 92.0 - t * 0.2), "comms_latency_ms": 2.5 + t * 0.03
        })

    # UAV-3
    for t in ticks:
        angle = t * 0.04
        rows.append({
            "tick": t, "uav_id": "UAV-3", "role": "Survey Bravo", "is_leader": 0,
            "pos_x": -20.0 + 15 * np.cos(angle), "pos_y": 30.0 + 25 * np.sin(angle), "altitude": 11.0,
            "vx": 0.8, "vy": 0.6, "vz": 0.0,
            "battery_pct": max(25.0, 90.0 - t * 0.09), "fault_label": "No Failure",
            "final_action": "NOMINAL_CONTINUE", "current_wp_idx": (t // 24) % 5,
            "tasks_completed": min(5, t // 22), "total_tasks": 5,
            "nfz_dist": 35.0, "nfz_violation": 0, "link_quality_pct": 85.0, "comms_latency_ms": 3.2
        })

    # UAV-4
    for t in ticks:
        angle = t * 0.06
        rows.append({
            "tick": t, "uav_id": "UAV-4", "role": "Swarm Reserve", "is_leader": 0,
            "pos_x": 10.0 + 20 * np.cos(angle), "pos_y": -15.0 + 15 * np.sin(angle), "altitude": 10.0,
            "vx": 1.1, "vy": -0.8, "vz": 0.0,
            "battery_pct": max(30.0, 96.0 - t * 0.06), "fault_label": "No Failure",
            "final_action": "NOMINAL_CONTINUE", "current_wp_idx": (t // 15) % 5,
            "tasks_completed": min(5, t // 15), "total_tasks": 5,
            "nfz_dist": 55.0, "nfz_violation": 0, "link_quality_pct": 91.0, "comms_latency_ms": 2.8
        })

    return pd.DataFrame(rows)


def plot_3d_trajectories(df):
    """Figure 1: 3D Spatial Trajectories with No-Fly Zone Geofence Obstacles."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot NFZ geofences as translucent cylinders
    for nfz in NFZ_ZONES:
        u = np.linspace(0, 2 * np.pi, 30)
        h = np.linspace(0, nfz["alt_ceil"], 15)
        U, H = np.meshgrid(u, h)
        X = nfz["x"] + nfz["radius"] * np.cos(U)
        Y = nfz["y"] + nfz["radius"] * np.sin(U)
        ax.plot_surface(X, Y, H, alpha=0.18, color=nfz["color"])
        ax.plot(nfz["x"] + nfz["radius"] * np.cos(u),
                nfz["y"] + nfz["radius"] * np.sin(u),
                nfz["alt_ceil"], color=nfz["color"], linestyle="--", linewidth=1.5,
                label=f"{nfz['name']} (r={nfz['radius']}m)")

    # Plot UAV flight tracks
    for uav_id, grp in df.groupby("uav_id"):
        color = UAV_COLORS.get(uav_id, "blue")
        grp = grp.sort_values("tick")
        clean_role = grp['role'].iloc[-1].encode("ascii", "ignore").decode("ascii").strip()
        ax.plot(grp["pos_x"], grp["pos_y"], grp["altitude"],
                label=f"{uav_id} ({clean_role})",
                color=color, linewidth=2.4)
        
        # Start marker
        ax.scatter(grp["pos_x"].iloc[0], grp["pos_y"].iloc[0], grp["altitude"].iloc[0],
                   color=color, marker="o", s=60, edgecolors="black")
        
        # End / Current marker
        ax.scatter(grp["pos_x"].iloc[-1], grp["pos_y"].iloc[-1], grp["altitude"].iloc[-1],
                   color=color, marker="^", s=110, edgecolors="black")

    ax.set_title("Autonomous UAV Swarm — 3D Spatial Trajectories & NFZ Geofence Avoidance", pad=15)
    ax.set_xlabel("X (East, meters)")
    ax.set_ylabel("Y (North, meters)")
    ax.set_zlabel("Altitude (Z, meters)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.95), frameon=True)
    ax.view_init(elev=28, azim=-55)

    out_file = os.path.join(OUTPUT_DIR, "swarm_3d_flight_trajectories.png")
    fig.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Generated: {out_file}")


def plot_fdi_and_safety_timeline(df):
    """Figure 2: Multi-panel timeline of Battery, FDI Risk, and Action responses."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Subplot 1: Battery Discharge
    for uav_id, grp in df.groupby("uav_id"):
        grp = grp.sort_values("tick")
        axes[0].plot(grp["tick"], grp["battery_pct"], label=uav_id, color=UAV_COLORS.get(uav_id, "black"), linewidth=2.0)
    axes[0].axhline(20.0, color="red", linestyle=":", label="Low Battery Threshold (20%)")
    axes[0].set_ylabel("Battery (%)")
    axes[0].set_title("Swarm Health & Energy Telemetry Timeline")
    axes[0].legend(loc="lower left", ncol=5)
    axes[0].set_ylim(0, 105)

    # Subplot 2: FDI Fault Status
    for uav_id, grp in df.groupby("uav_id"):
        grp = grp.sort_values("tick")
        is_fault = (grp["fault_label"] != "No Failure").astype(int)
        axes[1].step(grp["tick"], is_fault, label=uav_id, color=UAV_COLORS.get(uav_id, "black"), linewidth=2.0, where="post")
    axes[1].set_ylabel("FDI Fault (0=Nominal, 1=Fail)")
    axes[1].set_title("Autonomous Fault Detection & Isolation (FDI) Response")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["NOMINAL", "FAULT DETECTED"])
    axes[1].legend(loc="upper left")

    # Subplot 3: Flight Velocity Magnitude
    for uav_id, grp in df.groupby("uav_id"):
        grp = grp.sort_values("tick")
        spd = np.sqrt(grp["vx"]**2 + grp["vy"]**2)
        axes[2].plot(grp["tick"], spd, label=uav_id, color=UAV_COLORS.get(uav_id, "black"), linewidth=2.0)
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].set_xlabel("Simulation Tick")
    axes[2].set_title("Agentic Speed Modulation (Zero Velocity upon FDI Trigger)")
    axes[2].legend(loc="upper right")

    out_file = os.path.join(OUTPUT_DIR, "swarm_fdi_safety_timeline.png")
    fig.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Generated: {out_file}")


def plot_mesh_performance(df):
    """Figure 3: Mission Progress and Inter-Drone Mesh Network Latency/Quality."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Subplot 1: Tasks Completed
    for uav_id, grp in df.groupby("uav_id"):
        grp = grp.sort_values("tick")
        axes[0].step(grp["tick"], grp["tasks_completed"], label=f"{uav_id} Tasks", color=UAV_COLORS.get(uav_id, "black"), linewidth=2.0, where="post")
    axes[0].set_ylabel("Waypoints Completed")
    axes[0].set_title("Swarm Mission Task Progression per UAV")
    axes[0].legend(loc="upper left")

    # Subplot 2: Mesh Network Latency
    for uav_id, grp in df.groupby("uav_id"):
        if uav_id == "UAV-1":
            continue  # Master is 0ms
        grp = grp.sort_values("tick")
        axes[1].plot(grp["tick"], grp["comms_latency_ms"], label=f"{uav_id} Latency", color=UAV_COLORS.get(uav_id, "black"), linewidth=1.8)
    axes[1].set_ylabel("Mesh Latency (ms)")
    axes[1].set_xlabel("Simulation Tick")
    axes[1].set_title("P2P Decentralized Mesh Network Latency to Master UAV")
    axes[1].legend(loc="upper left")

    out_file = os.path.join(OUTPUT_DIR, "swarm_mesh_performance.png")
    fig.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Generated: {out_file}")


def main():
    print("=" * 70)
    print("🚀 Generating UAV Swarm Scientific Evaluation Plots...")
    print("=" * 70)
    df = load_telemetry()
    print(f"📊 Loaded {len(df)} telemetry samples across {df['uav_id'].nunique()} UAV agents.")
    
    plot_3d_trajectories(df)
    plot_fdi_and_safety_timeline(df)
    plot_mesh_performance(df)

    print("=" * 70)
    print(f"🎉 Evaluation figures successfully saved in:\n   {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
