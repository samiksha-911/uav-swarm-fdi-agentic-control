#!/usr/bin/env bash
# ==============================================================================
# launch_all.sh – Automated 4-Component Launcher for Autonomous UAV Swarm
# ==============================================================================
# Usage:
#   ./launch_all.sh          # Launches all 4 components in dedicated terminal tabs
#   ./launch_all.sh stop     # Gracefully stops all simulation and python processes
# ==============================================================================

set -e

SWARM_DIR="/home/samiksha/uav_fdi_data"
PX4_DIR="/home/samiksha/PX4-Autopilot"
QGC_BIN="/home/samiksha/QGroundControl.AppImage"

# Fallback if symlink missing
if [ ! -f "$QGC_BIN" ]; then
    QGC_BIN="/home/samiksha/Downloads/QGroundControl.AppImage"
fi

if [ "$1" = "stop" ] || [ "$1" = "kill" ]; then
    echo "🛑 Stopping all UAV Swarm, PX4 SITL, and Gazebo processes..."
    pkill -9 -f "step80_multiagent_swarm.py" 2>/dev/null || true
    pkill -9 -f "px4" 2>/dev/null || true
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "QGroundControl" 2>/dev/null || true
    rm -f /tmp/swarm_cmd /home/samiksha/uav_fdi_data/swarm_cmd.txt
    echo "✓ Clean shutdown complete."
    exit 0
fi

echo "=================================================================="
echo "  🛸 AUTONOMOUS UAV SWARM — ONE-CLICK LAUNCHER"
echo "=================================================================="

# Check gnome-terminal
if ! command -v gnome-terminal &>/dev/null; then
    echo "❌ gnome-terminal is not available. Please run in separate terminal tabs manually."
    exit 1
fi

echo "🚀 Spawning Terminal Windows / Tabs:"
echo "   [1] PX4 SITL & Gazebo Garden"
echo "   [2] QGroundControl Ground Station"
echo "   [3] Multi-Agent Swarm Orchestrator (LangGraph × 4)"
echo "   [4] Interactive Command & Fault Injector Console"
echo "=================================================================="

gnome-terminal \
    --tab --title="T1: PX4 SITL + Gazebo" -- bash -c "cd '$PX4_DIR' && make px4_sitl gz_x500; exec bash" \
    --tab --title="T2: QGroundControl" -- bash -c "sleep 3 && '$QGC_BIN'; exec bash" \
    --tab --title="T3: Swarm Orchestrator" -- bash -c "sleep 7 && cd '$SWARM_DIR' && source fdi_env/bin/activate && python step80_multiagent_swarm.py; exec bash" \
    --tab --title="T4: Command Console" -- bash -c "sleep 8 && cd '$SWARM_DIR' && ./inject.sh; exec bash"

echo "🎉 All 4 components initiated in new terminal tabs!"
echo "   To stop everything later, run: ./launch_all.sh stop"
echo "=================================================================="
