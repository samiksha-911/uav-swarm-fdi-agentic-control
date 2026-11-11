#!/bin/bash
# ==============================================================================
# start_presentation.sh – 1-Click Master Launch for Final Presentation
# ==============================================================================
# In your final viva/presentation, simply run:
#     ./start_presentation.sh
#
# It automatically:
#   1. Kills any stale background processes cleanly.
#   2. Starts PX4 Autopilot SITL + Gazebo 3D simulation (if built).
#   3. Starts Micro-XRCE-DDS Agent bridge (if available).
#   4. Launches QGroundControl Graphical Ground Station (if found).
#   5. Launches the flagship 4-UAV Multi-Agent Swarm Dashboard!
#   6. Automatically terminates all processes when you press Ctrl-C.
#
# If PX4 is not built, the swarm still runs in autonomous simulation mode.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PX4_BUILD="/home/samiksha/PX4-Autopilot/build/px4_sitl_default"
PX4_BIN="$PX4_BUILD/bin/px4"
PX4_ENV="$PX4_BUILD/rootfs/gz_env.sh"
PX4_GZ_DIR="$PX4_BUILD/src/modules/simulation/gz_bridge"

echo "========================================================================"
echo "  🚁 PREPARING 1-CLICK MULTI-AGENT UAV PRESENTATION ENVIRONMENT..."
echo "========================================================================"

# Step 0: Clean any stale background simulation processes
pkill -9 -f px4 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f ruby 2>/dev/null || true
pkill -9 -f MicroXRCEAgent 2>/dev/null || true
rm -f /tmp/px4* /tmp/*gz* 2>/dev/null || true
sleep 1

# Setup Python + ROS2 environments (non-fatal)
source fdi_env/bin/activate 2>/dev/null || true
source /opt/ros/humble/setup.bash 2>/dev/null || true
source /home/samiksha/uav_swarm_ws/install/setup.bash 2>/dev/null || true

# Cleanup handler on Ctrl-C
SITL_PID="" BRIDGE_PID="" QGC_PID=""
cleanup() {
    echo ""
    echo "🛑 Shutting down all presentation processes..."
    [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" 2>/dev/null || true
    [ -n "$SITL_PID"  ] && kill "$SITL_PID"  2>/dev/null || true
    [ -n "$QGC_PID"   ] && kill "$QGC_PID"   2>/dev/null || true
    pkill -9 -f px4 2>/dev/null || true
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f MicroXRCEAgent 2>/dev/null || true
    rm -f /tmp/px4* /tmp/*gz* 2>/dev/null || true
    echo "✓ Clean shutdown complete."
}
trap cleanup EXIT INT TERM

# Step 1: Start PX4 SITL + Gazebo in background (only if built)
if [ -f "$PX4_BIN" ]; then
    echo "🚀 [1/4] Starting PX4 SITL + Gazebo Garden..."
    source "$PX4_ENV" 2>/dev/null || true
    (cd "$PX4_BUILD" && PX4_SIM_MODEL=gz_x500 GZ_IP=127.0.0.1 "$PX4_BIN" -d -s etc/init.d-posix/rcS -i 0 > /tmp/px4_sitl.log 2>&1) &
    SITL_PID=$!
    echo "   ✓ PX4 SITL started (PID=$SITL_PID) | Log: /tmp/px4_sitl.log"
else
    echo "⚠️  [1/4] PX4 SITL not built – skipping Gazebo."
    echo "   Swarm will run in autonomous simulation mode (fully functional for demo)."
    echo "   To build PX4: ./launch_all.sh build"
fi

# Step 2: Start Micro-XRCE-DDS Agent (only if available)
if command -v MicroXRCEAgent &>/dev/null; then
    echo "🌐 [2/4] Starting Micro-XRCE-DDS Agent (UDP 8888)..."
    MicroXRCEAgent udp4 -p 8888 > /dev/null 2>&1 &
    BRIDGE_PID=$!
    echo "   ✓ DDS bridge started (PID=$BRIDGE_PID)"
else
    echo "⚠️  [2/4] MicroXRCEAgent not found – skipping DDS bridge."
fi

# Step 3: Launch QGroundControl GUI (only if found)
QGC_PATH=$(find /home/samiksha/Downloads -name "*.AppImage" 2>/dev/null | head -1)
if [ -n "$QGC_PATH" ]; then
    echo "🎮 [3/4] Launching QGroundControl Ground Station..."
    chmod +x "$QGC_PATH"
    "$QGC_PATH" > /dev/null 2>&1 &
    QGC_PID=$!
    echo "   ✓ QGroundControl started (PID=$QGC_PID)"
else
    echo "⚠️  [3/4] QGroundControl AppImage not found in ~/Downloads – skipping."
fi

echo "⏳ Waiting 3 seconds for simulation systems to stabilise..."
sleep 3

# Step 4: Run the flagship 4-UAV Multi-Agent Swarm with Leader Election
echo ""
echo "🛸 [4/4] Starting Flagship 4-UAV Multi-Agent Swarm Dashboard..."
echo "========================================================================"
if [ $# -gt 0 ]; then
    python3 step80_multiagent_swarm.py "$@"
else
    # Default demo: battery fault at tick 4 triggers leader election
    python3 step80_multiagent_swarm.py \
        --inject-uav UAV-1 \
        --inject-battery 14.0 \
        --inject-at-tick 4
fi
