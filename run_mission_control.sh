#!/usr/bin/env bash
# ==============================================================================
# run_mission_control.sh — Single-Click Mission Control & Dashboard Launcher
# ==============================================================================
# Usage:
#   ./run_mission_control.sh
# ==============================================================================

set -e

SWARM_DIR="/home/samiksha/uav_fdi_data"
PYTHON_BIN="$SWARM_DIR/fdi_env/bin/python3"
PORT=8080

echo "=================================================================="
echo "  🛸 LAUNCHING UAV SWARM UNIFIED MISSION CONTROL CENTER"
echo "=================================================================="

# Clean up any stale instances to ensure fresh code reload
pkill -9 -f "mission_control.py" 2>/dev/null || true
pkill -9 -f "step80_multiagent_swarm.py" 2>/dev/null || true
sleep 1

echo "🚀 Starting Mission Control Server on http://localhost:$PORT..."
$PYTHON_BIN "$SWARM_DIR/mission_control.py" &
SERVER_PID=$!
sleep 1.5

# Open web browser
echo "🌐 Opening Mission Control Dashboard in your browser..."
xdg-open "http://localhost:$PORT" &>/dev/null || sensible-browser "http://localhost:$PORT" &>/dev/null || echo "Open http://localhost:$PORT in your browser."

echo "=================================================================="
echo "🎉 Mission Control Dashboard is LIVE at: http://localhost:$PORT"
echo "   - One-Go Mission Setup (Fleet Size, Fault Scenarios)"
echo "   - Real-time 2D Swarm Radar with No-Fly Zones"
echo "   - Live FDI Fault Alerts & Battery Telemetry Cards"
echo "   - Master System Reports & Scientific Evaluation Graphs"
echo "=================================================================="

# Keep alive if started directly
wait
