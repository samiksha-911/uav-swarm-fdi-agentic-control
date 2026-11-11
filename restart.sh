#!/bin/bash
# ==============================================================================
# One-Click Instant Restart for UAV Swarm 3D Mission Control
# ==============================================================================

echo "🛑 [1/3] Terminating any existing Mission Control and Swarm processes..."
pkill -9 -f "mission_control.py" 2>/dev/null
pkill -9 -f "step80_multiagent_swarm.py" 2>/dev/null
sleep 1

echo "🧹 [2/3] Cleaning up temporary command pipes and locks..."
rm -f /tmp/swarm_cmd /home/samiksha/uav_fdi_data/swarm_cmd.txt

echo "🚀 [3/3] Launching 3D Mission Control on http://localhost:8080 ..."
cd /home/samiksha/uav_fdi_data
exec /home/samiksha/uav_fdi_data/fdi_env/bin/python3 mission_control.py
