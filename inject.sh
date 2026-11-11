#!/bin/bash
# ==============================================================================
# inject.sh – Live Swarm Command & Fault Injection Helper
# ==============================================================================
# Usage:
#   ./inject.sh 1 gps           # Inject GPS Failure on UAV-1
#   ./inject.sh 1 gyro          # Inject Gyroscope Failure on UAV-1
#   ./inject.sh 1 accel         # Inject Accelerometer Failure on UAV-1
#   ./inject.sh 1 baro          # Inject Barometer Failure on UAV-1
#   ./inject.sh 1 compass       # Inject Compass Failure on UAV-1
#   ./inject.sh 1 clear         # Clear faults on UAV-1
#   ./inject.sh clear           # Clear faults on ALL UAVs
#   ./inject.sh 1 batt 14       # Set UAV-1 battery to 14% (triggers Leader step-down)
#   ./inject.sh add 5 6         # Dynamically add UAV-5 and UAV-6 to active swarm
#   ./inject.sh remove 2        # Decommission UAV-2 and reallocate tasks
# ==============================================================================

CMD_FILE="/tmp/swarm_cmd"

if [ $# -eq 0 ]; then
    echo "=================================================================="
    echo "  🛸 LIVE SWARM COMMAND & FAULT INJECTION"
    echo "=================================================================="
    echo "Usage Examples:"
    echo "  ./inject.sh 1 gps           -> Inject GPS Failure on UAV-1"
    echo "  ./inject.sh 1 gyro          -> Inject Gyroscope Failure on UAV-1"
    echo "  ./inject.sh 1 accel         -> Inject Accelerometer Failure on UAV-1"
    echo "  ./inject.sh 1 clear         -> Clear faults on UAV-1"
    echo "  ./inject.sh clear           -> Clear faults on ALL UAVs"
    echo "  ./inject.sh 1 batt 14       -> Drop UAV-1 battery to 14% (Leader election)"
    echo "  ./inject.sh add 5 6         -> Add UAV-5 and UAV-6 dynamically"
    echo "  ./inject.sh remove 2        -> Remove UAV-2 dynamically"
    echo "=================================================================="
    exit 0
fi

# Detect command patterns
ARG1=$(echo "$1" | tr '[:upper:]' '[:lower:]')
ARG2=$(echo "$2" | tr '[:upper:]' '[:lower:]')
ARG3=$(echo "$3" | tr '[:upper:]' '[:lower:]')
ARG4=$(echo "$4" | tr '[:upper:]' '[:lower:]')

if [ "$ARG1" = "takeoff" ] || [ "$ARG1" = "arm" ]; then
    echo "🚀 Sending ARM & TAKEOFF command to PX4 (port 14540)..."
    /home/samiksha/uav_fdi_data/fdi_env/bin/python3 -c "
from pymavlink import mavutil
import time
try:
    mav = mavutil.mavlink_connection('udpout:127.0.0.1:14540', source_system=255)
    mav.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    time.sleep(0.3)
    mav.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 10.0)
    print('✓ PX4 Armed and Takeoff initiated to 10m altitude.')
except Exception as e:
    print('Warning:', e)
" 2>/dev/null || true
    echo "takeoff" > "$CMD_FILE"
    exit 0

elif [ "$ARG1" = "land" ]; then
    echo "🛬 Sending LAND command to PX4 (port 14540)..."
    /home/samiksha/uav_fdi_data/fdi_env/bin/python3 -c "
from pymavlink import mavutil
try:
    mav = mavutil.mavlink_connection('udpout:127.0.0.1:14540', source_system=255)
    mav.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    print('✓ PX4 Land initiated.')
except Exception as e:
    print('Warning:', e)
" 2>/dev/null || true
    echo "land" > "$CMD_FILE"
    exit 0

elif [ "$ARG1" = "status" ] || [ "$ARG1" = "report" ]; then
    REPORT_FILE="/home/samiksha/uav_fdi_data/master_system_report.json"
    if [ -f "$REPORT_FILE" ]; then
        echo "=================================================================="
        echo "  👑 MASTER UAV SYSTEM TELEMETRY REPORT"
        echo "=================================================================="
        /home/samiksha/uav_fdi_data/fdi_env/bin/python3 -m json.tool "$REPORT_FILE" 2>/dev/null || cat "$REPORT_FILE"
    else
        echo "⚠️ No master report generated yet. Is step80_multiagent_swarm.py running?"
    fi
    exit 0

elif [ "$ARG1" = "plot" ] || [ "$ARG1" = "graph" ]; then
    echo "📊 Running plot_results.py to generate high-resolution figures..."
    /home/samiksha/uav_fdi_data/fdi_env/bin/python3 /home/samiksha/uav_fdi_data/plot_results.py
    exit 0

# Format: clear [target]  e.g. 'clear UAV-2' or 'clear all' or 'clear'
elif [ "$ARG1" = "clear" ] || [ "$ARG1" = "reset" ]; then
    if [ -n "$2" ]; then
        CMD="clear $2"
    else
        CMD="clear all"
    fi

# Format: UAV-2 clear  or  2 clear
elif [ "$ARG2" = "clear" ] || [ "$ARG2" = "ok" ] || [ "$ARG2" = "reset" ]; then
    CMD="clear $1"

# Format: fault <target> <sensor>  e.g. 'fault UAV-2 GPS' or 'fault 2 gps'
elif [ "$ARG1" = "fault" ] || [ "$ARG1" = "inject" ] || [ "$ARG1" = "f" ]; then
    if [ -n "$3" ]; then
        CMD="fault $2 $3"
    else
        CMD="fault UAV-1 $2"
    fi

# Format: batt <target> <pct>  e.g. 'batt UAV-2 14' or 'batt 2 14'
elif [ "$ARG1" = "batt" ] || [ "$ARG1" = "battery" ] || [ "$ARG1" = "b" ]; then
    CMD="batt $2 $3"

# Format: <target> batt <pct>  e.g. 'UAV-2 batt 14' or '2 batt 14'
elif [ "$ARG2" = "batt" ] || [ "$ARG2" = "battery" ] || [ "$ARG2" = "b" ]; then
    CMD="batt $1 $ARG3"

# Format: add <uav_ids...>
elif [ "$ARG1" = "add" ] || [ "$ARG1" = "+" ]; then
    shift
    CMD="add $*"

# Format: remove <uav_ids...>
elif [ "$ARG1" = "remove" ] || [ "$ARG1" = "rm" ] || [ "$ARG1" = "-" ]; then
    shift
    CMD="remove $*"

# Format: <target> <sensor>  e.g. '2 gps' or 'UAV-2 gps'
elif [ -n "$ARG2" ]; then
    CMD="fault $1 $ARG2"

else
    CMD="$*"
fi

echo "$CMD" > "$CMD_FILE"
echo "$CMD" > "/home/samiksha/uav_fdi_data/swarm_cmd.txt" 2>/dev/null || true
echo "✓ Sent command to swarm: '$CMD'"
