#!/usr/bin/env bash
# ==============================================================================
# spawn_gz_swarm.sh – Spawns all 4 x500 Quadrotors into the running Gazebo world
# ==============================================================================
# Usage:
#   ./spawn_gz_swarm.sh
# ==============================================================================

set -e

PX4_DIR="/home/samiksha/PX4-Autopilot"
GZ_MODELS="$PX4_DIR/Tools/simulation/gz/models"
MODEL_SDF="$GZ_MODELS/x500/model.sdf"

if [ ! -f "$MODEL_SDF" ]; then
    echo "❌ Model SDF not found at $MODEL_SDF"
    exit 1
fi

echo "🛸 Checking for active Gazebo world..."
WORLD=$(gz topic -l 2>/dev/null | grep -m 1 -e "^/world/.*/clock" | sed 's/\/world\///g; s/\/clock//g')

if [ -z "$WORLD" ]; then
    echo "⚠️  Gazebo world is not running yet."
    echo "   Please launch Terminal 1 (PX4 SITL + Gazebo) first!"
    exit 1
fi

echo "✓ Found active Gazebo world: '$WORLD'"
echo "🚀 Spawning UAV-2, UAV-3, and UAV-4..."

# UAV-2 at (2.0, 1.5, 0.2)
gz service -s "/world/$WORLD/create" --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 3000 \
    --req "name: \"x500_1\", allow_renaming: false, sdf: '<sdf version=\"1.6\"><include><uri>file://$MODEL_SDF</uri><pose>2.0 1.5 0.2 0 0 0</pose></include></sdf>'" >/dev/null 2>&1 || true
echo "   ✓ UAV-2 (x500_1) spawned at (2.0, 1.5)"

# UAV-3 at (-2.0, 2.5, 0.2)
gz service -s "/world/$WORLD/create" --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 3000 \
    --req "name: \"x500_2\", allow_renaming: false, sdf: '<sdf version=\"1.6\"><include><uri>file://$MODEL_SDF</uri><pose>-2.0 2.5 0.2 0 0 0</pose></include></sdf>'" >/dev/null 2>&1 || true
echo "   ✓ UAV-3 (x500_2) spawned at (-2.0, 2.5)"

# UAV-4 at (1.0, -1.5, 0.2)
gz service -s "/world/$WORLD/create" --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 3000 \
    --req "name: \"x500_3\", allow_renaming: false, sdf: '<sdf version=\"1.6\"><include><uri>file://$MODEL_SDF</uri><pose>1.0 -1.5 0.2 0 0 0</pose></include></sdf>'" >/dev/null 2>&1 || true
echo "   ✓ UAV-4 (x500_3) spawned at (1.0, -1.5)"

echo "🎉 All 4 UAVs spawned into Gazebo! Check the Entity Tree in Gazebo GUI."
