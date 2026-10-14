#!/usr/bin/env bash
# Submit BN3D/PUN v3 manually from a Run:ai login node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The login node may expose this PVC under /mnt/imos/scratch, while the
# container mount below is explicitly /mnt/scratch. Commands passed to Run:ai
# must use the path inside the container.
CONTAINER_ROOT="/mnt/scratch/imos-students/ke"
WORKER="$CONTAINER_ROOT/repos/PUN/tools/run_bn3d_pun_native_v3.sh"
PROJECT="${RUNAI_PROJECT:-imos-ke}"
IMAGE="${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}"
GPU_AMOUNT="${MY_GPU_AMOUNT:-1.0}"
SCRATCH_PVC="${SCRATCH_PVC:-imos-scratch}"
HOME_PVC="${HOME_PVC:-home}"
JOB_NAME="${CLUSTER_USER:-ke}-bn3d-pun-v3-$(date +%H%M%S)"

command -v runai >/dev/null 2>&1 || {
  echo "Run:ai CLI is unavailable. Submit from the login node." >&2
  exit 1
}
[[ -f "$SCRIPT_DIR/run_bn3d_pun_native_v3.sh" ]] || {
  echo "Worker script not found beside submit script: $SCRIPT_DIR/run_bn3d_pun_native_v3.sh" >&2
  exit 1
}

runai submit --name "$JOB_NAME" \
  -p "$PROJECT" \
  --image "$IMAGE" \
  --gpu "$GPU_AMOUNT" \
  --large-shm \
  --node-pools "${NODE:-default}" \
  --working-dir "$CONTAINER_ROOT" \
  --environment CLUSTER_USER="${CLUSTER_USER:-ke}" \
  --environment CLUSTER_USER_ID="${CLUSTER_USER_ID:-308061}" \
  --environment CLUSTER_GROUP_NAME="${CLUSTER_GROUP_NAME:-imos}" \
  --environment CLUSTER_GROUP_ID="${CLUSTER_GROUP_ID:-79821}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/mnt/scratch" \
  --existing-pvc "claimname=${HOME_PVC},path=/mnt/home" \
  --command -- /bin/bash "$WORKER"

cat <<EOF
Submitted: $JOB_NAME
Logs:      runai logs -f $JOB_NAME -p $PROJECT
Status:    runai describe job $JOB_NAME -p $PROJECT
Output:    $CONTAINER_ROOT/outputs/bn3d_pun_native_v3/building1/summary_10000.csv
EOF
