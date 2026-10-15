#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/scratch/imos-students/ke
REPO="$ROOT/repos/myRepo"
PY=/mnt/home/conda-envs/pun/bin/python

case "${1:-}" in
  worker)
    [[ -x "$PY" ]] || { echo "Missing PUN Python: $PY" >&2; exit 1; }
    [[ -f "$ROOT/outputs/bn3d_pun_v3_v2bright_random100/building1/asset_manifest.json" ]] || {
      echo "BN3D capture is missing in the container" >&2; exit 1;
    }
    export WANDB_MODE=disabled
    cd "$REPO"
    "$PY" -B tools/bn3d_pun_render_predictions.py
    ;;
  submit)
    command -v runai >/dev/null || { echo "RunAI CLI is unavailable" >&2; exit 1; }
    job_name="${CLUSTER_USER:-ke}-bn3d-pun-render-$(date +%H%M%S)"
    runai submit --name "$job_name" \
      -p "${RUNAI_PROJECT:-imos-ke}" \
      --image "${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}" \
      --gpu "${MY_GPU_AMOUNT:-1.0}" --large-shm \
      --node-pools "${NODE:-default}" \
      --working-dir "$ROOT" \
      --environment "CLUSTER_USER=${CLUSTER_USER:-ke}" \
      --environment "CLUSTER_USER_ID=${CLUSTER_USER_ID:-308061}" \
      --environment "CLUSTER_GROUP_NAME=${CLUSTER_GROUP_NAME:-imos}" \
      --environment "CLUSTER_GROUP_ID=${CLUSTER_GROUP_ID:-79821}" \
      --existing-pvc "claimname=${SCRATCH_PVC:-imos-scratch},path=/mnt/scratch" \
      --existing-pvc "claimname=${HOME_PVC:-home},path=/mnt/home" \
      --command -- /bin/bash "$REPO/scripts/submit_bn3d_pun_render_predictions.sh" worker
    echo "Submitted $job_name; predictions will be saved under runs_reference/*/*/predictions"
    ;;
  *)
    echo "Usage: $0 {worker|submit}" >&2
    exit 2
    ;;
esac
