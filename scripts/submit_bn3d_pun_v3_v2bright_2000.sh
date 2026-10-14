#!/usr/bin/env bash
# Submit v3 geometry + v2 bright lighting, with three random policies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_ROOT="/mnt/scratch/imos-students/ke"
WORKER="$CONTAINER_ROOT/repos/PUN/tools/submit_bn3d_pun_v3_v2bright_2000.sh"
ITERATIONS="${BN3D_ITERATIONS:-${2:-2000}}"
case "$ITERATIONS" in
  2000|20000) ;;
  *) echo "BN3D_ITERATIONS must be 2000 or 20000" >&2; exit 1 ;;
esac
OUT="$CONTAINER_ROOT/outputs/bn3d_pun_v3_v2bright_${ITERATIONS}/building1"
SUMMARY="$OUT/summary_${ITERATIONS}.csv"

if [[ "${1:-}" == worker ]]; then
  [[ "${2:-}" == "$ITERATIONS" ]] || { echo "Worker iteration mismatch" >&2; exit 1; }
  PY="/mnt/home/conda-envs/pun/bin/python"
  [[ -x "$PY" ]] || { echo "Missing PUN environment: $PY" >&2; exit 1; }
  cd "$CONTAINER_ROOT/repos/PUN"
  "$PY" -B tools/bn3d_pun_native_v3.py prepare --root "$OUT"
  for light in original bright; do
    if [[ ! -f "$OUT/previews/$light/preview.json" ]]; then
      "$PY" -B tools/bn3d_pun_native_v3.py preview --root "$OUT" \
        --lighting "$light" --bright-lighting v2
    fi
    if [[ ! -f "$OUT/$light/selection_complete.json" ]]; then
      "$PY" -B tools/bn3d_pun_native_v3.py capture --root "$OUT" \
        --lighting "$light" --bright-lighting v2 --random-groups 3
    fi
    if [[ ! -f "$OUT/$light/capture_complete.json" ]]; then
      for split in avs20 random1 random2 random3 eval; do
        total=20
        if [[ "$split" == eval ]]; then total=40; fi
        for ((start=0; start<total; start+=10)); do
          "$PY" -B tools/bn3d_pun_native_v3.py render --root "$OUT" \
            --lighting "$light" --split "$split" --start "$start" --count 10
        done
      done
      "$PY" -B tools/bn3d_pun_native_v3.py finalize --root "$OUT" --lighting "$light"
    fi
  done
  for light in original bright; do
    for policy in avs20 random1 random2 random3; do
      if [[ ! -f "$OUT/runs_$ITERATIONS/$light/$policy/complete.json" ]]; then
        "$PY" -B tools/bn3d_pun_train_eval_v3.py train-eval --root "$OUT" \
          --lighting "$light" --policy "$policy" \
          --iterations "$ITERATIONS" --runs-dir "runs_$ITERATIONS"
      fi
    done
  done
  "$PY" -B tools/bn3d_pun_train_eval_v3.py summarize --root "$OUT" \
    --iterations "$ITERATIONS" --runs-dir "runs_$ITERATIONS" \
    --policies avs20 random1 random2 random3
  echo "Summary: $SUMMARY"
  exit 0
fi

[[ $# -eq 0 ]] || { echo "Usage: $0" >&2; exit 1; }
command -v runai >/dev/null 2>&1 || {
  echo "Run:ai CLI is unavailable. Submit from the login node." >&2
  exit 1
}
[[ -f "$SCRIPT_DIR/$(basename "$WORKER")" ]] || {
  echo "Worker script not found beside submit script" >&2
  exit 1
}
PROJECT="${RUNAI_PROJECT:-imos-ke}"
IMAGE="${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}"
GPU_AMOUNT="${MY_GPU_AMOUNT:-1.0}"
SCRATCH_PVC="${SCRATCH_PVC:-imos-scratch}"
HOME_PVC="${HOME_PVC:-home}"
JOB_NAME="${CLUSTER_USER:-ke}-bn3d-v3-v2b-${ITERATIONS}-$(date +%H%M%S)"
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
  --command -- /bin/bash "$WORKER" worker "$ITERATIONS"

cat <<EOF
Submitted: $JOB_NAME
Logs:      runai logs -f $JOB_NAME -p $PROJECT
Status:    runai describe job $JOB_NAME -p $PROJECT
Output:    $SUMMARY
EOF
