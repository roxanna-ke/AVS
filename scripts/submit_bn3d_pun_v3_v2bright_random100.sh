#!/usr/bin/env bash
# Random100 BN3D diagnostic: original/bright, 2000 and 20000 PUN NeRF steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_ROOT="/mnt/scratch/imos-students/ke"
WORKER="$CONTAINER_ROOT/repos/PUN/scripts/submit_bn3d_pun_v3_v2bright_random100.sh"
SOURCE="$CONTAINER_ROOT/outputs/bn3d_pun_v3_v2bright_2000/building1"
OUT="$CONTAINER_ROOT/outputs/bn3d_pun_v3_v2bright_random100/building1"

if [[ "${1:-}" == worker ]]; then
  [[ $# -eq 1 ]] || { echo "Usage: $0 worker" >&2; exit 1; }
  PY="/mnt/home/conda-envs/pun/bin/python"
  [[ -x "$PY" ]] || { echo "Missing PUN environment: $PY" >&2; exit 1; }
  [[ -f "$SOURCE/original/capture_complete.json" && -f "$SOURCE/bright/capture_complete.json" ]] || {
    echo "Complete v2-bright random1 source capture is required: $SOURCE" >&2
    exit 1
  }
  cd "$CONTAINER_ROOT/repos/PUN"
  "$PY" -B tools/bn3d_pun_native_v3.py prepare --root "$OUT"

  for light in original bright; do
    if [[ ! -f "$OUT/$light/selection_complete.json" ]]; then
      "$PY" -B tools/bn3d_pun_native_v3.py capture-random100 --root "$OUT" \
        --source-root "$SOURCE" --lighting "$light" --random-seed 142
    fi
    if [[ ! -f "$OUT/$light/capture_complete.json" ]]; then
      for ((start=20; start<100; start+=10)); do
        "$PY" -B tools/bn3d_pun_native_v3.py render --root "$OUT" \
          --lighting "$light" --split random100 --start "$start" --count 10
      done
      "$PY" -B tools/bn3d_pun_native_v3.py finalize --root "$OUT" --lighting "$light"
    fi
  done

  for iterations in 2000 20000; do
    for light in original bright; do
      if [[ ! -f "$OUT/runs_$iterations/$light/random100/complete.json" ]]; then
        "$PY" -B tools/bn3d_pun_train_eval_v3.py train-eval --root "$OUT" \
          --lighting "$light" --policy random100 --iterations "$iterations" \
          --runs-dir "runs_$iterations" --train-diagnostics
      fi
    done
    "$PY" -B tools/bn3d_pun_train_eval_v3.py summarize --root "$OUT" \
      --iterations "$iterations" --runs-dir "runs_$iterations" --policies random100
    "$PY" -c 'import json,sys; p=sys.argv[1]; assert json.load(open(p))["complete"], f"Incomplete summary: {p}"' \
      "$OUT/summary_$iterations.json"
  done
  echo "Evaluation: $OUT/summary_2000.csv and $OUT/summary_20000.csv"
  echo "Training fit: $OUT/train_summary_2000.csv and $OUT/train_summary_20000.csv"
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
JOB_NAME="${CLUSTER_USER:-ke}-bn3d-r100-v2-$(date +%H%M%S)"

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
  --command -- /bin/bash "$WORKER" worker

cat <<EOF
Submitted: $JOB_NAME
Logs:      runai logs -f $JOB_NAME -p $PROJECT
Output:    $OUT
EOF
