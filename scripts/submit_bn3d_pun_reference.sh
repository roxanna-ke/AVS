#!/usr/bin/env bash
# Run four fixed-input PUN trajectories. Submission is explicit: pass "submit".
set -euo pipefail

ROOT=/mnt/scratch/imos-students/ke
REPO="$ROOT/repos/myRepo"
CAPTURE="$ROOT/outputs/bn3d_pun_v3_v2bright_random100/building1"
PY=/mnt/home/conda-envs/pun/bin/python

case "${1:-}" in
  worker)
    [[ $# -eq 1 ]] || { echo "Usage: $0 worker" >&2; exit 2; }
    [[ -x "$PY" ]] || { echo "Missing PUN Python: $PY" >&2; exit 1; }
    [[ -f "$CAPTURE/original/capture_complete.json" &&
       -f "$CAPTURE/bright/capture_complete.json" ]] || {
      echo "Completed random100 captures are missing in the container: $CAPTURE" >&2; exit 1;
    }
    [[ -n "${WANDB_API_KEY:-}" ]] || {
      echo "WANDB_API_KEY was not injected into this job" >&2; exit 1;
    }
    export WANDB_MODE=online
    export WANDB_PROJECT="${WANDB_PROJECT:-bn3d-pun-reference}"
    attempt="${BN3D_RUN_ATTEMPT:-${RUNAI_JOB_NAME:-manual-$(date +%Y%m%d-%H%M%S)}}"
    cd "$REPO"
    "$PY" -B -c 'import wandb; print("wandb", wandb.__version__)'
    for steps in 2000 20000; do
      for light in original bright; do
        run="$CAPTURE/runs_reference/$light/$steps"
        if [[ -f "$run/complete.json" ]] || compgen -G "$run/attempts/*/complete.json" >/dev/null; then
          echo "Already complete: $run"
          continue
        fi
        "$PY" -B tools/bn3d_pun_reference_run.py --root "$CAPTURE" \
          --lighting "$light" --iterations "$steps" --seed 42 --attempt "$attempt"
      done
    done
    ;;
  submit)
    [[ $# -eq 1 ]] || { echo "Usage: $0 submit" >&2; exit 2; }
    command -v runai >/dev/null || { echo "RunAI CLI is unavailable" >&2; exit 1; }
    project="${RUNAI_PROJECT:-imos-ke}"
    image="${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}"
    [[ "${WANDB_CREDENTIAL_SOURCE:-environment}" == environment ]] || {
      echo "This RunAI CLI supports the exported WANDB_API_KEY route; set WANDB_CREDENTIAL_SOURCE=environment" >&2
      exit 2
    }
    [[ -n "${WANDB_API_KEY:-}" ]] || {
      echo "Export WANDB_API_KEY in this terminal before submitting" >&2; exit 1
    }
    job_name="${CLUSTER_USER:-ke}-bn3d-pun-ref-$(date +%H%M%S)"
    runai submit --name "$job_name" \
      -p "$project" --image "$image" \
      --gpu "${MY_GPU_AMOUNT:-1.0}" --large-shm \
      --node-pools "${NODE:-default}" \
      --working-dir "$ROOT" \
      --environment "CLUSTER_USER=${CLUSTER_USER:-ke}" \
      --environment "CLUSTER_USER_ID=${CLUSTER_USER_ID:-308061}" \
      --environment "CLUSTER_GROUP_NAME=${CLUSTER_GROUP_NAME:-imos}" \
      --environment "CLUSTER_GROUP_ID=${CLUSTER_GROUP_ID:-79821}" \
      --environment "BN3D_RUN_ATTEMPT=$job_name" \
      --existing-pvc "claimname=${SCRATCH_PVC:-imos-scratch},path=/mnt/scratch" \
      --existing-pvc "claimname=${HOME_PVC:-home},path=/mnt/home" \
      --environment "WANDB_API_KEY=$WANDB_API_KEY" \
      --environment "WANDB_PROJECT=${WANDB_PROJECT:-bn3d-pun-reference}" \
      --command -- /bin/bash "$REPO/scripts/submit_bn3d_pun_reference.sh" worker
    echo "Submitted $job_name; output: $CAPTURE/runs_reference"
    ;;
  *)
    echo "Usage: $0 {worker|submit}" >&2
    echo "Export WANDB_API_KEY before submitting from the RunAI login node." >&2
    exit 2
    ;;
esac
