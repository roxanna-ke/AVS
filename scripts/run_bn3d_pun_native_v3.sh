#!/usr/bin/env bash
# BN3D/PUN v3 10k follow-up: reuse captures, six NeRF runs, summarize.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$REPO/../.." && pwd)"
OUT="$ROOT/outputs/bn3d_pun_native_v3/building1"
ITERATIONS=10000
RUNS_DIR="runs_10000"
SUMMARY="$OUT/summary_10000.csv"
PY="/mnt/home/conda-envs/pun/bin/python"
[[ -x "$PY" ]] || { echo "Missing PUN environment: $PY" >&2; exit 1; }
cd "$REPO"

"$PY" -B tools/bn3d_pun_native_v3.py prepare --root "$OUT"
for light in original bright; do
  if [[ ! -f "$OUT/previews/$light/preview.json" ]]; then
    "$PY" -B tools/bn3d_pun_native_v3.py preview --root "$OUT" --lighting "$light"
  fi
  if [[ ! -f "$OUT/$light/selection_complete.json" ]]; then
    "$PY" -B tools/bn3d_pun_native_v3.py capture --root "$OUT" --lighting "$light"
  fi
  if [[ ! -f "$OUT/$light/capture_complete.json" ]]; then
    for split in avs20 random1 random2 eval; do
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

for light in bright original; do
  for policy in avs20 random1 random2; do
    if [[ ! -f "$OUT/$RUNS_DIR/$light/$policy/complete.json" ]]; then
      "$PY" -B tools/bn3d_pun_train_eval_v3.py train-eval --root "$OUT" \
        --lighting "$light" --policy "$policy" \
        --iterations "$ITERATIONS" --runs-dir "$RUNS_DIR"
    fi
  done
done
"$PY" -B tools/bn3d_pun_train_eval_v3.py summarize --root "$OUT" \
  --iterations "$ITERATIONS" --runs-dir "$RUNS_DIR"
echo "BN3D/PUN v3 10000-step summary: $SUMMARY"
