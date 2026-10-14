#!/usr/bin/env bash
# Submit the independent 20,000-step v3 geometry + v2 bright experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BN3D_ITERATIONS=20000 exec /bin/bash "$SCRIPT_DIR/submit_bn3d_pun_v3_v2bright_2000.sh" "$@"
