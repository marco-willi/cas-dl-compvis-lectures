#!/bin/bash
# Run the whole experiment: train 18 runs (resumable), evaluate, export, check the export.
#
# Usage:
#   bash experiments/cnn_on_mnist/run_all.sh

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$EXP_DIR/../../.venv/bin/python}"
cd "$EXP_DIR"
mkdir -p outputs

"$PYTHON" -u train.py --all 2>&1 | tee -a outputs/train_all.log
"$PYTHON" -u evaluate.py 2>&1 | tee outputs/evaluate.log
"$PYTHON" export.py
"$PYTHON" check_export.py
