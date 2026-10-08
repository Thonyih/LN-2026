#!/bin/sh
set -eu
TASK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK_PYTHON="$TASK_DIR/.venv/bin/python"
"$TASK_PYTHON" -m unittest discover -s "$TASK_DIR/tests" -v > "$TASK_DIR/tests.log" 2>&1
"$TASK_PYTHON" "$TASK_DIR/src/prepare.py" > "$TASK_DIR/prepare.log" 2>&1
"$TASK_PYTHON" "$TASK_DIR/src/classical.py" > "$TASK_DIR/classical.log" 2>&1
"$TASK_PYTHON" "$TASK_DIR/src/reconcile_classical_scores.py" > "$TASK_DIR/alignment.log" 2>&1
"$TASK_PYTHON" "$TASK_DIR/src/augmentation_candidates.py" > "$TASK_DIR/augmentation_candidates.log" 2>&1
"$TASK_PYTHON" "$TASK_DIR/src/augmentation_study.py" > "$TASK_DIR/augmentation.log" 2>&1
"$TASK_DIR/run_training_queue.sh"
"$TASK_PYTHON" "$TASK_DIR/src/finalize.py" > "$TASK_DIR/finalize.log" 2>&1
"$TASK_PYTHON" "$TASK_DIR/src/predict.py" > "$TASK_DIR/predict.log" 2>&1
"$TASK_PYTHON" "$TASK_DIR/src/memorisation.py" > "$TASK_DIR/memorisation.log" 2>&1
"$TASK_PYTHON" "$TASK_DIR/src/report.py"
"$TASK_PYTHON" "$TASK_DIR/src/verify.py"
