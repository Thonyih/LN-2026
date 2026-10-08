#!/bin/sh
set -eu
TASK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK_PYTHON="$TASK_DIR/.venv/bin/python"
FORCE_FLAG=""
if [ "${1:-}" = "--force" ]; then FORCE_FLAG="--force"; fi
for regime in random grouped; do
  for variant in head headtail; do
    if [ -z "$FORCE_FLAG" ] && [ -f "$TASK_DIR/artifacts/${regime}_clinicalbert_${variant}_training.json" ] && [ -f "$TASK_DIR/artifacts/${regime}_clinicalbert_${variant}_validation.json" ] && [ -f "$TASK_DIR/artifacts/${regime}_clinicalbert_${variant}_test.npz" ]; then
      printf 'Reusing completed BERT comparison: %s %s\n' "$regime" "$variant"
      continue
    fi
    "$TASK_PYTHON" "$TASK_DIR/src/bert.py" --regime "$regime" --variant "$variant" $FORCE_FLAG > "$TASK_DIR/bert_${regime}_${variant}.log" 2>&1
  done
done
