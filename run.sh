#!/bin/bash
# The public 0.3 suite through the engine, against the server serve.sh started; then the kit's own scorer.
#   pip install -r requirements.txt   (the kit itself: pip install -e <decision-index checkout>, commit below)
PORT=${PORT:-8000}
OUT=${OUT:-runs/anyjev-l0-gemma-4-31b}
PYTHONPATH=$(dirname "$0"):$PYTHONPATH python -m decision_index run --engine anyjev_engine:AnyJevEngine \
  --option base_url=http://127.0.0.1:$PORT --option model=google/gemma-4-31B-it \
  --option level=L0 --option max_rotations=2 --option big_rotations=2 --out $OUT --compact
python -m decision_index score --edition 0.3 --results $OUT/results.jsonl --out $OUT
