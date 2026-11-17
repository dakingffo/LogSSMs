#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
WORKERS="${ABLATION_WORKERS:-6}"
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""

"$PYTHON_BIN" -m experiment.ablation_online_analysis.run_online_analysis \
  --trees 32 64 128 \
  --batch-size 8 \
  --min-sampling-len 8192 \
  --hit-half-life 1024 \
  --workers "$WORKERS" \
  --out experiment/ablation_online_analysis/results/aligned-prefill-distribution

"$PYTHON_BIN" -m experiment.ablation_online_analysis.run_regularization \
  --workers "$WORKERS" \
  --out experiment/ablation_online_analysis/results/exponent-regularization
