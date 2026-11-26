#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" -m experiment.ablation_online_analysis.run_online_analysis "$@"
