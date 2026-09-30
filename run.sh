#!/usr/bin/env bash
# Rebuild the 2020 bandwidth-robustness CSV summaries from frozen evidence JSONs.
# Writes to results_rerun/ (git-ignored). Does not reconstruct 2021 intervals.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
python "$ROOT/code/robust_bandwidth_analysis_portable.py"
