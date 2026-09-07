#!/bin/bash
# Re-run Exp 1 to pick up lasagna conditions (spaghetti already done).
# Run this AFTER the main wrapper (run_qwen35_experiments.sh) finishes.
# Usage: nohup bash scripts/run_qwen35_lasagna_followup.sh > results/qwen35_lasagna_followup.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Qwen3.5 Lasagna Follow-up ==="
echo "Started: $(date -u)"
echo ""

echo "--- Exp 1: Size × Quant (lasagna conditions) ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_size_quant.py
echo "Finished: $(date -u)"
echo ""

echo "=== Lasagna follow-up complete ==="
echo "Finished: $(date -u)"
