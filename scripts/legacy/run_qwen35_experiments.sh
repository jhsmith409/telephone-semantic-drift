#!/bin/bash
# Run all Qwen3.5 experiments sequentially: size×quant first, then temperature.
# Usage: nohup bash scripts/run_qwen35_experiments.sh > results/qwen35_experiments.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Qwen3.5 Experiment Suite ==="
echo "Started: $(date -u)"
echo ""

echo "--- Experiment 1: Size × Quantization ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_size_quant.py
echo "Finished: $(date -u)"
echo ""

echo "--- Experiment 2: Temperature Ablation ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_temperature.py
echo "Finished: $(date -u)"
echo ""

echo "--- Experiment 4: System Prompt Ablation ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_sysprompt.py
echo "Finished: $(date -u)"
echo ""

echo "--- Experiment 1 (lasagna follow-up): Size × Quant lasagna ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_size_quant.py
echo "Finished: $(date -u)"
echo ""

echo "=== All Qwen3.5 experiments complete ==="
echo "Finished: $(date -u)"
