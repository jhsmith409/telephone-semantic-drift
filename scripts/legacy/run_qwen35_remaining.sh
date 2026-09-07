#!/bin/bash
# Run remaining Qwen3.5 experiments after the main wrapper finishes.
# Covers: Exp 4 (sysprompt), Exp 1 lasagna follow-up.
# Usage: nohup bash scripts/run_qwen35_remaining.sh > results/qwen35_remaining.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Qwen3.5 Remaining Experiments ==="
echo "Started: $(date -u)"
echo ""

echo "--- Experiment 4: System Prompt Ablation (9B + 27B × 20 prompts × 2 recipes × 5 seeds) ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_sysprompt.py
echo "Finished: $(date -u)"
echo ""

echo "--- Experiment 1 (lasagna follow-up): Size × Quant lasagna conditions ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_size_quant.py
echo "Finished: $(date -u)"
echo ""

echo "=== All remaining experiments complete ==="
echo "Finished: $(date -u)"
