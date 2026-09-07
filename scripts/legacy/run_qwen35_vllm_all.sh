#!/bin/bash
# Run all remaining vLLM experiments: temperature ablation, then system prompt ablation.
# Usage: nohup bash scripts/run_qwen35_vllm_all.sh > results/qwen35_vllm_all.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Qwen3.5 35B AWQ (vLLM) Experiments ==="
echo "Started: $(date -u)"
echo ""

echo "--- Temperature Ablation (2 recipes × 5 temps × seeds = 42 conditions) ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_vllm_temperature.py
echo "Finished: $(date -u)"
echo ""

echo "--- System Prompt Ablation (20 prompts × 2 recipes × 5 seeds = 200 conditions) ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_vllm_sysprompt.py
echo "Finished: $(date -u)"
echo ""

echo "=== All vLLM experiments complete ==="
echo "Finished: $(date -u)"
