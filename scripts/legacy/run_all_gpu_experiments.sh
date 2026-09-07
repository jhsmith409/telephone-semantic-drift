#!/bin/bash
# Chains all experiments sequentially (single GPU VRAM constraint).
# Order: A (RQ1 seeds) → C (temperature) → B q8 (KV cache) → E (lasagna) → D (embedding validation)
# After this completes, user switches Ollama KV cache to fp16, then runs B fp16.
set -e

cd "$(dirname "$0")/.."

echo "=============================================="
echo "GPU Experiment Chain — started $(date)"
echo "=============================================="

echo ""
echo ">>> Experiment A: RQ1 seeds (17 models × 5 seeds)"
echo ">>> Started: $(date)"
uv run python -u scripts/run_drift_experiment_rq1_seeds.py
echo ">>> Experiment A completed: $(date)"

echo ""
echo ">>> Experiment C: Temperature ablation (4 models × 5 temps)"
echo ">>> Started: $(date)"
uv run python -u scripts/run_drift_experiment_temperature.py
echo ">>> Experiment C completed: $(date)"

echo ""
echo ">>> Experiment B (q8): KV cache seeds (10 models × 5 seeds, q8 KV)"
echo ">>> Started: $(date)"
uv run python -u scripts/run_drift_experiment_rq2_seeds.py --kv-mode q8
echo ">>> Experiment B (q8) completed: $(date)"

echo ""
echo ">>> Experiment E: RQ1 lasagna (6 models × 5 seeds)"
echo ">>> Started: $(date)"
uv run python -u scripts/run_drift_experiment_rq1_lasagna.py
echo ">>> Experiment E completed: $(date)"

echo ""
echo ">>> Experiment D: Embedding validation (bge-m3, no LLM — but uses Ollama for embeddings)"
echo ">>> Started: $(date)"
uv run python -u scripts/validate_embeddings.py
echo ">>> Experiment D completed: $(date)"

echo ""
echo "=============================================="
echo "All experiments done — $(date)"
echo "=============================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Switch Ollama KV cache to fp16"
echo "  2. Run: nohup uv run python -u scripts/run_drift_experiment_rq2_seeds.py --kv-mode fp16 > results/drift_experiment_rq2_seeds/run_fp16.log 2>&1 &"
echo "  3. After fp16 completes, run: uv run python scripts/analyze_paper.py"
