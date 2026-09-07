#!/bin/bash
# Run after experiments.sh (PID 19447) finishes:
#   1. Re-run 22 timed-out Exp 1 spaghetti conditions (sequential)
#   2. (Exp 4 and Exp 1 lasagna are already chained in experiments.sh)
#
# Usage:
#   nohup bash scripts/run_qwen35_after_chain.sh > results/qwen35_rerun.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

# Wait for the main experiment chain to finish
WAIT_PID=19447
echo "=== Waiting for PID $WAIT_PID (experiments.sh) to finish ==="
echo "Started waiting: $(date -u)"
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 30
done
echo "PID $WAIT_PID finished: $(date -u)"
echo ""

echo "--- Re-running 22 timed-out Exp 1 conditions (sequential) ---"
echo "Started: $(date -u)"
uv run python -u scripts/run_qwen35_rerun_timeouts.py
echo "Finished: $(date -u)"
echo ""

echo "=== All done ==="
echo "Finished: $(date -u)"
