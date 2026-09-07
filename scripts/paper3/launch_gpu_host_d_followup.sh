#!/bin/bash
# Waits for the gpu-host-d spaghetti sysprompt run (PID 1943202), then runs
# option 1 (cross-host seed replication of baseline, 30 iters) and
# option 3 (long-horizon baseline, 100 iters) on gpu-host-d's Qwen3.6-35B.
RUN="uv run python -u /home/jhsmith/telephone/scripts/paper3/run.py"
D=/home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4-gpu-host-d
while kill -0 1943202 2>/dev/null; do sleep 30; done
cd /home/jhsmith/telephone
$RUN --label qwen3.6-35b-nvfp4-gpu-host-d --exp baseline --workers 8 > $D/baseline.log 2>&1
$RUN --label qwen3.6-35b-nvfp4-gpu-host-d --exp baseline --iterations 100 --out-name baseline_100iter --workers 8 > $D/baseline_100iter.log 2>&1
