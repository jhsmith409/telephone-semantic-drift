#!/bin/bash
# gpu-host-d :8005 Qwen3.6-35B replicate, Phase 2b (author order: 1,2,4 then 3)
RUN="uv run python -u /home/jhsmith/telephone/scripts/paper3/run.py --label qwen3.6-35b-nvfp4-gpu-host-d --workers 8"
D=/home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4-gpu-host-d
cd /home/jhsmith/telephone
$RUN --exp baseline --seeds 3,7,11,19,23,29,31,37,41,47 > $D/phase2b_baseline.log 2>&1
$RUN --exp sysprompt --recipes lasagna --seeds 42,137,256,512,1024 > $D/phase2b_sysprompt_lasagna.log 2>&1
$RUN --exp temperature --temps 0.0 --seeds 42,137,256,512,1024,3,7,11,19,23,29,31,37,41,47 --t0-all-seeds > $D/phase2b_t0.log 2>&1
$RUN --exp thinking --iterations 100 --seeds 42,137,256,512,1024 --out-name thinking_100iter > $D/phase2b_thinking_100iter.log 2>&1
echo "PHASE2B GPU_HOST_D-8005 QUEUE COMPLETE"
