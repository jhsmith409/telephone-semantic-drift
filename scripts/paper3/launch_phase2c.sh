#!/bin/bash
# Phase 2c: fill idle Blackwell hosts (2026-09-06 19:20 UTC)
RUN="uv run python -u /home/jhsmith/telephone/scripts/paper3/run.py"
cd /home/jhsmith/telephone
# gpu-host-e Qwen3.8-27B: T=0 x15, then thinking 100-iter
( $RUN --label qwen3.8-27b-nvfp4 --exp temperature --temps 0.0 --seeds 42,137,256,512,1024,3,7,11,19,23,29,31,37,41,47 --t0-all-seeds --workers 4 > /home/jhsmith/telephone/results/paper3/qwen3.8-27b-nvfp4/phase2c_t0.log 2>&1
  $RUN --label qwen3.8-27b-nvfp4 --exp thinking --iterations 100 --seeds 42,137,256,512,1024 --out-name thinking_100iter --workers 4 > /home/jhsmith/telephone/results/paper3/qwen3.8-27b-nvfp4/phase2c_thinking_100iter.log 2>&1
  echo "PHASE2C GPU_HOST_E COMPLETE" >> /home/jhsmith/telephone/results/paper3/qwen3.8-27b-nvfp4/phase2c_thinking_100iter.log ) &
# gpu-host-d:8006 Flash-Next: T=0 x15, then thinking 100-iter
( $RUN --label qwen3.8-flash-next-nvfp4 --exp temperature --temps 0.0 --seeds 42,137,256,512,1024,3,7,11,19,23,29,31,37,41,47 --t0-all-seeds --workers 4 > /home/jhsmith/telephone/results/paper3/qwen3.8-flash-next-nvfp4/phase2c_t0.log 2>&1
  $RUN --label qwen3.8-flash-next-nvfp4 --exp thinking --iterations 100 --seeds 42,137,256,512,1024 --out-name thinking_100iter --workers 4 > /home/jhsmith/telephone/results/paper3/qwen3.8-flash-next-nvfp4/phase2c_thinking_100iter.log 2>&1
  echo "PHASE2C GPU_HOST_D-8006 COMPLETE" >> /home/jhsmith/telephone/results/paper3/qwen3.8-flash-next-nvfp4/phase2c_thinking_100iter.log ) &
# gpu-host-f Qwen3.6-35B: complete thinking+sysprompt to all 20 prompts (lasagna)
( $RUN --label qwen3.6-35b-nvfp4 --exp sysprompt --thinking --recipes lasagna --seeds 42,137,256,512,1024 --out-name sysprompt_think --workers 8 > /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4/phase2c_sysprompt_think.log 2>&1
  echo "PHASE2C GPU_HOST_F COMPLETE" >> /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4/phase2c_sysprompt_think.log ) &
# gpu-host-d:8005 Qwen3.6-35B replicate: long-horizon to 15 samples, off and on
( $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --exp baseline --iterations 100 --seeds 3,7,11,19,23,29,31,37,41,47 --out-name baseline_100iter --workers 8 > /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4-gpu-host-d/phase2c_baseline_100iter.log 2>&1
  $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --exp thinking --iterations 100 --seeds 3,7,11,19,23,29,31,37,41,47 --out-name thinking_100iter --workers 8 > /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4-gpu-host-d/phase2c_thinking_100iter.log 2>&1
  echo "PHASE2C GPU_HOST_D-8005 COMPLETE" >> /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4-gpu-host-d/phase2c_thinking_100iter.log ) &
wait
