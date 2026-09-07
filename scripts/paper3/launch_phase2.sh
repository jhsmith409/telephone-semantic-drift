#!/usr/bin/env bash
# Paper 3 Phase 2 launcher (spec: paper3/PHASE2_EXEC.md, approved 2026-09-06).
#
# One background queue per host; hosts run in parallel, items within a host are
# strictly serial. Everything is resume-safe: existing COMPLETE conditions are
# never re-run and never overwritten (verified by --dry-run before launch).
#
#   bash /home/jhsmith/telephone/scripts/paper3/launch_phase2.sh

set -u
REPO=/home/jhsmith/telephone
RUN="uv run python -u $REPO/scripts/paper3/run.py"
RES=$REPO/results/paper3

# Phase 2 adds 10 independent samples per condition (seeds are NOT reproducible,
# so these are just extra samples written to the SAME data files).
P2=3,7,11,19,23,29,31,37,41,47
P1=42,137,256,512,1024
ALL=$P1,$P2

cd "$REPO" || exit 1

# ---- gpu-host-f .210:8005 — Qwen3.6-35B-A3B NVFP4, vLLM ------------------------
# Item 1 is the determinism control and MUST run first, alone, at workers 1 so
# nothing of ours batches with it. The rest of the queue follows serially.
mkdir -p "$RES/qwen3.6-35b-nvfp4"
nohup bash -c "
  $RUN --label qwen3.6-35b-nvfp4 --exp baseline --recipes spaghetti --seeds $P1 --workers 1 --out-name baseline_w1a
  $RUN --label qwen3.6-35b-nvfp4 --exp baseline --recipes spaghetti --seeds $P1 --workers 1 --out-name baseline_w1b
  $RUN --label qwen3.6-35b-nvfp4 --exp baseline  --seeds $P2 --workers 8
  $RUN --label qwen3.6-35b-nvfp4 --exp thinking  --seeds $P2 --workers 8
  $RUN --label qwen3.6-35b-nvfp4 --exp sysprompt --thinking --recipes lasagna --prompt-ids 1,4,7,11,18 --seeds $P1 --workers 8 --out-name sysprompt_think
  $RUN --label qwen3.6-35b-nvfp4 --exp sysprompt --recipes spaghetti --seeds $P2 --workers 8
  echo 'PHASE2 GPU_HOST_F QUEUE COMPLETE'
" > "$RES/qwen3.6-35b-nvfp4/phase2.log" 2>&1 &
echo "gpu-host-f   (qwen3.6-35b-nvfp4)        PID $!  log $RES/qwen3.6-35b-nvfp4/phase2.log"

# ---- gpu-host-e .208:8005 — Qwen3.8-27B NVFP4, vLLM TP2, 4 workers ------------
mkdir -p "$RES/qwen3.8-27b-nvfp4"
nohup bash -c "
  $RUN --label qwen3.8-27b-nvfp4 --exp baseline --seeds $P2 --workers 4
  $RUN --label qwen3.8-27b-nvfp4 --exp thinking --seeds $P2 --workers 4
  $RUN --label qwen3.8-27b-nvfp4 --exp sysprompt --recipes lasagna --prompt-ids 7,8,9,10,11,12,13,14,15,16,17,18,19,20 --seeds $P1 --workers 4
  $RUN --label qwen3.8-27b-nvfp4 --exp sysprompt --recipes spaghetti --seeds $P1 --workers 4
  echo 'PHASE2 GPU_HOST_E QUEUE COMPLETE'
" > "$RES/qwen3.8-27b-nvfp4/phase2.log" 2>&1 &
echo "gpu-host-e  (qwen3.8-27b-nvfp4)        PID $!  log $RES/qwen3.8-27b-nvfp4/phase2.log"

# ---- gpu-host-d .211:8006 — Qwen3.8-Flash-Next NVFP4, sglang, 4 workers -------
mkdir -p "$RES/qwen3.8-flash-next-nvfp4"
nohup bash -c "
  $RUN --label qwen3.8-flash-next-nvfp4 --exp baseline --seeds $P2 --workers 4
  $RUN --label qwen3.8-flash-next-nvfp4 --exp thinking --seeds $P2 --workers 4
  $RUN --label qwen3.8-flash-next-nvfp4 --exp sysprompt --recipes lasagna --prompt-ids 13,14,15,16,17,18,19,20 --seeds $P1 --workers 4
  $RUN --label qwen3.8-flash-next-nvfp4 --exp sysprompt --recipes spaghetti --seeds $P1 --workers 4
  echo 'PHASE2 GPU_HOST_D-8006 QUEUE COMPLETE'
" > "$RES/qwen3.8-flash-next-nvfp4/phase2.log" 2>&1 &
echo "gpu-host-d  (qwen3.8-flash-next :8006) PID $!  log $RES/qwen3.8-flash-next-nvfp4/phase2.log"

# ---- gpu-host-d .211:8005 — Qwen3.6-35B NVFP4 replicate, vLLM, 8 workers ------
mkdir -p "$RES/qwen3.6-35b-nvfp4-gpu-host-d"
nohup bash -c "
  $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --exp thinking --seeds $ALL --workers 8
  $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --exp temperature --workers 8
  echo 'PHASE2 GPU_HOST_D-8005 QUEUE COMPLETE'
" > "$RES/qwen3.6-35b-nvfp4-gpu-host-d/phase2.log" 2>&1 &
echo "gpu-host-d  (qwen3.6-35b-nvfp4 :8005)  PID $!  log $RES/qwen3.6-35b-nvfp4-gpu-host-d/phase2.log"

# ---- gpu-host-b .209:8080 — llama.cpp, STRICTLY SERIAL, 1 worker, ONE process ---
# Waits for the still-running Phase 1 gpu-host-b queue to exit first. Each label
# switch is a llama.cpp cold-swap (~2 min). Expected to run past the night.
mkdir -p "$RES/qwen3.8-27b-q4kxl" "$RES/qwen3.8-27b-q4kxl-mtp" \
         "$RES/qwen3.6-35b-q4kxl" "$RES/qwen3.6-27b-q4kxl"
# Capture the Phase 1 gpu-host-b PIDs now, in the parent shell, so the waiter cannot
# match its own command line with pgrep.
P1PIDS=$(pgrep -f 'run.py --label qwen3.8-27b-q4kxl' | tr '\n' ' ')
echo "gpu-host-b    phase1 PIDs to wait on: ${P1PIDS:-<none>}"

nohup bash -c "
  echo \"[\$(date -u +%FT%TZ)] waiting for the Phase 1 gpu-host-b queue (PIDs: ${P1PIDS:-none}) ...\"
  while true; do
    if tail -3 '$RES/gpu_host_b_phase1.log' | grep -q '^Done:'; then
      echo \"[\$(date -u +%FT%TZ)] phase1 log reports Done\"; break
    fi
    alive=0
    for pid in ${P1PIDS:-}; do kill -0 \"\$pid\" 2>/dev/null && alive=1; done
    if [ \"\$alive\" -eq 0 ]; then
      echo \"[\$(date -u +%FT%TZ)] no phase1 gpu-host-b process remains\"; break
    fi
    sleep 60
  done
  sleep 30
  echo \"[\$(date -u +%FT%TZ)] starting Phase 2 gpu-host-b queue\"
  $RUN --label qwen3.8-27b-q4kxl     --exp baseline --seeds $P2  --workers 1
  $RUN --label qwen3.8-27b-q4kxl-mtp --exp baseline --seeds $ALL --workers 1
  $RUN --label qwen3.6-35b-q4kxl     --exp baseline --seeds $ALL --workers 1
  $RUN --label qwen3.6-27b-q4kxl     --exp baseline --seeds $ALL --workers 1
  echo 'PHASE2 GPU_HOST_B QUEUE COMPLETE'
" > "$RES/gpu_host_b_phase2.log" 2>&1 &
echo "gpu-host-b    (serial, 1 worker)         PID $!  log $RES/gpu_host_b_phase2.log"
