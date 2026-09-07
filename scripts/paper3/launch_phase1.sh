#!/usr/bin/env bash
# Paper 3 Phase 1 launcher. One background queue per host; the hosts run in parallel.
# Budgets below are derived from the 2026-09-06 smoke measurements (see report),
# targeting ~3.5 h wall-clock per host. Everything is resume-safe, so a cut-off
# mid-queue is harmless.
#
#   bash /home/jhsmith/telephone/scripts/paper3/launch_phase1.sh

set -u
REPO=/home/jhsmith/telephone
RUN="uv run python -u $REPO/scripts/paper3/run.py"
RES=$REPO/results/paper3

cd "$REPO" || exit 1

# ---- gpu-host-f .210:8005 — Qwen3.6-35B-A3B NVFP4, vLLM, 8 workers -------------
# smoke: spaghetti 0.8 s/iter, lasagna 2.8 s/iter. Entire queue projects to ~1.2 h.
mkdir -p "$RES/qwen3.6-35b-nvfp4"
nohup bash -c "
  $RUN --label qwen3.6-35b-nvfp4 --exp baseline    --workers 8
  $RUN --label qwen3.6-35b-nvfp4 --exp thinking    --workers 8
  $RUN --label qwen3.6-35b-nvfp4 --exp temperature --workers 8
  $RUN --label qwen3.6-35b-nvfp4 --exp sysprompt --recipes lasagna --workers 8
" > "$RES/qwen3.6-35b-nvfp4/phase1.log" 2>&1 &
echo "gpu-host-f  (qwen3.6-35b-nvfp4)      PID $!  log $RES/qwen3.6-35b-nvfp4/phase1.log"

# ---- gpu-host-e .208:8005 — Qwen3.8-27B NVFP4, vLLM TP2, 4 workers ------------
# smoke: spaghetti 4.5 s/iter, lasagna 14.4 s/iter (slowest GPU host).
# baseline+thinking+temperature ~1.9 h; sysprompt limited to prompts 1-6 (~1.6 h).
mkdir -p "$RES/qwen3.8-27b-nvfp4"
nohup bash -c "
  $RUN --label qwen3.8-27b-nvfp4 --exp baseline    --workers 4
  $RUN --label qwen3.8-27b-nvfp4 --exp thinking    --workers 4
  $RUN --label qwen3.8-27b-nvfp4 --exp temperature --workers 4
  $RUN --label qwen3.8-27b-nvfp4 --exp sysprompt --recipes lasagna --prompt-ids 1,2,3,4,5,6 --workers 4
" > "$RES/qwen3.8-27b-nvfp4/phase1.log" 2>&1 &
echo "gpu-host-e (qwen3.8-27b-nvfp4)      PID $!  log $RES/qwen3.8-27b-nvfp4/phase1.log"

# ---- gpu-host-d .211:8006 — Qwen3.8-Flash-Next NVFP4, sglang, 4 workers -------
# smoke: spaghetti 2.7 s/iter, lasagna 8.1 s/iter; thinking works here (rc>0).
# baseline+thinking+temperature ~1.3 h; sysprompt limited to prompts 1-12 (~1.8 h).
mkdir -p "$RES/qwen3.8-flash-next-nvfp4"
nohup bash -c "
  $RUN --label qwen3.8-flash-next-nvfp4 --exp baseline    --workers 4
  $RUN --label qwen3.8-flash-next-nvfp4 --exp thinking    --workers 4
  $RUN --label qwen3.8-flash-next-nvfp4 --exp temperature --workers 4
  $RUN --label qwen3.8-flash-next-nvfp4 --exp sysprompt --recipes lasagna --prompt-ids 1,2,3,4,5,6,7,8,9,10,11,12 --workers 4
" > "$RES/qwen3.8-flash-next-nvfp4/phase1.log" 2>&1 &
echo "gpu-host-d (qwen3.8-flash-next)     PID $!  log $RES/qwen3.8-flash-next-nvfp4/phase1.log"

# ---- gpu-host-b .209:8080 — llama.cpp, STRICTLY SERIAL, 1 worker ----------------
# smoke: qwen3.8-27b spaghetti 2.4 s/iter (~1.2 min/chain).
# Order minimises nothing but follows the spec priority: the clean MTP spaghetti
# pair first, then lasagna as filler. Each label switch is a ~2 min cold load.
mkdir -p "$RES/qwen3.8-27b-q4kxl" "$RES/qwen3.8-27b-q4kxl-mtp"
nohup bash -c "
  $RUN --label qwen3.8-27b-q4kxl     --exp baseline --recipes spaghetti --workers 1
  $RUN --label qwen3.8-27b-q4kxl-mtp --exp baseline --recipes spaghetti --workers 1
  $RUN --label qwen3.8-27b-q4kxl     --exp baseline --recipes lasagna   --workers 1
  $RUN --label qwen3.8-27b-q4kxl-mtp --exp baseline --recipes lasagna   --workers 1
" > "$RES/gpu-host-b_phase1.log" 2>&1 &
echo "gpu-host-b   (qwen3.8-27b + mtp)      PID $!  log $RES/gpu-host-b_phase1.log"
