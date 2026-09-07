#!/bin/bash
# Paper 3, second knowledge domain (construction SOW / CSI MasterFormat).
# Each host subshell first waits for that host's Phase 2c jobs to drain, then
# runs its concrete-domain commands in order.  Launched via nohup.
RUN="uv run python -u /home/jhsmith/telephone/scripts/paper3/run.py"
RES=/home/jhsmith/telephone/results/paper3
ALL15=42,137,256,512,1024,3,7,11,19,23,29,31,37,41,47
P1=42,137,256,512,1024
NEUTRAL=1,2,4,10,17,20
cd /home/jhsmith/telephone

# wait_for <label> : block until no run.py is running for that exact label
wait_for () {
  while pgrep -f "run\.py --label $1 " > /dev/null; do sleep 30; done
}

# gpu-host-f (<GPU-HOST-C>:8005) Qwen3.6-35B
( wait_for qwen3.6-35b-nvfp4
  $RUN --label qwen3.6-35b-nvfp4 --domain concrete --exp baseline --seeds $ALL15 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4/concrete_baseline.log 2>&1
  $RUN --label qwen3.6-35b-nvfp4 --domain concrete --exp thinking --seeds $ALL15 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4/concrete_thinking.log 2>&1
  echo "CONCRETE GPU_HOST_F COMPLETE" >> $RES/qwen3.6-35b-nvfp4/concrete_thinking.log ) &

# gpu-host-d (<GPU-HOST-D>:8005) Qwen3.6-35B replicate
( wait_for qwen3.6-35b-nvfp4-gpu-host-d
  $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --domain concrete --exp sysprompt --recipes concrete_long \
      --prompt-ids $NEUTRAL --seeds $P1 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4-gpu-host-d/concrete_sysprompt_long.log 2>&1
  $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --domain concrete --exp sysprompt --recipes concrete_short \
      --prompt-ids $NEUTRAL --seeds $P1 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4-gpu-host-d/concrete_sysprompt_short.log 2>&1
  $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --domain concrete --exp baseline --seeds $ALL15 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4-gpu-host-d/concrete_baseline.log 2>&1
  echo "CONCRETE GPU_HOST_D-8005 COMPLETE" >> $RES/qwen3.6-35b-nvfp4-gpu-host-d/concrete_baseline.log ) &

# gpu-host-e (<GPU-HOST-A>:8005) Qwen3.8-27B
( wait_for qwen3.8-27b-nvfp4
  $RUN --label qwen3.8-27b-nvfp4 --domain concrete --exp baseline --seeds $ALL15 --workers 4 \
      > $RES/qwen3.8-27b-nvfp4/concrete_baseline.log 2>&1
  $RUN --label qwen3.8-27b-nvfp4 --domain concrete --exp thinking --seeds $ALL15 --workers 4 \
      > $RES/qwen3.8-27b-nvfp4/concrete_thinking.log 2>&1
  echo "CONCRETE GPU_HOST_E COMPLETE" >> $RES/qwen3.8-27b-nvfp4/concrete_thinking.log ) &

# gpu-host-d (<GPU-HOST-D>:8006) Qwen3.8-Flash-Next
( wait_for qwen3.8-flash-next-nvfp4
  $RUN --label qwen3.8-flash-next-nvfp4 --domain concrete --exp baseline --seeds $ALL15 --workers 4 \
      > $RES/qwen3.8-flash-next-nvfp4/concrete_baseline.log 2>&1
  $RUN --label qwen3.8-flash-next-nvfp4 --domain concrete --exp thinking --seeds $ALL15 --workers 4 \
      > $RES/qwen3.8-flash-next-nvfp4/concrete_thinking.log 2>&1
  echo "CONCRETE GPU_HOST_D-8006 COMPLETE" >> $RES/qwen3.8-flash-next-nvfp4/concrete_thinking.log ) &

wait
