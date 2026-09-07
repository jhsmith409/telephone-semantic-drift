#!/bin/bash
# Paper 3, 16-domain breadth prompt set (scripts/paper3/domain_prompts.py).
# One subshell per host; each waits for that host's concrete queue
# (launch_concrete.sh) to drain before starting.  Launched via nohup.
RUN="uv run python -u /home/jhsmith/telephone/scripts/paper3/run.py"
RES=/home/jhsmith/telephone/results/paper3
ALL15=42,137,256,512,1024,3,7,11,19,23,29,31,37,41,47
P1=42,137,256,512,1024
cd /home/jhsmith/telephone

# wait_for <label> : block until no run.py is running for that exact label
wait_for () {
  while pgrep -f "run\.py --label $1 " > /dev/null; do sleep 30; done
}

# gpu-host-f (<GPU-HOST-C>:8005) Qwen3.6-35B
( wait_for qwen3.6-35b-nvfp4
  $RUN --label qwen3.6-35b-nvfp4 --domain breadth --exp baseline --seeds $ALL15 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4/breadth_baseline.log 2>&1
  $RUN --label qwen3.6-35b-nvfp4 --domain breadth --exp sysprompt --prompt-ids 4 \
      --seeds $P1 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4/breadth_sysprompt_p04.log 2>&1
  echo "BREADTH GPU_HOST_F COMPLETE" >> $RES/qwen3.6-35b-nvfp4/breadth_sysprompt_p04.log ) &

# gpu-host-d (<GPU-HOST-D>:8005) Qwen3.6-35B replicate
( wait_for qwen3.6-35b-nvfp4-gpu-host-d
  $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --domain breadth --exp thinking --seeds $P1 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4-gpu-host-d/breadth_thinking.log 2>&1
  $RUN --label qwen3.6-35b-nvfp4-gpu-host-d --domain breadth --exp baseline --seeds $ALL15 --workers 8 \
      > $RES/qwen3.6-35b-nvfp4-gpu-host-d/breadth_baseline.log 2>&1
  echo "BREADTH GPU_HOST_D-8005 COMPLETE" >> $RES/qwen3.6-35b-nvfp4-gpu-host-d/breadth_baseline.log ) &

wait
