#!/bin/bash
# Copyright (c) 2026 James H. Smith. MIT License.
# Sampler A/B for runaway thinking, Qwen3.6-35B NVFP4 on host A, thinking ON (2026-09-07)
RUN="uv run python -u /home/jhsmith/telephone/scripts/paper3/run.py --label qwen3.6-35b-nvfp4 --workers 8"
cd /home/jhsmith/telephone
# default sampler: top up prompt 11 lasagna to 15 chains (appends to sysprompt_think.json)
$RUN --exp sysprompt --thinking --recipes lasagna --prompt-ids 11 --seeds 3,7,11,19,23,29,31,37,41,47 --out-name sysprompt_think > /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4/sampler_default_p11.log 2>&1
# bare sampler
PAPER3_SAMPLER='{"top_p":1.0,"top_k":-1,"presence_penalty":0.0}' $RUN --exp sysprompt --thinking --recipes lasagna --prompt-ids 11 --seeds 42,137,256,512,1024,3,7,11,19,23,29,31,37,41,47 --out-name sampler_bare_p11 > /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4/sampler_bare_p11.log 2>&1
# vendor thinking-mode sampler
PAPER3_SAMPLER='{"top_p":0.95,"top_k":20,"presence_penalty":1.5}' $RUN --exp sysprompt --thinking --recipes lasagna --prompt-ids 11 --seeds 42,137,256,512,1024,3,7,11,19,23,29,31,37,41,47 --out-name sampler_vendor_p11 > /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4/sampler_vendor_p11.log 2>&1
# bare sampler on plain spaghetti (base rate zero under default)
PAPER3_SAMPLER='{"top_p":1.0,"top_k":-1,"presence_penalty":0.0}' $RUN --exp thinking --recipes spaghetti --seeds 42,137,256,512,1024,3,7,11,19,23,29,31,37,41,47 --out-name sampler_bare_spaghetti > /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4/sampler_bare_spaghetti.log 2>&1
echo "SAMPLER AB COMPLETE" >> /home/jhsmith/telephone/results/paper3/qwen3.6-35b-nvfp4/sampler_bare_spaghetti.log
