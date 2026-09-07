# Legacy experiment scripts (Papers 1 and 2)

These are the original run scripts that produced the data for Paper 1 and Paper 2
(now consolidated into `paper/`). They are kept **as they were run**, for
reproducibility and provenance, not as a maintained interface.

Three things to know before using them:

1. **They were written against a private lab network.** Server addresses were
   hard-coded. Every RFC1918 address and internal hostname has been replaced by a
   placeholder (`<GPU-HOST-A>`, `<GPU-HOST-B>`, `<GPU-HOST-C>`, `<GPU-HOST-D>`,
   `<EMBED-HOST>`, `gpu-host-b`, ...). None of them will connect as written — edit
   the endpoint constants at the top of the script to your own server first.

2. **They are historically accurate, not tidy.** One script per experiment arm,
   with reruns, timeout-recovery and per-backend variants accumulated over time
   (`run_qwen35_rerun_*`, `run_qwen35_vllm_*`, `run_nemo_30b*`, ...). Duplication
   between them is intentional: each was frozen once its data was collected.

3. **They are superseded by `scripts/paper3/`.** The current runner
   (`scripts/paper3/run.py` + `common.py`) unifies all of this behind one CLI with a
   table-driven endpoint registry, resumable runs and a shared chain implementation,
   and writes the same on-disk schema. Use it for new work.

`scripts/paper3/common.py` still imports the prompt and system-prompt definitions
from `run_qwen35_nvfp4_sysprompt.py` in this directory rather than retyping them, so
this directory is a live dependency of the current runner — do not delete it.

Paths inside these scripts are relative to the repository root; run them from there
(`uv run python scripts/legacy/<script>.py`).
