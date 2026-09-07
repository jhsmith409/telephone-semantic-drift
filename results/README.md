# Results

## What is in the repository

Only the small, published artefacts are committed here:

| Path | Contents |
| --- | --- |
| `paper_figures/` | Figures for Paper A (`paper/main.tex`) |
| `paper3_figures/` | Figures for Paper B (`paper3/main.tex`), plus `all_stats.json` (every statistic quoted in the paper) and `summary_for_text.md` |
| `qwen35_paper_figures/` | Figures for the Qwen3.5 paper (`paper2/main.tex`) |
| `embedding_validation/` | bge-m3 vs. Qwen3-Embedding agreement check for Papers 1: `correlation_stats.json`, `raw_similarities.json`, `embedding_scatter.png` |
| `embedding_validation_qwen35/` | The same validation for the Qwen3.5 data |

## What is in the Release

The **full chain data — every iteration's input text, output text, similarity score
and timing, for every condition in every experiment** — is roughly 970 MB
uncompressed across 39 experiment directories and does not belong in git. It is
attached to the GitHub Release [`v1.1-data`](https://github.com/jhsmith409/telephone-semantic-drift/releases/tag/v1.1-data):

```bash
gh release download v1.1-data \
  --repo jhsmith409/telephone-semantic-drift \
  --pattern 'telephone-results-*.tar.gz'

sha256sum -c <<< "b042dfbe46d9b7d38e2b87a75abd944242ebed1c7f85944200ceb4d889256c22  telephone-results-2026-09-07-v1.1.tar.gz"

tar -xzf telephone-results-2026-09-07-v1.1.tar.gz   # unpacks into ./results/
```

`v1.1-data` supersedes `v1.0-data`: it is rebuilt from the finished results tree,
so `results/paper3/` is now **complete** rather than a mid-run snapshot. The older
[`v1.0-data`](https://github.com/jhsmith409/telephone-semantic-drift/releases/tag/v1.0-data)
release is left in place for anyone who cited it.

Unpack it at the repository root — it expands to `results/`, merging with (and
overwriting) the figure directories already committed here. Both analysis scripts
then run from the repository root:

```bash
uv run python scripts/analyze_paper.py
uv run python scripts/analyze_paper3.py
uv run python scripts/analyze_qwen35_paper.py
```

### Asset

| | |
| --- | --- |
| File | `telephone-results-2026-09-07-v1.1.tar.gz` |
| Size | 60,305,192 bytes (58 MiB compressed, ~970 MB unpacked) |
| SHA-256 | `b042dfbe46d9b7d38e2b87a75abd944242ebed1c7f85944200ceb4d889256c22` |
| Files | 152 across 39 experiment directories |

### Data format

Every experiment writes one or more `drift_data.json` (or `<exp>.json`) files with
the shape:

```json
{"models": {"<condition_key>": [ {iteration record}, ... ]}, "timestamp": "..."}
```

A condition key encodes the model label, the seed prompt ("recipe"), and the arm
under test (seed, temperature, system prompt, thinking mode).

### Notes

- `results/paper3/` is **complete** as of `v1.1-data` — it holds every condition
  behind Paper B (2,662 chains, 85,460 model calls, eight configurations). In
  `v1.0-data` it was only a mid-run checkpoint.
- Run logs (`*.log`, `*.out`) were excluded from both the repository and the archive.
- All RFC1918 addresses and internal lab hostnames were replaced by placeholders
  (`<GPU-HOST-A>`, `gpu-host-d`, ...) inside the archived JSON as well as in the
  code. This affects a handful of condition keys, which carry the placeholder name
  rather than the original host name — notably the Paper B replication host, whose
  directory and condition keys read `qwen3.6-35b-nvfp4-gpu-host-d`.
  `scripts/analyze_paper3.py` expects exactly that placeholder name.


## v1.2-data (2026-09-07, addendum)

Sampler A/B for runaway thinking (Qwen3.6-35B NVFP4, thinking on): `telephone-results-2026-09-07-v1.2-sampler.tar.gz` contains `results/paper3/qwen3.6-35b-nvfp4/sampler_bare_p11.json`, `sampler_vendor_p11.json`, `sampler_bare_spaghetti.json` and the updated `sysprompt_think.json` (prompt 11 topped up to 15 chains). Unpack over the v1.1 tree. SHA-256 `6f40420f9d89232e59a4e0b90633b62528f518667e5141df4ef78325b145ab19`.
