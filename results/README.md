# Results

## What is in the repository

Only the small, published artefacts are committed here:

| Path | Contents |
| --- | --- |
| `paper_figures/` | Figures for Paper A (`paper/main.tex`) |
| `qwen35_paper_figures/` | Figures for the Qwen3.5 paper (`paper2/main.tex`) |
| `embedding_validation/` | bge-m3 vs. Qwen3-Embedding agreement check for Papers 1: `correlation_stats.json`, `raw_similarities.json`, `embedding_scatter.png` |
| `embedding_validation_qwen35/` | The same validation for the Qwen3.5 data |

## What is in the Release

The **full chain data — every iteration's input text, output text, similarity score
and timing, for every condition in every experiment** — is roughly 920 MB
uncompressed across 33 experiment directories and does not belong in git. It is
attached to the GitHub Release [`v1.0-data`](https://github.com/jhsmith409/telephone-semantic-drift/releases/tag/v1.0-data):

```bash
gh release download v1.0-data \
  --repo jhsmith409/telephone-semantic-drift \
  --pattern 'telephone-results-*.tar.gz'

sha256sum -c <<< "b9c24a5bd1d1b8d23873039c48c04998c79c80b18fdb74417c4981100f61453b  telephone-results-2026-09-07.tar.gz"

tar -xzf telephone-results-2026-09-07.tar.gz   # unpacks into ./results/
```

Unpack it at the repository root — it expands to `results/`, merging with (and
overwriting) the figure directories already committed here. Both analysis scripts
then run from the repository root:

```bash
uv run python scripts/analyze_paper.py
uv run python scripts/analyze_qwen35_paper.py
```

### Asset

| | |
| --- | --- |
| File | `telephone-results-2026-09-07.tar.gz` |
| Size | 56,228,417 bytes (54 MiB compressed, ~920 MB unpacked) |
| SHA-256 | `b9c24a5bd1d1b8d23873039c48c04998c79c80b18fdb74417c4981100f61453b` |
| Files | 138 across 33 experiment directories |

### Data format

Every experiment writes one or more `drift_data.json` (or `<exp>.json`) files with
the shape:

```json
{"models": {"<condition_key>": [ {iteration record}, ... ]}, "timestamp": "..."}
```

A condition key encodes the model label, the seed prompt ("recipe"), and the arm
under test (seed, temperature, system prompt, thinking mode).

### Notes

- `results/paper3/` is a **snapshot taken while the Paper 3 runs were still in
  progress**; some conditions in it are incomplete. The runner is resumable and
  skips conditions already present, so the file is a valid checkpoint, not a final
  dataset.
- Run logs (`*.log`) were excluded from both the repository and the archive.
- All RFC1918 addresses and internal lab hostnames were replaced by placeholders
  (`<GPU-HOST-A>`, `gpu-host-d`, ...) inside the archived JSON as well as in the
  code. This affects a handful of condition keys, which carry the placeholder name
  rather than the original host name.
