# telephone-semantic-drift

Research code, data-generation harness, analysis pipeline and paper sources for a
study of **semantic drift in iterated LLM paraphrase chains** — the "telephone game"
played by language models.

A short seed message (for example a recipe) is handed to a model with the
instruction *"Paraphrase this to the next agent"*. The model's output becomes the
next iteration's input, for 30 iterations. After each hop the output is embedded and
compared by cosine similarity against the **original** message. The resulting
trajectory tells you whether a model/serving configuration converges to a stable
fixed point, decays slowly, or runs away into unrelated content.

The repository covers model architecture and scale, quantization format, serving
backend, sampling temperature, system-prompt engineering, and (in Paper B)
run-to-run reproducibility, reasoning ("thinking") mode, speculative decoding,
knowledge domain, and whether cosine similarity tracks factual survival at all.

## Papers

| Path | Paper |
| --- | --- |
| `paper/` | **Paper A** — *Semantic Drift in Iterated LLM Paraphrase Chains: Model Architecture, Scale, Quantization, Serving Infrastructure, and Prompt Engineering.* The current, consolidated paper. Sources: `paper/main.tex`, `paper/references.bib`, `paper/tables/*.tex`, compiled `paper/main.pdf`. |
| `paper2/` | Archival source for the earlier Qwen3.5-family paper, **merged into `paper/`**. Kept for provenance; not maintained. |
| `paper3/` | **Paper B** — *Semantic Drift Across Qwen Generations: Reproducibility, Thinking Mode, Serving Stack, and What Cosine Similarity Misses in Iterated Paraphrase Chains.* Sources: `paper3/main.tex`, `paper3/references.bib`, `paper3/tables/*.tex`, compiled `paper3/main.pdf`. |

Build a paper with `latexmk`:

```bash
cd paper && latexmk -pdf main.tex
```

### Paper B

**Semantic Drift Across Qwen Generations: Reproducibility, Thinking Mode,
Serving Stack, and What Cosine Similarity Misses in Iterated Paraphrase Chains**
(`paper3/main.tex`) extends the study to the Qwen3.6 and Qwen3.8 generations.
Across 2,662 completed chains and 85,460 model calls on eight configurations
served by vLLM, SGLang and llama.cpp, it asks whether a sampling seed reproduces
a chain at all (it does not, even at temperature 0), what reasoning ("thinking")
mode does to drift, whether speculative decoding changes it, how drift behaves
across sixteen knowledge domains, and whether cosine similarity actually tracks
the loss of information practitioners care about (it tracks genre and length
change far more).

Build it, and regenerate its figures and tables, with:

```bash
uv run python scripts/analyze_paper3.py   # -> results/paper3_figures/, paper3/tables/
cd paper3 && latexmk -pdf main.tex
```

`scripts/analyze_paper3.py` reads `results/paper3/<label>/*.json` plus the Paper A
anchor datasets, so unpack the data archive at the repository root first (see
[`results/README.md`](results/README.md)) and run it from the repository root.
It also writes `results/paper3_figures/all_stats.json` (every statistic quoted in
the paper) and `results/paper3_figures/summary_for_text.md` (a human-readable
digest, including the data anomalies it detected).

## Repository layout

```
src/telephone/       Flask web app + batch API (the chain engine)
tests/               pytest suite for the app
scripts/paper3/      current experiment runner (run.py, common.py, prompt sets)
scripts/analyze_paper.py         Paper A analysis: figures + tables
scripts/analyze_paper3.py        Paper B analysis: figures + tables
scripts/analyze_qwen35_paper.py  Qwen3.5 analysis: figures + tables
scripts/legacy/      Paper 1/2 run scripts (see scripts/legacy/README.md)
paper/ paper2/ paper3/           LaTeX sources
results/             figures and validation stats (full data: see results/README.md)
```

## Running the web app

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env      # point AI_HOSTS / EMBEDDING_HOST at your own servers
uv run python -m telephone.main        # web UI on port 5000
uv run python -m telephone.batch_api   # batch JSON API on port 5001
```

Or via Docker (`docker compose up`) — both services, `network_mode: host`.

`AI_HOSTS` is a comma-separated list of hosts to probe: Ollama is detected on port
11434 and OpenAI-compatible servers (vLLM, SGLang, llama.cpp) on ports 8000-8010.
`EMBEDDING_HOST`/`EMBEDDING_PORT`/`EMBEDDING_MODEL` name an OpenAI-compatible
`/v1/embeddings` endpoint used for the similarity scores.

Tests:

```bash
uv run pytest
```

## Running a drift chain

`scripts/paper3/run.py` drives chains against **any OpenAI-compatible endpoint**.
Endpoints are declared in the `ENDPOINTS` table at the top of
`scripts/paper3/common.py` as `Endpoint(label, host, port, model, backend, max_workers)`.
The hosts shipped here are placeholders (`<GPU-HOST-A>` etc.) because the original
lab addresses were removed — edit that table to point at your own server, or add a
new entry:

```python
ENDPOINTS["my-model"] = Endpoint(
    "my-model", "127.0.0.1", 8000, "Qwen/Qwen3-8B", "vllm", 4)
```

Then:

```bash
# 5 seeds x 2 recipes, 30 iterations, T=0.7
uv run python -u scripts/paper3/run.py --label my-model \
    --exp baseline --recipes spaghetti,lasagna --workers 4

# other experiment arms
uv run python -u scripts/paper3/run.py --label my-model --exp thinking   ...
uv run python -u scripts/paper3/run.py --label my-model --exp temperature ...
uv run python -u scripts/paper3/run.py --label my-model --exp sysprompt   ...
```

`scripts/paper3/smoke.py` runs a short single chain to check an endpoint first.
Runs are resumable: completed conditions are detected in the output JSON and skipped.
Output lands in `results/paper3/<label>/<exp>.json`, in the same schema as Papers 1-2
(`{"models": {condition_key: [iteration records...]}, "timestamp": ...}`).

## Regenerating figures and tables

Both analysis scripts read data with **repo-root-relative** paths, so run them from
the repository root after unpacking the data archive (see `results/README.md`):

```bash
uv run python scripts/analyze_paper.py         # -> results/paper_figures/, paper/tables/
uv run python scripts/analyze_paper3.py        # -> results/paper3_figures/, paper3/tables/
uv run python scripts/analyze_qwen35_paper.py  # -> results/qwen35_paper_figures/, paper2/tables/
```

Then recompile the papers with `latexmk -pdf main.tex`.

## Data

`results/` in this repository carries only the published figures and the embedding
validation statistics. The **full chain data — every iteration's input and output
text for every condition — is ~970 MB and is attached to the GitHub Release
[`v1.1-data`](https://github.com/jhsmith409/telephone-semantic-drift/releases/tag/v1.1-data)**,
which supersedes `v1.0-data` and adds the complete Paper B runs. See
[`results/README.md`](results/README.md) for the download and checksum.

## A note on hostnames

This work ran on a private lab network. Every RFC1918 address and internal hostname
has been replaced with a stable placeholder (`<GPU-HOST-A>`, `gpu-host-b`, ...) in
the code, the notes and the released data. Nothing in this repository points at a
reachable host; substitute your own endpoints.

## License

MIT — see [LICENSE](LICENSE).

## Use of AI assistance

The majority of the code in this repository (experiment runners, analysis scripts,
and the web application) was written by large language models — Claude Opus and
Claude Sonnet (Anthropic) and Grok (xAI) — working under the author's direction,
with the author specifying the experiments, reviewing the code and its outputs,
and correcting errors. Claude Fable and Claude Opus, Grok, and Gemini (Google)
were used to aid the writing and layout of the two manuscripts. The experimental
design, the interpretation of results, and all claims in the papers are the
author's responsibility.

Copyright (c) 2026 James H. Smith
