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
backend, sampling temperature, system-prompt engineering, and (in the current
Paper 3 work) knowledge domain and thinking mode.

## Papers

| Path | Paper |
| --- | --- |
| `paper/` | **Paper A** — *Semantic Drift in Iterated LLM Paraphrase Chains: Model Architecture, Scale, Quantization, Serving Infrastructure, and Prompt Engineering.* The current, consolidated paper. Sources: `paper/main.tex`, `paper/references.bib`, `paper/tables/*.tex`, compiled `paper/main.pdf`. |
| `paper2/` | Archival source for the earlier Qwen3.5-family paper, **merged into `paper/`**. Kept for provenance; not maintained. |
| `paper3/` | Working material for the in-progress follow-up (`moved_from_paper2.tex`). |

Build a paper with `latexmk`:

```bash
cd paper && latexmk -pdf main.tex
```

## Repository layout

```
src/telephone/       Flask web app + batch API (the chain engine)
tests/               pytest suite for the app
scripts/paper3/      current experiment runner (run.py, common.py, prompt sets)
scripts/analyze_paper.py         Paper A analysis: figures + tables
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
uv run python scripts/analyze_qwen35_paper.py  # -> results/qwen35_paper_figures/, paper2/tables/
```

Then recompile the papers with `latexmk -pdf main.tex`.

## Data

`results/` in this repository carries only the published figures and the embedding
validation statistics. The **full chain data — every iteration's input and output
text for every condition — is ~1 GB and is attached to the GitHub Release**; see
[`results/README.md`](results/README.md) for the download and checksum.

## A note on hostnames

This work ran on a private lab network. Every RFC1918 address and internal hostname
has been replaced with a stable placeholder (`<GPU-HOST-A>`, `gpu-host-b`, ...) in
the code, the notes and the released data. Nothing in this repository points at a
reachable host; substitute your own endpoints.

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 James H. Smith
