# Telephone — AI Telephone Game

## Build & Run

Requires Python >=3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                    # install deps
uv run python -m telephone.main            # web GUI on port 5000
uv run python -m telephone.batch_api       # batch API on port 5001
docker compose up                          # both services via Docker
```

## Testing

```bash
uv run pytest                                       # all tests
uv run pytest tests/test_routes.py                  # single file
uv run pytest tests/test_routes.py::test_name       # single test
```

No linter or formatter is configured.

## Architecture

Three-tier: Web UI (port 5000), Batch API (port 5001), external AI services (Ollama / OpenAI-compatible).

- **`game_logic.py`** — `run_chain()` generator pipes messages through agents sequentially, yielding a `RelayStep` after each. Five prompt styles: repeat, paraphrase, exaggerate, improv, custom.
- **`routes.py`** — streams `RelayStep`s as SSE events to the browser; computes cosine similarity to the original after each step.
- **`discovery.py`** — reads `AI_HOSTS` from `.env`, scans Ollama (port 11434) and OpenAI-compatible (ports 8000-8010) concurrently.
- **`ai_client.py`** — unified client abstracting Ollama and OpenAI APIs (120s timeout, 2 retries).
- **`embeddings.py`** — cosine similarity via external embedding service; gracefully returns `None` when unavailable.
- **`batch_api.py`** — single-agent JSON in/out (no SSE), used by drift experiment scripts.
- **Frontend** — single-page app (`templates/index.html`, `static/app.js`, `static/style.css`): dark theme, vanilla JS, Canvas 2D similarity chart.

## Configuration

`.env` file at project root:

| Variable | Purpose |
|---|---|
| `AI_HOSTS` | Comma-separated host IPs to scan for AI services |
| `EMBEDDING_HOST` | Embedding service host |
| `EMBEDDING_PORT` | Embedding service port |
| `EMBEDDING_MODEL` | Embedding model name |
| `APP_API_KEY` | API key for batch endpoint auth |

Key limits: 30 max agents, 120s AI timeout, 1MB max content, 10k char message, 1k char custom prompt.

---

# Project Notes

## Batch Testing API (completed)

Added a batch testing API for automated LLM experiments:
- `src/telephone/embeddings.py` — embedding client + cosine similarity via `Qwen/Qwen3-Embedding-0.6B` on `<GPU-HOST-A>:8002`
- `src/telephone/batch_api.py` — Flask app on port 5001 with `GET /api/services` and `POST /api/run`
- `scripts/run_batch_tests.py` — Cartesian-product test matrix runner
- `tests/test_embeddings.py` (10 tests), `tests/test_batch_api.py` (11 tests)
- Docker compose `batch` service added
- Max agents raised from 10 to 30; fixed hanging test in `test_routes.py` (mocked `run_chain`)
- `matplotlib` added as dependency for plotting

## Drift Experiments (completed)

30-iteration semantic drift experiment. Each model paraphrases the same prompt 30 times in sequence (output becomes next input), measuring cosine similarity to the original at each step.

- **Prompt:** "Step 1: Boil water. Step 2: Add pasta for 8 minutes. Step 3: Drain and serve with sauce."
- **Settings:** prompt_style=paraphrase, temperature=0.7
- **Embedding model:** Qwen/Qwen3-Embedding-0.6B on <GPU-HOST-A>:8002
- **All models on:** <GPU-HOST-D> (Ollama port 11434, vLLM port 8005)

### First experiment (4 models)
- Script: `scripts/run_drift_experiment.py`
- Models: gemma3:27b, qwen3-next, llama3.1:8b, gpt-oss:120b
- Results: `results/drift_experiment/drift_data.json`, `results/drift_experiment/drift_plot.png`

### Second experiment (16 additional models)
- Script: `scripts/run_drift_experiment_all.py`
- Skips the 4 already done, merges with previous data for combined plot
- All remaining ollama inference models + vLLM qwen3-vl-30b (AWQ)
- Results: `results/drift_experiment_all/drift_data_combined.json` (all 20 models merged)

### Final cleaned plot (17 models)
- Excluded: glm-ocr (vision-only, failed), bible-expert-12b (irrelevant), qwen3-vl:30b ollama-alt (duplicate tag)
- Renamed: qwen3-vl:30b ollama → "gguf", vllm → "awq" to clarify quantization difference
- Plot: `results/drift_experiment_all/drift_plot_final.png`
- Also copied to `/mnt/ssd/drift_plot_final.png`

### Key findings
- **Best:** qwen3:30b-instruct (0.91) — near-perfect orbit, instruction tuning dominates
- **Stable tier (≥0.80):** devstral-small-2:24b (0.84), devstral-2:123b (0.84), qwen3-vl:30b AWQ (0.83), gemma3:27b (0.82), qwen3:30b-thinking (0.80)
- **Catastrophic:** llama3.1:8b (0.20, cliff at iter 10), glm-4.7-flash (0.19), glm-4.5-air (0.20)
- **AWQ vs GGUF:** Same model (qwen3-vl:30b) — AWQ held flat at 0.83, GGUF drifted to 0.79 after iter 22
- **Drift is sudden, not gradual** — tipping-point dynamics (llama lost 0.60 similarity in one step)
- **Specialist models drift toward training distribution** (medgemma→clinical, coder→lists, qwen-next→agent framing)

### Quantization investigation
- Both ollama tags (`qwen3-vl:30b` and `qwen3-vl:30b-a3b-instruct-q4_K_M`) are identical: Q4_K_M GGUF, 31.1B params, same Modelfile params (temp=1, top_k=20, top_p=0.95)
- vLLM version: AWQ 4-bit (`cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit`), max_model_len=131072
- Ollama context_length=262144

## Qwen3 Quantization Drift Experiment (completed)

Systematic Qwen3 drift experiment: 6 parameter sizes × 3 quantization levels (fp16, q8_0, q4_K_M), plus thinking variants and qwen3-next:80b. Same settings as original drift experiments.

- **Script:** `scripts/run_drift_experiment_qwen3.py` — initial 20 models
- **Script:** `scripts/run_drift_experiment_qwen3_missing.py` — 2 missing models (30b-instruct-fp16, 30b-thinking-q8_0), merged into existing data
- **Results:** `results/drift_experiment_qwen3/drift_data.json` (22 models), `results/drift_experiment_qwen3/drift_plot.png`
- **All models on:** <GPU-HOST-D> (Ollama port 11434)

### 22 models tested
- **0.6B:** fp16, q8_0, q4_K_M
- **1.7B:** fp16, q8_0, q4_K_M
- **4B (instruct-2507):** fp16, q8_0, q4_K_M
- **8B:** fp16, q8_0, q4_K_M
- **30B MoE (A3B instruct-2507):** fp16, q8_0, q4_K_M
- **30B MoE (A3B thinking-2507):** fp16, q8_0, q4_K_M
- **32B dense:** fp16, q8_0, q4_K_M
- **80B MoE (qwen3-next):** q4_K_M

### Key findings
- **Best:** qwen3:4b-instruct-2507-q4_K_M (0.94) — small instruct model held tightest
- **Stable tier (≥0.80):** 8b-q4_K_M (0.90), 4b-q8_0 (0.90), 30b-instruct-q8_0 (0.89), 30b-instruct-fp16 (0.89), 4b-fp16 (0.89), 0.6b-q4_K_M (0.83), 30b-instruct-q4_K_M (0.81)
- **Catastrophic (<0.50):** 0.6b-fp16 (0.23), 0.6b-q8_0 (0.23), 1.7b-q8_0 (0.41), 1.7b-fp16 (0.47)
- **30B instruct row:** q8_0 (0.894) ≈ fp16 (0.889) >> q4_K_M (0.806) — heavy quantization hurts
- **30B thinking row:** q4_K_M (0.743) > fp16 (0.693) > q8_0 (0.663) — reversed ordering, thinking models behave unpredictably with quantization
- **Quantization is not monotonic** — higher precision does not always mean less drift; interaction with model architecture and training matters
- **Instruct tuning >> thinking tuning for stability** — instruct variants consistently outperform thinking variants at every quant level
- **Size is not monotonic either** — 4B instruct beat 8B, 30B, and 32B variants; 0.6b-q4_K_M (0.83) beat 32B-fp16 (0.68)

## Blog Post (published)

- **j8web post:** `_posts/2026-02-20-llm-telephone-game-semantic-drift.md`
- **Image:** `images/llm-telephone-drift/drift_plot_final.png` (resized to 869x542)
- **URL:** https://joshua8.ai/llm-telephone-game-semantic-drift/
- **References verified on arXiv:** Perez et al. 2025 (2407.04503, ICLR), Mohamed et al. 2025 (2502.20258, ACL), Rath 2026 (2601.04170)
- Removed all references to unpublished repo
- Also saved standalone versions to `/mnt/ssd/`: `drift_blog_post_v2.md`, `drift_results.md` (full table), `drift_plot_final.png`

## Live Similarity Chart (completed)

Added a real-time cosine similarity chart to the web GUI that updates as each agent completes:
- `src/telephone/routes.py` — calls `compute_similarity(original, output)` after each step, includes `cosine_similarity` in the SSE step event payload
- `templates/index.html` — `<canvas>` element between progress area and diff area
- `static/app.js` — `renderSimilarityChart()` using vanilla Canvas 2D (dark theme, blue line, gridlines, dot markers, value annotation); chart hidden gracefully when embedding service unavailable
- `static/style.css` — chart container styles matching existing dark theme
- `tests/test_routes.py` — `test_step_event_includes_cosine_similarity` (mocks `run_chain` + `compute_similarity`)
- UI agent count limit raised from 10 to 30 (matching server-side max)
- Removed redundant per-agent rows in "same model for all agents" mode
- Project synced to `gpu-host-e:~/telephone/`

## KV Cache Precision Comparison (completed)

Controlled experiment: 22 Qwen3 models tested under q8_0 KV cache vs fp16 KV cache (Ollama default is fp16; we explicitly set both). Same prompt, settings, and models as Qwen3 quantization experiment.

- **Script:** `scripts/analyze_kv_cache_comparison.py` — loads both datasets, computes deltas, generates 4 plots + console summary
- **Data sources:**
  - `results/drift_experiment_qwen3/drift_data.json` — q8_0 KV cache (22 models)
  - `results/drift_experiment_qwen3_fp16kv/drift_data.json` — fp16 KV cache (22 models)
- **Results:** `results/kv_cache_comparison/` — 4 plots + `summary_stats.json`
- **Blog images:** `../j8web/images/llm-telephone-drift-kv/` — blog-sized copies of all 4 plots

### 4 plots generated
| Plot | Filename | Description |
|---|---|---|
| Hero overlay | `drift_overlay_selected.png` | 7 selected models, dashed (q8 KV) vs solid (fp16 KV) |
| Delta bar chart | `delta_bar_chart.png` | All 22 models sorted by delta, green/red |
| Heatmap | `delta_heatmap.png` | Size × weight quant grid showing delta values |
| Small multiples | `all_models_comparison.png` | 22 subplots, one per model |

### Key findings
- **Biggest winner:** 0.6B fp16 weights (+0.62, from 0.23 to 0.84) — KV cache noise was causing catastrophic drift in tiny models
- **Biggest loser:** qwen3-next:80b q4_K_M (-0.36, from 0.79 to 0.43) — fp16 KV unmasked the agent-framing attractor
- **17 models improved**, 4 regressed, 1 neutral (mean delta +0.09)
- **q4_K_M weights are largely immune** to KV cache changes — weight quantization noise dominates
- **Instruct tuning buffers against infrastructure noise** — instruct variants consistently positive, thinking variants unpredictable
- **Non-monotonic interaction** — more precision does not always mean better; depends on model, weight quant, and training objective

## Blog Post Part 2 (published)

- **j8web post:** `_posts/2026-02-21-kv-cache-precision-semantic-drift-qwen3.md`
- **Images:** `images/llm-telephone-drift-kv/` (4 plots)
- **URL:** https://joshua8.ai/kv-cache-precision-semantic-drift-qwen3/
- **Part 1 link:** https://joshua8.ai/llm-telephone-game-semantic-drift/
- **Revised** with literature integration (Han et al. 2026 "Quantization Trap", KVQuant, KIVI, KVLinC), expanded limitations (PQT/QAT speculation, prompt engineering speculation), and condensed results narrative
- **References (all verified on arXiv):**
  - Perez et al. 2025 (arXiv:2407.04503, ICLR) — cultural attractors
  - Mohamed et al. 2025 (arXiv:2502.20258, ACL) — broken telephone distortion
  - Rath 2026 (arXiv:2601.04170) — agent drift / ASI
  - Han et al. 2026 (arXiv:2602.13595) — quantization trap / sequential amortization failure
  - Hooper et al. 2024 (KVQuant, NeurIPS) — KV cache quantization outliers
  - Liu et al. 2024 (KIVI, ICML) — asymmetric KV quantization
  - Inline: KVLinC 2025 (arXiv:2510.05373), D2Quant 2026 (arXiv:2602.02546), EfficientQAT 2025 (arXiv:2407.11062, ACL), PTQTP 2025 (arXiv:2509.16989)
- **j8web commits:** `11c6da6` (initial), `630d273` (corrections), `beb9917` (revised version) — all pushed to GitHub

## System Prompt Ablation Experiment (completed)

Holds the model constant (qwen3-vl:30b AWQ on vLLM/gpu-host-d) and varies the system prompt to measure how prompt engineering affects drift stability. 20 system prompts across 6 categories, 5 seeds each, 30 iterations.

- **Batch API changes:** Added `system_prompt` and `seed` passthrough to `ai_client.py`, `game_logic.py`, `batch_api.py` (backward-compatible defaults: system_prompt="", seed=None)
- **Tests:** 5 new tests in `tests/test_batch_api.py` (system_prompt passthrough, default, seed passthrough, default, invalid seed)
- **Embedding model:** Qwen/Qwen3-Embedding-0.6B on <GPU-HOST-A>:8002

### Pasta prompt experiment (3-step, completed)
- **Script:** `scripts/run_drift_experiment_sysprompt.py`
- **Prompt:** "Step 1: Boil water. Step 2: Add pasta for 8 minutes. Step 3: Drain and serve with sauce."
- **Matrix:** 20 prompts × 2 emphasis (1x, 2x) × 5 seeds = 200 runs
- **Results:** `results/drift_experiment_sysprompt/drift_data.json`, `drift_plot_1x.png`, `drift_plot_2x.png`, `drift_bar_comparison.png`
- **Finding:** System prompt made almost no difference — all 20 prompts landed in 0.82–0.86 range. The short prompt gave the model too little to differentiate on.

### Lasagna prompt experiment (21-step, completed)
- **Script:** `scripts/run_drift_experiment_sysprompt_lasagna.py`
- **Prompt:** 21-step homemade beef lasagna recipe (detailed ingredients, measurements, layering instructions)
- **Matrix:** 20 prompts × 5 seeds = 100 runs (no 2x emphasis)
- **Results:** `results/drift_experiment_sysprompt_lasagna/drift_data.json`, `drift_plot.png`, `drift_bar.png`

### 20 system prompts tested
| Category | Prompts |
|---|---|
| Baseline (2) | None (empty), Minimal Helpful |
| Anti-Drift (6) | Strong Anti-Meta, Ultra-Constrained Fidelity, 100% Fidelity, Precise Transmitter, Multi-Agent Relay, No Meta Ever |
| Constrained (4) | Deterministic Relay, Numbered-Only Strict, Zero Fluff, Locked Format |
| Creative (4) | Natural & Engaging, Progressive Refinement, Friendly Conversational, Balanced Clarity |
| VL-Leveraged (3) | Expert VL Cooking Assistant, Visual Tutorial Style, Multimodal Visual Focus |
| Thinking (1) | Think-Step-by-Step |

### Key findings (lasagna experiment)
- **Best:** Multi-Agent Relay (0.94), No Meta Ever (0.94), Expert VL Cooking Assistant (0.93)
- **Stable tier (≥0.92):** Think-Step-by-Step (0.93), Numbered-Only Strict (0.93), 100% Fidelity (0.93), Zero Fluff (0.93), Progressive Refinement (0.92), Multimodal Visual Focus (0.92), Locked Format (0.92), Deterministic Relay (0.92)
- **Worst:** Ultra-Constrained Fidelity (0.89), Baseline - None (0.89), Natural & Engaging (0.89)
- **System prompts matter — but only with complex prompts** — lasagna showed 0.054 spread (0.885–0.940) vs nearly flat with 3-step pasta
- **Anti-drift prompts work** — top 2 are both Anti-Drift category; "never reference the chain" and "never mention the task" were the most effective strategies
- **Over-constraining hurts** — Ultra-Constrained Fidelity was worst despite aggressive anti-drift language; being too rigid with a complex recipe backfires
- **VL-leveraged prompts perform well** — Expert VL Cooking Assistant was #3; the model's visual training data provides useful cooking domain knowledge
- **Fixed-point attractors are common** — many seeds locked into exact similarity values for 10+ consecutive iterations (e.g., 0.9691, 0.9611, 0.9204)
- **Seed variance is higher with longer prompts** — same prompt+seed combos showed 0.85–0.97 range vs tight 0.82–0.84 with pasta

## System Prompt Ablation on qwen3:8b (in progress)

Repeats the 20-prompt system prompt ablation on qwen3:8b (q4_K_M weights, q8 KV cache) via Ollama on gpu-host-d:11434. Both spaghetti (3-step) and lasagna (21-step) prompts in a single script for direct comparison.

- **Script:** `scripts/run_drift_experiment_sysprompt_qwen3_8b.py`
- **Model:** qwen3:8b-q4_K_M on <GPU-HOST-D>:11434 (Ollama, q8 KV cache default)
- **Matrix:** 2 recipes × 20 prompts × 5 seeds = 200 runs of 30 iterations (6,000 API calls)
- **Condition key format:** `"{recipe}_{id:02d}_{slug}_s{seed}"`
- **Seeds:** [42, 137, 256, 512, 1024]
- **Settings:** prompt_style=paraphrase, temperature=0.7
- **Embedding model:** Qwen/Qwen3-Embedding-0.6B on <GPU-HOST-A>:8002
- **Results:** `results/drift_experiment_sysprompt_qwen3_8b/`
  - `drift_data.json` — all 200 runs
  - `drift_plot_spaghetti.png` — line plot, 20 prompts, spaghetti recipe (mean ± min/max band)
  - `drift_plot_lasagna.png` — line plot, 20 prompts, lasagna recipe
  - `drift_bar_comparison.png` — grouped bar chart: spaghetti (solid) vs lasagna (hatched), sorted by lasagna mean
- **JSON schema:** Uses `"initial_prompts"` dict (spaghetti + lasagna) instead of single `"initial_prompt"`
- **Parallelism:** ThreadPoolExecutor(max_workers=4), resumable (skips completed condition keys)
- **Launched:** `nohup uv run python -u scripts/run_drift_experiment_sysprompt_qwen3_8b.py > results/drift_experiment_sysprompt_qwen3_8b/run.log 2>&1 &`
- **Monitor:** `tail -f results/drift_experiment_sysprompt_qwen3_8b/run.log`
- **Check count:** `python3 -c "import json; d=json.load(open('results/drift_experiment_sysprompt_qwen3_8b/drift_data.json')); print(len(d['models']), 'of 200 done')"`
- **ETA:** ~3.5-4 hours total (~9s/iteration, ~4.5min/chain, 50 batches of 4)

## System Prompt Ablation Analysis (completed)

Cross-experiment analysis of 3 system prompt ablation datasets (500 runs total, 15,000 LLM inferences) with 5 publication-quality plots and a Part 3 blog post.

- **Script:** `scripts/analyze_sysprompt_ablation.py` — loads 3 datasets, generates 5 plots, prints summary, saves JSON
- **Data sources:**
  - `results/drift_experiment_sysprompt/drift_data.json` — 30B spaghetti 1x/2x (200 runs)
  - `results/drift_experiment_sysprompt_lasagna/drift_data.json` — 30B lasagna (100 runs)
  - `results/drift_experiment_sysprompt_qwen3_8b/drift_data.json` — 8B spaghetti+lasagna (200 runs)
- **Results:** `results/sysprompt_ablation_analysis/` — 5 plots + `summary_stats.json`
- **Blog images:** `../j8web/images/llm-telephone-drift-sysprompt/` — blog-sized copies of all 5 plots

### 5 plots generated
| Plot | Filename | Description |
|---|---|---|
| Hero line | `hero_lasagna_30b.png` | 30B lasagna, 20 prompts, mean ± min/max band, category-colored |
| Complexity threshold | `complexity_threshold.png` | 30B spaghetti (solid) vs lasagna (hatched) grouped bars |
| Cross-model slope | `cross_model_ranking.png` | 30B vs 8B lasagna rankings, Spearman rho annotated |
| Seed variance | `seed_variance_8b.png` | 6 most variable prompts on 8B spaghetti, individual seeds |
| Category heatmap | `category_heatmap.png` | 6 categories × 4 conditions |

### Key findings
- **Complexity threshold:** System prompts invisible on spaghetti (0.008 spread on 30B) but meaningful on lasagna (0.054 spread)
- **30B best:** Multi-Agent Relay (0.940), No Meta Ever (0.939) — target meta-commentary failure mode
- **30B worst:** Ultra-Constrained Fidelity (0.885) — conflicting objectives backfire
- **8B best:** Natural & Engaging (0.903), 100% Fidelity (0.897) — lightweight prompts win
- **Rankings uncorrelated:** Spearman rho = 0.15 (p=0.53) between 30B and 8B lasagna
- **8B seed variance extreme:** Locked Format spans 0.41–0.92 across 5 seeds
- **Fixed-point attractors:** 54% of 30B lasagna runs lock into constant similarity
- **VL-Leveraged safest cross-model category**

## Blog Post Part 3 (published)

- **j8web post:** `_posts/2026-02-22-system-prompt-engineering-semantic-drift.md`
- **Images:** `images/llm-telephone-drift-sysprompt/` (5 plots)
- **URL:** https://joshua8.ai/system-prompt-engineering-semantic-drift/
- **Part 1 link:** https://joshua8.ai/llm-telephone-game-semantic-drift/
- **Part 2 link:** https://joshua8.ai/kv-cache-precision-semantic-drift-qwen3/
- **References (all verified on arXiv):**
  - Perez et al. 2025 (arXiv:2407.04503, ICLR) — cultural attractors
  - Mohamed et al. 2025 (arXiv:2502.20258, ACL) — broken telephone distortion
  - Rath 2026 (arXiv:2601.04170) — agent drift / ASI
  - Han et al. 2026 (arXiv:2602.13595) — quantization trap / sequential amortization failure
  - Jaroslawicz et al. 2025 (arXiv:2507.11538) — IFScale, instruction-following capacity limits
  - Wang et al. 2025 (arXiv:2502.15208) — attractor cycles in iterated paraphrasing
  - Chytas & Singh 2025 (arXiv:2601.11575) — concept attractors as iterated function systems
  - Brinkmann et al. 2023 (arXiv:2311.11388) — machine culture (Nature Human Behaviour)
  - Part 1 + Part 2 links
- All 8 references verified on arXiv (2025-02-22)
- Revised with expanded related work (dynamical-systems grounding, instruction-following capacity)
- Callbacks to Part 2's Speculation section — grades 3 predictions
- Three-part series arc: model choice → infrastructure → prompt engineering
- Includes appendix with full lasagna recipe text
- **j8web commit:** `d3abaf5` — pushed to GitHub

## arXiv Paper — Seeded Replication Experiments (in progress)

Converting the 3-part blog series into an arXiv paper. Parts 1-2 lacked seeds/replication (single run per condition), so we re-run them with 5 seeds each, plus add a temperature ablation and embedding model validation.

### Paper structure
- **File:** `paper/main.tex` + `paper/references.bib`
- **Title:** "Semantic Drift in Iterated LLM Paraphrase Chains: Effects of Model Architecture, Serving Infrastructure, and Prompt Engineering"
- **Template:** article class, natbib (author-year), 14 citations in references.bib
- **Sections:** Abstract, Introduction, Related Work, Methodology, Results (RQ1-RQ3 + temperature + embedding), Discussion, Conclusion, Appendices (A-E)
- **Tables:** `paper/tables/` — auto-generated by `scripts/analyze_paper.py`

### Experiment A: RQ1 replication with seeds (running)
- **Script:** `scripts/run_drift_experiment_rq1_seeds.py`
- **Models:** 17 from original Part 1 (same as drift_data_combined.json, excluding glm-ocr, bible-expert-12b, qwen3-vl:30b ollama-alt)
- **Seeds:** [42, 137, 256, 512, 1024]
- **Prompt:** spaghetti, T=0.7, 30 iterations
- **API calls:** 17 × 5 × 30 = 2,550
- **Output:** `results/drift_experiment_rq1_seeds/drift_data.json`
- **Parallelism:** ThreadPoolExecutor(max_workers=4), resumable

### Experiment B: RQ2 KV cache replication with seeds (q8 queued, fp16 needs manual switch)
- **Script:** `scripts/run_drift_experiment_rq2_seeds.py --kv-mode q8|fp16`
- **Models:** 10 key Qwen3 (0.6B-fp16, 0.6B-q4_K_M, 1.7B-q8_0, 4B-inst-q4_K_M, 4B-inst-fp16, 8B-q4_K_M, 30B-inst-q8_0, 30B-inst-q4_K_M, 30B-think-q4_K_M, 80B-next-q4_K_M)
- **Seeds:** [42, 137, 256, 512, 1024]
- **Prompt:** spaghetti, T=0.7, 30 iterations
- **API calls:** 10 × 2 × 5 × 30 = 3,000
- **Output:** `results/drift_experiment_rq2_seeds/drift_data_q8.json`, `drift_data_fp16.json`
- **Design:** Run once with `--kv-mode q8`, user switches Ollama KV cache config, run again with `--kv-mode fp16`

### Experiment C: Temperature ablation (queued after A)
- **Script:** `scripts/run_drift_experiment_temperature.py`
- **Models:** qwen3:30b-instruct, gemma3:27b, qwen3:8b, llama3.1:8b
- **Temperatures:** [0.0, 0.3, 0.5, 0.7, 1.0]
- **Seeds:** 5 per temp (T=0.0 → 1 seed, deterministic)
- **Prompt:** spaghetti, 30 iterations
- **API calls:** 4 × (4×5 + 1×1) × 30 = 2,640
- **Output:** `results/drift_experiment_temperature/drift_data.json`

### Experiment D: Embedding model validation (queued after E)
- **Script:** `scripts/validate_embeddings.py`
- **No LLM re-runs.** Loads saved output_message texts from all result JSONs
- **Models:** Qwen3-Embedding-0.6B (original, vLLM <GPU-HOST-A>:8002) + bge-m3:latest (Ollama <GPU-HOST-D>:11434)
- **Method:** Re-embed all text pairs, compute Pearson/Spearman correlation between two embedding models
- **Data sources:** All available result JSONs (skips missing files gracefully)
- **Output:** `results/embedding_validation/correlation_stats.json`, `raw_similarities.json`, `embedding_scatter.png`
- **Processing:** ~15,000 pairs from Part 3 data (new experiment data added when available)

### Experiment E: Lasagna on diverse models (queued after B q8, nice-to-have)
- **Script:** `scripts/run_drift_experiment_rq1_lasagna.py`
- **Models:** 6 representative (qwen3:30b-instruct, gemma3:27b, devstral-small-2:24b, qwen3:8b, llama3.1:8b, glm-4.7-flash)
- **Seeds:** [42, 137, 256, 512, 1024]
- **Prompt:** lasagna (21 steps), T=0.7, 30 iterations
- **API calls:** 6 × 5 × 30 = 900
- **Output:** `results/drift_experiment_rq1_lasagna/drift_data.json`

### Unified analysis script
- **Script:** `scripts/analyze_paper.py`
- **Loads:** All datasets (new seeded + existing Part 3)
- **Generates:** 10 main figures + 3 appendix figures at 300 DPI in `results/paper_figures/`
- **Statistics:** Bootstrap 95% CIs (B=10,000), Kruskal-Wallis + Dunn's post-hoc, Wilcoxon signed-rank, Spearman with bootstrap CIs, Cohen's d, fixed-point attractor detection
- **Tables:** LaTeX export to `paper/tables/table1_rq1.tex`, `table2_rq2.tex`
- **Summary:** `results/paper_figures/all_stats.json`

### 13 figures planned
| # | Figure | Source |
|---|---|---|
| 1 | 17-model line plot with 95% CI bands | Exp A |
| 2 | Box plot of final similarities by model | Exp A |
| 3 | KV cache paired delta bar chart with CIs | Exp B |
| 4 | KV cache heatmap (size × quant) | Exp B |
| 5 | Complexity threshold bars (spag vs lasagna) | Part 3 |
| 6 | 30B lasagna hero line plot | Part 3 |
| 7 | Cross-model slope chart (30B vs 8B) | Part 3 |
| 8 | Category heatmap (6 cat × 4 cond) | Part 3 |
| 9 | Temperature × model interaction | Exp C |
| 10 | Embedding validation scatter | Exp D |
| A1 | KV cache hero overlay with CI bands | Exp B |
| A2 | 8B seed variance subplots | Part 3 |
| A3 | Per-seed small multiples (select models) | Exp A |

### Run orchestration
- **Wrapper:** `scripts/run_all_gpu_experiments.sh` — chained A → C → B(q8) → E sequentially
- **Chain completed:** 2026-02-24 07:21 UTC — all 4 experiments finished successfully

### Experiment status (as of 2026-02-24)
| Experiment | Status | Count |
|---|---|---|
| A: RQ1 seeds | **Complete** | 85/85 |
| C: Temperature | **Complete** | 84/84 |
| B q8: RQ2 seeds | **Complete** | 50/50 |
| E: Lasagna diverse | **Complete** | 30/30 |
| D: Embedding validation | **Complete** | 21,636 pairs, Pearson r=0.960, Spearman rho=0.857 |
| B fp16: RQ2 seeds | **Running** (4/50 as of 13:05 UTC) | 50 conditions, ~5-6h |

- **Exp D results:** Pearson r=0.960, Spearman rho=0.857, 21,636 pairs, 0 errors. Qwen mean=0.823, bge-m3 mean=0.881. Strong agreement validates Qwen3-Embedding-0.6B as primary metric.
- **Exp D output:** `results/embedding_validation/correlation_stats.json`, `raw_similarities.json`, `embedding_scatter.png`
- **Exp B fp16 launched:** `nohup uv run python -u scripts/run_drift_experiment_rq2_seeds.py --kv-mode fp16 > results/drift_experiment_rq2_seeds/run_fp16.log 2>&1 &`
- **Monitor B fp16:** `tail -f results/drift_experiment_rq2_seeds/run_fp16.log`
- **Check B fp16 count:** `python3 -c "import json; d=json.load(open('results/drift_experiment_rq2_seeds/drift_data_fp16.json')); print(len(d['models']), 'of 50 done')"`
- **After B fp16 completes:** All experiments done. Run `uv run python scripts/analyze_paper.py`

### Data reuse
| Dataset | Status | Paper Use |
|---|---|---|
| Part 3 sysprompt (3 files, 500 runs, 5 seeds) | **Use as-is** | RQ3 results |
| Part 1 original (17 models, no seeds) | Pilot only | Motivation; stats from Exp A |
| Part 2 original (22 models × 2 KV, no seeds) | Pilot only | Motivation; stats from Exp B |

## Exported Files on /mnt/ssd/
- `drift_plot_final.png` — 17-model drift chart
- `drift_results.md` — full cosine similarity table (17 models × 30 iterations)
- `drift_blog_post_v2.md` — standalone blog post markdown
- `2026-02-21-kv-cache-precision-semantic-drift-qwen3.md` — Part 2 blog post (revised)
- `drift_overlay_selected.png` — hero overlay plot (KV cache comparison)
- `delta_bar_chart.png` — sorted delta bar chart
- `delta_heatmap.png` — size × weight quant heatmap
- `all_models_comparison.png` — 22-model small multiples
- `2026-02-22-system-prompt-engineering-semantic-drift.md` — Part 3 blog post
- `hero_lasagna_30b.png` — 30B lasagna hero line plot
- `complexity_threshold.png` — spaghetti vs lasagna grouped bars
- `cross_model_ranking.png` — 30B vs 8B slope chart
- `seed_variance_8b.png` — 8B seed variance subplots
- `category_heatmap.png` — 6 categories × 4 conditions
