#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Embedding model validation for Qwen3.5 experiment data.

Compares Qwen3-Embedding-0.6B (primary) vs bge-m3 (validation) by re-embedding
all text pairs from Qwen3.5 experiment results.

No LLM re-runs. Loads saved output_message texts from all Qwen3.5 result JSONs,
re-embeds with bge-m3 via Ollama on .211:8003, and computes Pearson/Spearman
correlation between the two embedding models' similarity scores.

Usage:
    nohup uv run python -u scripts/validate_embeddings_qwen35.py \
        > results/embedding_validation_qwen35/run.log 2>&1 &
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import httpx

# --- Embedding endpoints ---
QWEN_EMBED_HOST = os.environ.get("EMBEDDING_HOST", "<GPU-HOST-A>")
QWEN_EMBED_PORT = os.environ.get("EMBEDDING_PORT", "8002")
QWEN_EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

BGE_HOST = "<GPU-HOST-A>"
BGE_PORT = 8003
BGE_MODEL = "BAAI/bge-m3"

# --- Data sources ---
DATA_SOURCES = [
    {
        "name": "Exp 1 size_quant",
        "path": Path("results/qwen35_size_quant/drift_data.json"),
        "prompt_key": "initial_prompts",
    },
    {
        "name": "Exp 2 temperature",
        "path": Path("results/qwen35_temperature/drift_data.json"),
        "prompt_key": "initial_prompts",
    },
    {
        "name": "Exp 3a vLLM drift",
        "path": Path("results/qwen35_vllm/drift_data.json"),
        "prompt_key": "initial_prompts",
    },
    {
        "name": "Exp 3b vLLM temperature",
        "path": Path("results/qwen35_vllm_temperature/drift_data.json"),
        "prompt_key": "initial_prompts",
    },
    {
        "name": "Exp 3c vLLM sysprompt",
        "path": Path("results/qwen35_vllm_sysprompt/drift_data.json"),
        "prompt_key": "initial_prompts",
    },
    {
        "name": "Exp 4 sysprompt",
        "path": Path("results/qwen35_sysprompt/drift_data.json"),
        "prompt_key": "initial_prompts",
    },
]

OUT_DIR = Path("results/embedding_validation_qwen35")


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_pair(
    client: httpx.Client,
    text_a: str,
    text_b: str,
    host: str,
    port: int | str,
    model: str,
) -> float | None:
    try:
        port_int = int(port)
        if port_int == 11434:
            # Ollama embedding API
            resp = client.post(
                f"http://{host}:{port}/api/embed",
                json={"model": model, "input": [text_a, text_b]},
                timeout=60.0,
            )
            data = resp.json()
            vec_a = data["embeddings"][0]
            vec_b = data["embeddings"][1]
        else:
            # OpenAI-compatible API (vLLM or sglang)
            resp = client.post(
                f"http://{host}:{port}/v1/embeddings",
                json={"input": [text_a, text_b], "model": model},
                timeout=60.0,
            )
            data = resp.json()
            vec_a = data["data"][0]["embedding"]
            vec_b = data["data"][1]["embedding"]

        return cosine_sim(vec_a, vec_b)
    except Exception as e:
        print(f"    [embed error ({model}): {e}]")
        return None


def extract_pairs(source: dict, raw: dict) -> list[tuple[str, str, float | None]]:
    """Extract (original_text, output_text, qwen_sim) pairs from a dataset."""
    pairs = []
    models_data = raw.get("models", {})

    # Determine the initial prompt(s)
    prompt_key = source["prompt_key"]
    if prompt_key == "initial_prompts":
        initial_prompts = raw.get("initial_prompts", {})
    else:
        initial_prompts = {"default": raw.get("initial_prompt", "")}

    for cond_key, iterations in models_data.items():
        # Figure out which initial prompt was used
        if prompt_key == "initial_prompts":
            if cond_key.startswith("lasagna_") or "_lasagna_" in cond_key:
                original = initial_prompts.get("lasagna", "")
            else:
                original = initial_prompts.get("spaghetti", "")
        else:
            original = initial_prompts.get("default", "")

        if not original:
            continue

        for step in iterations:
            if step.get("status") != "success":
                continue
            output = step.get("output_message", "")
            if not output:
                continue
            qwen_sim = step.get("cosine_similarity")
            pairs.append((original, output, qwen_sim))

    return pairs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Qwen3.5 Embedding Model Validation")
    print(f"  Primary:    {QWEN_EMBED_MODEL} on {QWEN_EMBED_HOST}:{QWEN_EMBED_PORT}")
    print(f"  Validation: {BGE_MODEL} on {BGE_HOST}:{BGE_PORT}")
    print()

    all_pairs: list[tuple[str, str, float | None]] = []

    for source in DATA_SOURCES:
        if not source["path"].exists():
            print(f"  SKIP (not found): {source['name']} -- {source['path']}")
            continue

        raw = json.loads(source["path"].read_text())
        pairs = extract_pairs(source, raw)
        print(f"  Loaded {len(pairs)} pairs from: {source['name']}")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("\nERROR: No data found. Run experiments first.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal pairs to re-embed: {len(all_pairs)}")

    qwen_sims: list[float] = []
    bge_sims: list[float] = []
    errors = 0

    with httpx.Client() as client:
        # Test bge-m3 connectivity
        print(f"\nTesting bge-m3 connectivity at {BGE_HOST}:{BGE_PORT}...")
        test_sim = embed_pair(client, "hello", "world", BGE_HOST, BGE_PORT, BGE_MODEL)
        if test_sim is None:
            print("ERROR: Cannot reach bge-m3 embedding endpoint", file=sys.stderr)
            sys.exit(1)
        print(f"  bge-m3 test similarity: {test_sim:.4f}")

        print(f"\nProcessing {len(all_pairs)} pairs...")
        t0 = time.time()

        for idx, (original, output, qwen_sim) in enumerate(all_pairs):
            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(all_pairs) - idx - 1) / rate / 60
                print(f"  [{idx + 1}/{len(all_pairs)}] {rate:.1f} pairs/s, ETA {eta:.1f}min")

            bge_sim = embed_pair(client, original, output, BGE_HOST, BGE_PORT, BGE_MODEL)

            if qwen_sim is None:
                qwen_sim = embed_pair(
                    client, original, output,
                    QWEN_EMBED_HOST, QWEN_EMBED_PORT, QWEN_EMBED_MODEL,
                )

            if qwen_sim is not None and bge_sim is not None:
                qwen_sims.append(qwen_sim)
                bge_sims.append(bge_sim)
            else:
                errors += 1

    print(f"\nDone. Valid pairs: {len(qwen_sims)}, errors: {errors}")
    print(f"Time: {time.time() - t0:.1f}s")

    if len(qwen_sims) < 10:
        print("ERROR: Too few valid pairs for correlation", file=sys.stderr)
        sys.exit(1)

    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(qwen_sims, bge_sims)
    spearman_rho, spearman_p = spearmanr(qwen_sims, bge_sims)

    print(f"\n{'='*60}")
    print("EMBEDDING MODEL CORRELATION (Qwen3.5 data)")
    print(f"{'='*60}")
    print(f"  Pairs:       {len(qwen_sims)}")
    print(f"  Pearson r:   {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman rho: {spearman_rho:.4f} (p={spearman_p:.2e})")
    print(f"  Qwen mean:   {sum(qwen_sims)/len(qwen_sims):.4f}")
    print(f"  bge-m3 mean: {sum(bge_sims)/len(bge_sims):.4f}")

    results = {
        "n_pairs": len(qwen_sims),
        "n_errors": errors,
        "pearson_r": round(pearson_r, 6),
        "pearson_p": pearson_p,
        "spearman_rho": round(spearman_rho, 6),
        "spearman_p": spearman_p,
        "qwen_model": QWEN_EMBED_MODEL,
        "bge_model": BGE_MODEL,
        "qwen_mean": round(sum(qwen_sims) / len(qwen_sims), 6),
        "bge_mean": round(sum(bge_sims) / len(bge_sims), 6),
    }
    stats_path = OUT_DIR / "correlation_stats.json"
    stats_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Stats saved: {stats_path}")

    raw_path = OUT_DIR / "raw_similarities.json"
    raw_path.write_text(json.dumps({
        "qwen_sims": [round(s, 6) for s in qwen_sims],
        "bge_sims": [round(s, 6) for s in bge_sims],
    }))
    print(f"  Raw data saved: {raw_path}")

    # Generate scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8))

    n = len(qwen_sims)
    if n > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 5000, replace=False)
        plot_qwen = [qwen_sims[i] for i in idx]
        plot_bge = [bge_sims[i] for i in idx]
        label = f"N={n} (5000 shown)"
    else:
        plot_qwen = qwen_sims
        plot_bge = bge_sims
        label = f"N={n}"

    ax.scatter(plot_qwen, plot_bge, alpha=0.15, s=8, color="#4488ff", edgecolors="none")
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1, alpha=0.5, label="y=x")

    qw = np.array(plot_qwen)
    bg = np.array(plot_bge)
    m, b = np.polyfit(qw, bg, 1)
    fit_x = np.linspace(qw.min(), qw.max(), 100)
    ax.plot(fit_x, m * fit_x + b, "-", color="#ff4444", linewidth=1.5,
            label=f"Linear fit (r={pearson_r:.3f})")

    ax.set_xlabel("Qwen3-Embedding-0.6B Cosine Similarity", fontsize=11)
    ax.set_ylabel("bge-m3 Cosine Similarity", fontsize=11)
    ax.set_title(
        f"Embedding Validation (Qwen3.5 data): Qwen3-Embedding vs bge-m3\n"
        f"Pearson r={pearson_r:.3f}, Spearman rho={spearman_rho:.3f} ({label})",
        fontsize=12,
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")

    plot_path = OUT_DIR / "embedding_scatter.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plot saved: {plot_path}")

    print("\nEmbedding validation complete!")


if __name__ == "__main__":
    main()
