#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Experiment D: Embedding model validation — compare Qwen3-Embedding-0.6B vs bge-m3.

No LLM re-runs. Loads saved output_message texts from all seeded result JSONs,
re-embeds with bge-m3:latest via Ollama, and computes Pearson/Spearman correlation
between the two embedding models' similarity scores.

Usage:
    nohup uv run python -u scripts/validate_embeddings.py \
        > results/embedding_validation/run.log 2>&1 &
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
# Primary: Qwen3-Embedding-0.6B on vLLM
QWEN_EMBED_HOST = os.environ.get("EMBEDDING_HOST", "<GPU-HOST-A>")
QWEN_EMBED_PORT = os.environ.get("EMBEDDING_PORT", "8002")
QWEN_EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Validation: bge-m3 on Ollama
BGE_HOST = "<GPU-HOST-D>"
BGE_PORT = 11434
BGE_MODEL = "bge-m3:latest"

# --- Data sources ---
DATA_SOURCES = [
    {
        "name": "RQ1 seeds (17 models)",
        "path": Path("results/drift_experiment_rq1_seeds/drift_data.json"),
        "initial_prompt_key": "initial_prompt",
    },
    {
        "name": "RQ2 seeds q8",
        "path": Path("results/drift_experiment_rq2_seeds/drift_data_q8.json"),
        "initial_prompt_key": "initial_prompt",
    },
    {
        "name": "RQ2 seeds fp16",
        "path": Path("results/drift_experiment_rq2_seeds/drift_data_fp16.json"),
        "initial_prompt_key": "initial_prompt",
    },
    {
        "name": "Temperature ablation",
        "path": Path("results/drift_experiment_temperature/drift_data.json"),
        "initial_prompt_key": "initial_prompt",
    },
    {
        "name": "Sysprompt 30B spaghetti",
        "path": Path("results/drift_experiment_sysprompt/drift_data.json"),
        "initial_prompt_key": "initial_prompt",
    },
    {
        "name": "Sysprompt 30B lasagna",
        "path": Path("results/drift_experiment_sysprompt_lasagna/drift_data.json"),
        "initial_prompt_key": "initial_prompt",
    },
    {
        "name": "Sysprompt 8B",
        "path": Path("results/drift_experiment_sysprompt_qwen3_8b/drift_data.json"),
        "initial_prompt_key": "initial_prompts",  # dict with spaghetti + lasagna
    },
    {
        "name": "RQ1 lasagna (6 models)",
        "path": Path("results/drift_experiment_rq1_lasagna/drift_data.json"),
        "initial_prompt_key": "initial_prompt",
    },
]

OUT_DIR = Path("results/embedding_validation")


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
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
    """Compute cosine similarity between two texts using a given embedding endpoint."""
    try:
        # Use Ollama embedding API for bge-m3
        if "ollama" in str(port) or int(port) == 11434:
            resp = client.post(
                f"http://{host}:{port}/api/embed",
                json={"model": model, "input": [text_a, text_b]},
                timeout=60.0,
            )
            data = resp.json()
            vec_a = data["embeddings"][0]
            vec_b = data["embeddings"][1]
        else:
            # OpenAI-compatible API
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
    """Extract (original_text, output_text, qwen_sim) pairs from a dataset.

    Returns list of (original, output, existing_qwen_similarity).
    """
    pairs = []
    models_data = raw.get("models", {})

    # Determine the initial prompt(s)
    if source["initial_prompt_key"] == "initial_prompts":
        initial_prompts = raw.get("initial_prompts", {})
    else:
        initial_prompts = {"default": raw.get("initial_prompt", "")}

    for cond_key, iterations in models_data.items():
        # Figure out which initial prompt was used
        if source["initial_prompt_key"] == "initial_prompts":
            # Key format: "{recipe}_{id}_{slug}_s{seed}"
            if cond_key.startswith("spaghetti_"):
                original = initial_prompts.get("spaghetti", "")
            elif cond_key.startswith("lasagna_"):
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

    print("Embedding Model Validation")
    print(f"  Primary:    {QWEN_EMBED_MODEL} on {QWEN_EMBED_HOST}:{QWEN_EMBED_PORT}")
    print(f"  Validation: {BGE_MODEL} on {BGE_HOST}:{BGE_PORT}")
    print()

    # Collect all text pairs from available datasets
    all_pairs: list[tuple[str, str, float | None]] = []

    for source in DATA_SOURCES:
        if not source["path"].exists():
            print(f"  SKIP (not found): {source['name']} — {source['path']}")
            continue

        raw = json.loads(source["path"].read_text())
        pairs = extract_pairs(source, raw)
        print(f"  Loaded {len(pairs)} pairs from: {source['name']}")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("\nERROR: No data found. Run experiments first.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal pairs to re-embed: {len(all_pairs)}")

    # Compute bge-m3 similarities and collect qwen similarities
    qwen_sims: list[float] = []
    bge_sims: list[float] = []
    errors = 0

    with httpx.Client() as client:
        # Test bge-m3 connectivity
        print(f"\nTesting bge-m3 connectivity...")
        test_sim = embed_pair(client, "hello", "world", BGE_HOST, BGE_PORT, BGE_MODEL)
        if test_sim is None:
            print("ERROR: Cannot reach bge-m3 embedding endpoint", file=sys.stderr)
            sys.exit(1)
        print(f"  bge-m3 test similarity: {test_sim:.4f}")

        # Also re-compute Qwen embeddings for pairs where original sim was None
        print(f"\nProcessing {len(all_pairs)} pairs...")
        t0 = time.time()

        for idx, (original, output, qwen_sim) in enumerate(all_pairs):
            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(all_pairs) - idx - 1) / rate / 60
                print(f"  [{idx + 1}/{len(all_pairs)}] {rate:.1f} pairs/s, ETA {eta:.1f}min")

            # Get bge-m3 similarity
            bge_sim = embed_pair(client, original, output, BGE_HOST, BGE_PORT, BGE_MODEL)

            # If qwen sim was None, recompute it
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

    # Compute correlations
    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(qwen_sims, bge_sims)
    spearman_rho, spearman_p = spearmanr(qwen_sims, bge_sims)

    print(f"\n{'='*60}")
    print("EMBEDDING MODEL CORRELATION")
    print(f"{'='*60}")
    print(f"  Pairs:       {len(qwen_sims)}")
    print(f"  Pearson r:   {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman rho: {spearman_rho:.4f} (p={spearman_p:.2e})")
    print(f"  Qwen mean:   {sum(qwen_sims)/len(qwen_sims):.4f}")
    print(f"  bge-m3 mean: {sum(bge_sims)/len(bge_sims):.4f}")

    # Save results
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

    # Save raw data for scatter plot
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

    # Subsample if too many points for readability
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

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1, alpha=0.5, label="y=x")

    # Linear fit
    qw = np.array(plot_qwen)
    bg = np.array(plot_bge)
    m, b = np.polyfit(qw, bg, 1)
    fit_x = np.linspace(qw.min(), qw.max(), 100)
    ax.plot(fit_x, m * fit_x + b, "-", color="#ff4444", linewidth=1.5,
            label=f"Linear fit (r={pearson_r:.3f})")

    ax.set_xlabel(f"Qwen3-Embedding-0.6B Cosine Similarity", fontsize=11)
    ax.set_ylabel(f"bge-m3 Cosine Similarity", fontsize=11)
    ax.set_title(
        f"Embedding Model Validation: Qwen3-Embedding vs bge-m3\n"
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
