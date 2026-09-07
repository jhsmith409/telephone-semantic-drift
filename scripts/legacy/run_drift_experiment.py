#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Run a 30-iteration telephone chain for multiple models, plotting cosine similarity drift.

Usage:
    nohup uv run python scripts/run_drift_experiment.py &

Outputs:
    results/drift_experiment/drift_plot.png
    results/drift_experiment/drift_data.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BATCH_API_URL = "http://localhost:5001"
ITERATIONS = 30
PROMPT_STYLE = "paraphrase"
TEMPERATURE = 0.7

INITIAL_PROMPT = (
    "Step 1: Boil water. Step 2: Add pasta for 8 minutes. "
    "Step 3: Drain and serve with sauce."
)

MODELS = {
    "gemma3:27b": {
        "model": "gemma3:27b-it-qat",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3-next": {
        "model": "qwen3-next:80b-a3b-instruct-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "llama3.1:8b": {
        "model": "llama3.1:8b",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "gpt-oss:120b": {
        "model": "gpt-oss:120b",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
}

OUT_DIR = Path("results/drift_experiment")


def run_iteration(client: httpx.Client, message: str, model_info: dict) -> dict:
    """Call the batch API for a single iteration."""
    resp = client.post(
        f"{BATCH_API_URL}/api/run",
        json={
            "message": message,
            "prompt_style": PROMPT_STYLE,
            "temperature": TEMPERATURE,
            **model_info,
        },
        timeout=300.0,
    )
    return resp.json()


def get_similarity(client: httpx.Client, text_a: str, text_b: str) -> float | None:
    """Compute cosine similarity between two texts via the embeddings module.

    Uses a direct call to the embedding endpoint rather than going through
    the batch API, so we can compare against the *original* prompt.
    """
    import os
    host = os.environ.get("EMBEDDING_HOST", "<GPU-HOST-A>")
    port = os.environ.get("EMBEDDING_PORT", "8002")
    model = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

    try:
        resp = client.post(
            f"http://{host}:{port}/v1/embeddings",
            json={"input": [text_a, text_b], "model": model},
            timeout=30.0,
        )
        data = resp.json()
        vec_a = data["data"][0]["embedding"]
        vec_b = data["data"][1]["embedding"]

        import math
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(x * x for x in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception as e:
        print(f"  [embedding error: {e}]")
        return None


def run_model_chain(client: httpx.Client, label: str, model_info: dict) -> list[dict]:
    """Run 30 iterations for one model, returning per-iteration data."""
    print(f"\n{'='*60}")
    print(f"Model: {label} ({model_info['model']})")
    print(f"{'='*60}")

    results = []
    current_message = INITIAL_PROMPT

    for i in range(1, ITERATIONS + 1):
        t0 = time.time()
        resp = run_iteration(client, current_message, model_info)
        elapsed = time.time() - t0

        if resp.get("status") != "success":
            error = resp.get("error", "unknown")
            print(f"  iter {i:2d}/{ITERATIONS}: FAIL ({error})")
            results.append({
                "iteration": i,
                "status": "error",
                "error": error,
                "cosine_similarity": None,
            })
            break

        output = resp["output_message"]

        # Compute similarity vs ORIGINAL prompt (not vs previous iteration)
        sim = get_similarity(client, INITIAL_PROMPT, output)

        sim_str = f"{sim:.4f}" if sim is not None else "N/A"
        print(f"  iter {i:2d}/{ITERATIONS}: sim={sim_str}  ({elapsed:.1f}s)  [{output[:80]}...]")

        results.append({
            "iteration": i,
            "status": "success",
            "input_message": current_message,
            "output_message": output,
            "cosine_similarity": sim,
            "elapsed_seconds": round(elapsed, 2),
        })

        current_message = output

    return results


def plot_results(all_data: dict) -> None:
    """Generate the drift plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, iterations in all_data.items():
        xs = [r["iteration"] for r in iterations if r["cosine_similarity"] is not None]
        ys = [r["cosine_similarity"] for r in iterations if r["cosine_similarity"] is not None]
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, label=label)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax.set_title("Semantic Drift: Cosine Similarity vs Iteration (paraphrase, t=0.7)", fontsize=13)
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    plot_path = OUT_DIR / "drift_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wait for batch API to be ready
    print(f"Waiting for batch API at {BATCH_API_URL}...")
    with httpx.Client() as client:
        for attempt in range(30):
            try:
                r = client.get(f"{BATCH_API_URL}/api/services", timeout=5.0)
                if r.status_code == 200:
                    print("Batch API is ready.\n")
                    break
            except httpx.ConnectError:
                pass
            time.sleep(1)
        else:
            print("ERROR: Batch API not reachable after 30s", file=sys.stderr)
            sys.exit(1)

    print(f"Initial prompt: {INITIAL_PROMPT}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Prompt style: {PROMPT_STYLE}, Temperature: {TEMPERATURE}")
    print(f"Models: {', '.join(MODELS.keys())}")

    all_data = {}
    with httpx.Client() as client:
        for label, model_info in MODELS.items():
            all_data[label] = run_model_chain(client, label, model_info)

    # Save raw data
    data_path = OUT_DIR / "drift_data.json"
    data_path.write_text(json.dumps({
        "initial_prompt": INITIAL_PROMPT,
        "iterations": ITERATIONS,
        "prompt_style": PROMPT_STYLE,
        "temperature": TEMPERATURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": all_data,
    }, indent=2))
    print(f"\nData saved: {data_path}")

    # Plot
    plot_results(all_data)
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
