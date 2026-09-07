#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Run 30-iteration drift experiment for Qwen3 family with FP16 KV Cache.

Re-run of the original 22-model Qwen3 quantization experiment after Ollama
was updated from FP8 to FP16 KV Cache. Same prompt, settings, and models.

Usage:
    nohup uv run python -u scripts/run_drift_experiment_qwen3_fp16kv.py &
"""

from __future__ import annotations

import json
import math
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
    # 0.6B
    "qwen3:0.6b-fp16": {
        "model": "qwen3:0.6b-fp16",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:0.6b-q8_0": {
        "model": "qwen3:0.6b-q8_0",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:0.6b-q4_K_M": {
        "model": "qwen3:0.6b-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    # 1.7B
    "qwen3:1.7b-fp16": {
        "model": "qwen3:1.7b-fp16",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:1.7b-q8_0": {
        "model": "qwen3:1.7b-q8_0",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:1.7b-q4_K_M": {
        "model": "qwen3:1.7b-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    # 4B
    "qwen3:4b-instruct-2507-fp16": {
        "model": "qwen3:4b-instruct-2507-fp16",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:4b-instruct-2507-q8_0": {
        "model": "qwen3:4b-instruct-2507-q8_0",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:4b-instruct-2507-q4_K_M": {
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    # 8B
    "qwen3:8b-fp16": {
        "model": "qwen3:8b-fp16",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:8b-q8_0": {
        "model": "qwen3:8b-q8_0",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:8b-q4_K_M": {
        "model": "qwen3:8b-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    # 30B MoE (A3B active) — instruct
    "qwen3:30b-a3b-instruct-2507-fp16": {
        "model": "qwen3:30b-a3b-instruct-2507-fp16",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:30b-a3b-instruct-2507-q8_0": {
        "model": "qwen3:30b-a3b-instruct-2507-q8_0",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:30b-a3b-instruct-2507-q4_K_M": {
        "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    # 30B MoE (A3B active) — thinking
    "qwen3:30b-a3b-thinking-2507-fp16": {
        "model": "qwen3:30b-a3b-thinking-2507-fp16",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:30b-a3b-thinking-2507-q8_0": {
        "model": "qwen3:30b-a3b-thinking-2507-q8_0",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:30b-a3b-thinking-2507-q4_K_M": {
        "model": "qwen3:30b-a3b-thinking-2507-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    # 32B dense
    "qwen3:32b-fp16": {
        "model": "qwen3:32b-fp16",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:32b-q8_0": {
        "model": "qwen3:32b-q8_0",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:32b-q4_K_M": {
        "model": "qwen3:32b-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    # 80B MoE (qwen3-next)
    "qwen3-next:80b-a3b-instruct-q4_K_M": {
        "model": "qwen3-next:80b-a3b-instruct-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
}

OUT_DIR = Path("results/drift_experiment_qwen3_fp16kv")


def get_similarity(client: httpx.Client, text_a: str, text_b: str) -> float | None:
    """Compute cosine similarity between two texts via embedding endpoint."""
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
        try:
            resp = client.post(
                f"{BATCH_API_URL}/api/run",
                json={
                    "message": current_message,
                    "prompt_style": PROMPT_STYLE,
                    "temperature": TEMPERATURE,
                    **model_info,
                },
                timeout=300.0,
            )
            api_result = resp.json()
        except Exception as exc:
            api_result = {"status": "error", "error": str(exc)}

        elapsed = time.time() - t0

        if api_result.get("status") != "success":
            error = api_result.get("error", "unknown")
            print(f"  iter {i:2d}/{ITERATIONS}: FAIL ({error})")
            results.append({
                "iteration": i,
                "status": "error",
                "error": error,
                "cosine_similarity": None,
            })
            break

        output = api_result["output_message"]
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
    """Generate the drift plot for all Qwen3 models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort models by final similarity (descending) for legend order
    def final_sim(label):
        iters = all_data[label]
        sims = [r["cosine_similarity"] for r in iters if r["cosine_similarity"] is not None]
        return sims[-1] if sims else 0

    sorted_labels = sorted(all_data.keys(), key=final_sim, reverse=True)

    model_count = len(sorted_labels)
    fig, ax = plt.subplots(figsize=(16, 9))

    for label in sorted_labels:
        iterations = all_data[label]
        xs = [r["iteration"] for r in iterations if r["cosine_similarity"] is not None]
        ys = [r["cosine_similarity"] for r in iterations if r["cosine_similarity"] is not None]
        if xs:
            final = ys[-1]
            ax.plot(
                xs, ys, marker="o", markersize=3, linewidth=1.3,
                label=f"{label} ({final:.2f})",
            )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax.set_title(
        f"Qwen3 Semantic Drift: Size × Quantization — FP16 KV Cache (paraphrase, t=0.7)\n"
        f"{model_count} Models — {ITERATIONS} Iterations",
        fontsize=13,
    )
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    plot_path = OUT_DIR / "drift_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wait for batch API
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
    print(f"Models to run: {len(MODELS)}")
    print(f"Models: {', '.join(MODELS.keys())}")

    all_data = {}
    with httpx.Client() as client:
        for label, model_info in MODELS.items():
            all_data[label] = run_model_chain(client, label, model_info)

    # Save data
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

    # Generate plot
    plot_results(all_data)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<50} {'Final Sim':>10}")
    print(f"{'-'*50} {'-'*10}")

    def final_sim(label):
        iters = all_data[label]
        sims = [r["cosine_similarity"] for r in iters if r["cosine_similarity"] is not None]
        return sims[-1] if sims else 0

    for label in sorted(all_data.keys(), key=final_sim, reverse=True):
        f = final_sim(label)
        count = len([r for r in all_data[label] if r.get("status") == "success"])
        print(f"  {label:<48} {f:>8.4f}  ({count}/{ITERATIONS} iters)")

    print(f"\nExperiment complete!")


if __name__ == "__main__":
    main()
