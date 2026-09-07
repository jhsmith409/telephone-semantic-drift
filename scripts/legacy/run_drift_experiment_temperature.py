#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Experiment C: Temperature ablation — 4 models × 5 temperatures × 5 seeds.

Tests how sampling temperature affects drift stability across 4 representative models.
T=0.0 gets only 1 seed (deterministic), T=0.3/0.5/0.7/1.0 get 5 seeds each.

Uses ThreadPoolExecutor(max_workers=4) for parallelism.
Resumable: skips condition keys already present in the data file.

Usage:
    nohup uv run python -u scripts/run_drift_experiment_temperature.py \
        > results/drift_experiment_temperature/run.log 2>&1 &
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

BATCH_API_URL = "http://localhost:5001"
ITERATIONS = 30
PROMPT_STYLE = "paraphrase"
SEEDS = [42, 137, 256, 512, 1024]
TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]

INITIAL_PROMPT = (
    "Step 1: Boil water. Step 2: Add pasta for 8 minutes. "
    "Step 3: Drain and serve with sauce."
)

# 4 representative models spanning stable → catastrophic
MODELS = {
    "qwen3:30b-instruct": {
        "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "gemma3:27b": {
        "model": "gemma3:27b-it-qat",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
    "qwen3:8b": {
        "model": "qwen3:8b-q4_K_M",
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
}

OUT_DIR = Path("results/drift_experiment_temperature")


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


def run_single_chain(
    condition_key: str,
    model_info: dict,
    seed: int,
    temperature: float,
) -> tuple[str, list[dict]]:
    """Run 30 iterations for one condition, returning (key, results)."""
    print(f"\n{'='*60}")
    print(f"Condition: {condition_key}")
    print(f"{'='*60}")

    results = []
    current_message = INITIAL_PROMPT

    with httpx.Client() as client:
        for i in range(1, ITERATIONS + 1):
            t0 = time.time()
            try:
                resp = client.post(
                    f"{BATCH_API_URL}/api/run",
                    json={
                        "message": current_message,
                        "prompt_style": PROMPT_STYLE,
                        "temperature": temperature,
                        "seed": seed,
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
                print(f"  [{condition_key}] iter {i:2d}/{ITERATIONS}: FAIL ({error})")
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
            print(f"  [{condition_key}] iter {i:2d}/{ITERATIONS}: sim={sim_str}  ({elapsed:.1f}s)")

            results.append({
                "iteration": i,
                "status": "success",
                "input_message": current_message,
                "output_message": output,
                "cosine_similarity": sim,
                "elapsed_seconds": round(elapsed, 2),
            })

            current_message = output

    return condition_key, results


def build_condition_matrix() -> list[tuple[str, dict, int, float]]:
    """Build the condition matrix: 4 models × 5 temps × seeds.

    T=0.0 gets 1 seed (deterministic), others get 5 seeds.
    Returns list of (condition_key, model_info, seed, temperature).
    """
    conditions = []
    for label, model_info in MODELS.items():
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            for seed in seeds_for_temp:
                key = f"{label}_t{temp:.1f}_s{seed}"
                conditions.append((key, model_info, seed, temp))
    return conditions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / "drift_data.json"

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

    # Load existing data for resumability
    all_data: dict[str, list[dict]] = {}
    if data_path.exists():
        existing = json.loads(data_path.read_text())
        all_data = existing.get("models", {})
        print(f"Loaded {len(all_data)} existing runs from {data_path}")

    # Build condition matrix
    conditions = build_condition_matrix()
    total = len(conditions)

    # Filter out already-completed conditions
    remaining = [(k, mi, s, t) for k, mi, s, t in conditions if k not in all_data]
    print(f"Total conditions: {total}")
    print(f"Already completed: {total - len(remaining)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Seeds: {SEEDS}")

    def _save_json() -> None:
        save_data = {
            "initial_prompt": INITIAL_PROMPT,
            "iterations": ITERATIONS,
            "prompt_style": PROMPT_STYLE,
            "temperatures": TEMPERATURES,
            "seeds": SEEDS,
            "models_config": {label: info for label, info in MODELS.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": all_data,
        }
        data_path.write_text(json.dumps(save_data, indent=2))

    if not remaining:
        print("\nAll conditions already completed.")
    else:
        data_lock = threading.Lock()
        completed = [0]

        def _run_and_save(args: tuple) -> str:
            key, mi, seed, temp = args
            cond_key, results = run_single_chain(key, mi, seed, temp)

            with data_lock:
                all_data[cond_key] = results
                completed[0] += 1
                _save_json()
                print(f"\n  -> Saved {cond_key} ({completed[0]}/{len(remaining)} done)")

            return cond_key

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run_and_save, args): args[0] for args in remaining}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nERROR: {key} failed: {exc}", file=sys.stderr)

    # Final save
    _save_json()
    print(f"\nData saved: {data_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY — Mean Final Similarity by Model × Temperature")
    print(f"{'='*70}")

    header = f"{'Model':<25}"
    for t in TEMPERATURES:
        header += f" {'T='+str(t):>10}"
    print(header)
    print(f"{'-'*25}" + f" {'-'*10}" * len(TEMPERATURES))

    for label in MODELS:
        row = f"  {label:<23}"
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            finals = []
            for seed in seeds_for_temp:
                key = f"{label}_t{temp:.1f}_s{seed}"
                if key not in all_data:
                    continue
                iters = all_data[key]
                sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                if sims:
                    finals.append(sims[-1])
            if finals:
                mean = sum(finals) / len(finals)
                row += f" {mean:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print(f"\nExperiment complete! {len(all_data)} total runs.")


if __name__ == "__main__":
    main()
