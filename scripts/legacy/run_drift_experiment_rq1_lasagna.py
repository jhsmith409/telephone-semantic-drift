#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Experiment E (nice-to-have): Lasagna prompt on 6 representative models × 5 seeds.

Tests whether the model hierarchy from RQ1 (spaghetti) holds on a more complex prompt.

Uses ThreadPoolExecutor(max_workers=4) for parallelism.
Resumable: skips condition keys already present in the data file.

Usage:
    nohup uv run python -u scripts/run_drift_experiment_rq1_lasagna.py \
        > results/drift_experiment_rq1_lasagna/run.log 2>&1 &
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
TEMPERATURE = 0.7
SEEDS = [42, 137, 256, 512, 1024]

INITIAL_PROMPT = """\
Follow these exact steps to prepare a classic homemade beef lasagna that serves 8-10 people:

Step 1: Preheat your oven to 375°F (190°C) and lightly grease a 9x13 inch baking dish.
Step 2: Bring a large pot of lightly salted water to a boil.
Step 3: Heat 2 tablespoons of olive oil in a large skillet over medium heat.
Step 4: Add 1 large finely diced onion and cook for 5 minutes until translucent.
Step 5: Add 5 minced garlic cloves and cook for 1 minute until fragrant.
Step 6: Add 1.25 lbs ground beef and 0.75 lb Italian sausage. Cook until browned, breaking up the meat with a spoon.
Step 7: Drain excess grease from the skillet.
Step 8: Stir in two 28-oz cans of crushed tomatoes, 3 tablespoons tomato paste, 2 teaspoons dried basil, 1.5 teaspoons dried oregano, 0.5 teaspoon fennel seed, 1 teaspoon sugar, 1 teaspoon salt, and 0.5 teaspoon black pepper.
Step 9: Bring to a simmer and cook the meat sauce for 30 minutes, stirring occasionally.
Step 10: In a large bowl, combine 15 oz ricotta cheese, 1 large egg, 1/2 cup grated Parmesan cheese, 1/4 cup chopped fresh parsley, 1/2 teaspoon salt, and 1/4 teaspoon black pepper. Mix well and set aside.
Step 11: Cook 12 lasagna noodles in the boiling water according to package directions until al dente. Drain and rinse with cold water.
Step 12: Spread a thin layer of meat sauce on the bottom of the prepared baking dish.
Step 13: Arrange 4 lasagna noodles in a single layer over the sauce.
Step 14: Spread one-third of the ricotta mixture evenly over the noodles.
Step 15: Sprinkle with 1.5 cups shredded mozzarella cheese.
Step 16: Repeat the layering process two more times (sauce, noodles, ricotta, mozzarella).
Step 17: Top the final layer with the remaining meat sauce and sprinkle generously with 2 cups mozzarella and 1/4 cup Parmesan cheese.
Step 18: Cover the dish with aluminum foil and bake for 25 minutes.
Step 19: Remove the foil and bake for an additional 25-30 minutes until the cheese is bubbly and golden brown.
Step 20: Remove from oven and let the lasagna rest for 15-20 minutes before slicing.
Step 21: Garnish with fresh basil if desired and serve hot."""

# 6 representative models spanning the stability spectrum
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
    "devstral-small-2:24b": {
        "model": "devstral-small-2:24b",
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
    "glm-4.7-flash": {
        "model": "glm-4.7-flash:Q4_K_M",
        "service_type": "ollama",
        "host": "<GPU-HOST-D>",
        "port": 11434,
    },
}

OUT_DIR = Path("results/drift_experiment_rq1_lasagna")


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
) -> tuple[str, list[dict]]:
    """Run 30 iterations for one model+seed, returning (key, results)."""
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
                        "temperature": TEMPERATURE,
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


def build_condition_matrix() -> list[tuple[str, dict, int]]:
    """Build the condition matrix: 6 models × 5 seeds = 30 runs."""
    conditions = []
    for label, model_info in MODELS.items():
        for seed in SEEDS:
            key = f"{label}_s{seed}"
            conditions.append((key, model_info, seed))
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
    remaining = [(k, mi, s) for k, mi, s in conditions if k not in all_data]
    print(f"Total conditions: {total}")
    print(f"Already completed: {total - len(remaining)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Models: {len(MODELS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Prompt: lasagna (21 steps)")

    def _save_json() -> None:
        save_data = {
            "initial_prompt": INITIAL_PROMPT,
            "iterations": ITERATIONS,
            "prompt_style": PROMPT_STYLE,
            "temperature": TEMPERATURE,
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
            key, mi, seed = args
            cond_key, results = run_single_chain(key, mi, seed)

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
    print(f"SUMMARY — Mean Final Similarity by Model (lasagna, 5 seeds)")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    rows = []
    for label in MODELS:
        finals = []
        for seed in SEEDS:
            key = f"{label}_s{seed}"
            if key not in all_data:
                continue
            iters = all_data[key]
            sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
            if sims:
                finals.append(sims[-1])

        if finals:
            mean = sum(finals) / len(finals)
            std = (sum((x - mean) ** 2 for x in finals) / len(finals)) ** 0.5
            rows.append((label, mean, std, min(finals), max(finals)))

    rows.sort(key=lambda x: x[1], reverse=True)
    for label, mean, std, mn, mx in rows:
        print(f"  {label:<28} {mean:>8.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f}")

    print(f"\nExperiment complete! {len(all_data)} total runs.")


if __name__ == "__main__":
    main()
