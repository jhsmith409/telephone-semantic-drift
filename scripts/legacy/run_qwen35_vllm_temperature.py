#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Qwen3.5 35B AWQ temperature ablation on vLLM.

1 model × 2 recipes × 5 temperatures × 5 seeds (T=0: 1 seed) = 42 conditions.

Usage:
    nohup uv run python -u scripts/run_qwen35_vllm_temperature.py \
        > results/qwen35_vllm_temperature/run.log 2>&1 &
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

PROMPTS = {
    "spaghetti": (
        "Step 1: Boil water. Step 2: Add pasta for 8 minutes. "
        "Step 3: Drain and serve with sauce."
    ),
    "lasagna": (
        "Follow these exact steps to prepare a classic homemade beef lasagna that serves 8-10 people:\n\n"
        "Step 1: Preheat your oven to 375\u00b0F (190\u00b0C) and lightly grease a 9x13 inch baking dish.\n"
        "Step 2: Bring a large pot of lightly salted water to a boil.\n"
        "Step 3: Heat 2 tablespoons of olive oil in a large skillet over medium heat.\n"
        "Step 4: Add 1 large finely diced onion and cook for 5 minutes until translucent.\n"
        "Step 5: Add 5 minced garlic cloves and cook for 1 minute until fragrant.\n"
        "Step 6: Add 1.25 lbs ground beef and 0.75 lb Italian sausage. Cook until browned, breaking up the meat with a spoon.\n"
        "Step 7: Drain excess grease from the skillet.\n"
        "Step 8: Stir in two 28-oz cans of crushed tomatoes, 3 tablespoons tomato paste, 2 teaspoons dried basil, "
        "1.5 teaspoons dried oregano, 0.5 teaspoon fennel seed, 1 teaspoon sugar, 1 teaspoon salt, and 0.5 teaspoon black pepper.\n"
        "Step 9: Bring to a simmer and cook the meat sauce for 30 minutes, stirring occasionally.\n"
        "Step 10: In a large bowl, combine 15 oz ricotta cheese, 1 large egg, 1/2 cup grated Parmesan cheese, "
        "1/4 cup chopped fresh parsley, 1/2 teaspoon salt, and 1/4 teaspoon black pepper. Mix well and set aside.\n"
        "Step 11: Cook 12 lasagna noodles in the boiling water according to package directions until al dente. Drain and rinse with cold water.\n"
        "Step 12: Spread a thin layer of meat sauce on the bottom of the prepared baking dish.\n"
        "Step 13: Arrange 4 lasagna noodles in a single layer over the sauce.\n"
        "Step 14: Spread one-third of the ricotta mixture evenly over the noodles.\n"
        "Step 15: Sprinkle with 1.5 cups shredded mozzarella cheese.\n"
        "Step 16: Repeat the layering process two more times (sauce, noodles, ricotta, mozzarella).\n"
        "Step 17: Top the final layer with the remaining meat sauce and sprinkle generously with 2 cups mozzarella and 1/4 cup Parmesan cheese.\n"
        "Step 18: Cover the dish with aluminum foil and bake for 25 minutes.\n"
        "Step 19: Remove the foil and bake for an additional 25-30 minutes until the cheese is bubbly and golden brown.\n"
        "Step 20: Remove from oven and let the lasagna rest for 15-20 minutes before slicing.\n"
        "Step 21: Garnish with fresh basil if desired and serve hot."
    ),
}

MODEL_INFO = {
    "model": "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit",
    "service_type": "openai",
    "host": "<GPU-HOST-D>",
    "port": 8005,
}
MODEL_LABEL = "qwen3.5:35b-awq"

OUT_DIR = Path("results/qwen35_vllm_temperature")


def get_similarity(client: httpx.Client, text_a: str, text_b: str) -> float | None:
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
    seed: int,
    temperature: float,
    initial_prompt: str,
) -> tuple[str, list[dict]]:
    print(f"\n{'='*60}")
    print(f"Condition: {condition_key}")
    print(f"{'='*60}")

    results = []
    current_message = initial_prompt

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
                        **MODEL_INFO,
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
            sim = get_similarity(client, initial_prompt, output)

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


def build_condition_matrix() -> list[tuple[str, int, float, str]]:
    conditions = []
    for recipe, prompt in PROMPTS.items():
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            for seed in seeds_for_temp:
                prefix = "" if recipe == "spaghetti" else "lasagna_"
                key = f"{prefix}{MODEL_LABEL}_t{temp:.1f}_s{seed}"
                conditions.append((key, seed, temp, prompt))
    return conditions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / "drift_data.json"

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

    all_data: dict[str, list[dict]] = {}
    if data_path.exists():
        existing = json.loads(data_path.read_text())
        all_data = existing.get("models", {})
        print(f"Loaded {len(all_data)} existing runs from {data_path}")

    conditions = build_condition_matrix()
    total = len(conditions)

    remaining = [(k, s, t, p) for k, s, t, p in conditions if k not in all_data]
    print(f"Total conditions: {total}")
    print(f"Already completed: {total - len(remaining)}")
    print(f"Remaining: {len(remaining)}")

    def _save_json() -> None:
        save_data = {
            "initial_prompts": PROMPTS,
            "iterations": ITERATIONS,
            "prompt_style": PROMPT_STYLE,
            "temperatures": TEMPERATURES,
            "seeds": SEEDS,
            "model_info": MODEL_INFO,
            "model_label": MODEL_LABEL,
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
            key, seed, temp, prompt = args
            cond_key, results = run_single_chain(key, seed, temp, prompt)
            with data_lock:
                all_data[cond_key] = results
                completed[0] += 1
                _save_json()
                print(f"\n  -> Saved {cond_key} ({completed[0]}/{len(remaining)} done)")
            return cond_key

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(_run_and_save, args): args[0] for args in remaining}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nERROR: {key} failed: {exc}", file=sys.stderr)

    _save_json()
    print(f"\nData saved: {data_path}")

    # Summary
    for recipe in ["spaghetti", "lasagna"]:
        prefix = "" if recipe == "spaghetti" else "lasagna_"
        print(f"\n{'='*70}")
        print(f"SUMMARY — {MODEL_LABEL} — {recipe.upper()} — by Temperature")
        print(f"{'='*70}")
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            finals = []
            for seed in seeds_for_temp:
                key = f"{prefix}{MODEL_LABEL}_t{temp:.1f}_s{seed}"
                if key not in all_data:
                    continue
                iters = all_data[key]
                sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                if sims:
                    finals.append(sims[-1])
            if finals:
                mean = sum(finals) / len(finals)
                print(f"  T={temp:.1f}  mean={mean:.4f}  n={len(finals)}")

    print(f"\nExperiment complete! {len(all_data)} total runs.")


if __name__ == "__main__":
    main()
