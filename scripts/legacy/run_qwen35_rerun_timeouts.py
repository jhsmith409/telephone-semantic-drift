#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Re-run timed-out Exp 1 conditions sequentially (max_workers=1).

Identifies conditions in drift_data.json that have <30 iterations and
failed with a timeout error, removes them from the data, then re-runs
them one at a time to avoid GPU contention.

Skips 0.8B "missing required fields" errors (genuine model failures).

Usage:
    uv run python -u scripts/run_qwen35_rerun_timeouts.py
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
SEEDS = [42, 137, 256, 512, 1024]

PROMPTS = {
    "spaghetti": (
        "Step 1: Boil water. Step 2: Add pasta for 8 minutes. "
        "Step 3: Drain and serve with sauce."
    ),
    "lasagna": (
        "Follow these exact steps to prepare a classic homemade beef lasagna that serves 8-10 people: "
        "Step 1: Preheat your oven to 375\u00b0F (190\u00b0C) and lightly grease a 9\u00d713 inch baking dish. "
        "Step 2: Bring a large pot of lightly salted water to a boil. "
        "Step 3: Heat 2 tablespoons of olive oil in a large skillet over medium heat. "
        "Step 4: Add 1 large finely diced onion and cook for 5 minutes until translucent. "
        "Step 5: Add 5 minced garlic cloves and cook for 1 minute until fragrant. "
        "Step 6: Add 1.25 lbs ground beef and 0.75 lb Italian sausage. Cook until browned. "
        "Step 7: Drain excess grease from the skillet. "
        "Step 8: Stir in two 28-oz cans of crushed tomatoes, 3 tbsp tomato paste, "
        "2 tsp dried basil, 1.5 tsp dried oregano, 0.5 tsp fennel seed, 1 tsp sugar, "
        "1 tsp salt, and 0.5 tsp black pepper. "
        "Step 9: Bring to a simmer and cook the meat sauce for 30 minutes. "
        "Step 10: Combine 15 oz ricotta, 1 egg, 1/2 cup Parmesan, 1/4 cup parsley, "
        "1/2 tsp salt, 1/4 tsp pepper. "
        "Step 11: Cook 12 lasagna noodles until al dente. Drain and rinse. "
        "Step 12-17: Layer sauce, noodles, ricotta, mozzarella (3 layers). "
        "Step 18: Cover with foil and bake 25 minutes. "
        "Step 19: Remove foil and bake 25-30 more minutes until golden. "
        "Step 20: Rest 15-20 minutes before slicing. "
        "Step 21: Garnish with fresh basil and serve hot."
    ),
}

HOST = "<GPU-HOST-D>"
PORT = 11434

MODELS = {
    "qwen3.5:0.8b-q8_0": {"model": "qwen3.5:0.8b-q8_0", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:0.8b-bf16": {"model": "qwen3.5:0.8b-bf16", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:2b-q4_K_M": {"model": "qwen3.5:2b-q4_K_M", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:2b-q8_0": {"model": "qwen3.5:2b-q8_0", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:2b-bf16": {"model": "qwen3.5:2b-bf16", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:4b-q4_K_M": {"model": "qwen3.5:4b-q4_K_M", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:4b-q8_0": {"model": "qwen3.5:4b-q8_0", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:4b-bf16": {"model": "qwen3.5:4b-bf16", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:9b-q4_K_M": {"model": "qwen3.5:9b", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:9b-q8_0": {"model": "qwen3.5:9b-q8_0", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:9b-bf16": {"model": "qwen3.5:9b-bf16", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:27b-q4_K_M": {"model": "qwen3.5:27b", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:27b-q8_0": {"model": "qwen3.5:27b-q8_0", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:27b-bf16": {"model": "qwen3.5:27b-bf16", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:35b-q4_K_M": {"model": "qwen3.5:35b", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:35b-q8_0": {"model": "qwen3.5:35b-a3b-q8_0", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:35b-bf16": {"model": "qwen3.5:35b-a3b-bf16", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:122b-q4_K_M": {"model": "qwen3.5:122b", "service_type": "ollama", "host": HOST, "port": PORT},
}

OUT_DIR = Path("results/qwen35_size_quant")


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
    model_info: dict,
    seed: int,
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


def identify_timeout_conditions(all_data: dict) -> list[str]:
    """Find condition keys that failed due to timeout (not empty-output errors)."""
    timeout_keys = []
    for key, iters in all_data.items():
        if len(iters) >= ITERATIONS:
            continue
        if not iters:
            continue
        last = iters[-1]
        error = last.get("error", "")
        if "timed out" in error or "timed_out" in error:
            timeout_keys.append(key)
    return sorted(timeout_keys)


def resolve_condition(key: str) -> tuple[dict, int, str] | None:
    """Resolve a condition key back to (model_info, seed, initial_prompt)."""
    # Try lasagna prefix
    if key.startswith("lasagna_"):
        rest = key[len("lasagna_"):]
        recipe = "lasagna"
    else:
        rest = key
        recipe = "spaghetti"

    # Extract seed from end: ..._s{seed}
    parts = rest.rsplit("_s", 1)
    if len(parts) != 2:
        return None
    model_label = parts[0]
    try:
        seed = int(parts[1])
    except ValueError:
        return None

    if model_label not in MODELS:
        return None

    return MODELS[model_label], seed, PROMPTS[recipe]


def main() -> None:
    data_path = OUT_DIR / "drift_data.json"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found", file=sys.stderr)
        sys.exit(1)

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

    # Load existing data
    existing = json.loads(data_path.read_text())
    all_data = existing.get("models", {})
    print(f"Loaded {len(all_data)} conditions from {data_path}")

    # Find timeout conditions
    timeout_keys = identify_timeout_conditions(all_data)
    print(f"Found {len(timeout_keys)} timed-out conditions to re-run:")
    for k in timeout_keys:
        n = len(all_data[k])
        print(f"  {k}: had {n} iters")

    if not timeout_keys:
        print("\nNothing to re-run.")
        return

    # Delete timed-out entries so they get re-run fresh
    for key in timeout_keys:
        del all_data[key]
    print(f"\nRemoved {len(timeout_keys)} incomplete entries. Running sequentially...\n")

    def save_json() -> None:
        save_data = {
            "initial_prompts": PROMPTS,
            "iterations": ITERATIONS,
            "prompt_style": PROMPT_STYLE,
            "temperature": TEMPERATURE,
            "seeds": SEEDS,
            "models_config": {label: info for label, info in MODELS.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": all_data,
        }
        data_path.write_text(json.dumps(save_data, indent=2))

    # Save with deletions first
    save_json()

    # Run each one sequentially
    completed = 0
    for key in timeout_keys:
        resolved = resolve_condition(key)
        if resolved is None:
            print(f"WARNING: Could not resolve {key}, skipping")
            continue

        model_info, seed, initial_prompt = resolved
        cond_key, results = run_single_chain(key, model_info, seed, initial_prompt)

        all_data[cond_key] = results
        completed += 1
        save_json()
        n_iters = len(results)
        last_status = results[-1]["status"] if results else "empty"
        print(f"\n  -> Saved {cond_key} ({completed}/{len(timeout_keys)}): "
              f"{n_iters} iters, last={last_status}")

    print(f"\n{'='*60}")
    print(f"Done. Re-ran {completed} conditions.")
    print(f"Total conditions in file: {len(all_data)}")


if __name__ == "__main__":
    main()
