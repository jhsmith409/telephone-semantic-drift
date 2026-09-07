#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""NVFP4 experiment: Qwen3.5-35B-A3B-NVFP4 on sglang.

Runs Exp 1 (drift baseline) and Exp 2 (temperature ablation) for the NVFP4
quantization variant, writing results to separate data files.

Exp 1: 2 recipes x 5 seeds x 30 iters = 300 API calls
Exp 2: 5 temps x 2 recipes x 5 seeds x 30 iters = 1,260 API calls (T=0 -> 1 seed)

sglang endpoint: <GPU-HOST-C>:8005 (OpenAI-compatible, separate GPU).
Parallelism: 4 workers (sglang can handle concurrent requests).

Usage:
    nohup uv run python -u scripts/run_qwen35_nvfp4.py \
        > results/qwen35_nvfp4/run.log 2>&1 &
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
MAX_WORKERS = 4

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

MODEL_INFO = {
    "model": "txn545/Qwen3.5-35B-A3B-NVFP4",
    "service_type": "openai",
    "host": "<GPU-HOST-C>",
    "port": 8005,
}
MODEL_LABEL = "qwen3.5:35b-nvfp4"

# Output files — separate from Ollama data to avoid race conditions
EXP1_DATA_PATH = Path("results/qwen35_nvfp4/exp1_drift_data.json")
EXP2_DATA_PATH = Path("results/qwen35_nvfp4/exp2_drift_data.json")
LOG_DIR = Path("results/qwen35_nvfp4")

# Thread lock for safe JSON writes
_save_lock = threading.Lock()


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
    print(f"  seed={seed}, temp={temperature}")
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
                    timeout=1200.0,
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

            output = api_result["output_message"] or ""
            sim = get_similarity(client, initial_prompt, output if output else "")

            sim_str = f"{sim:.4f}" if sim is not None else "N/A"
            empty_tag = " [EMPTY]" if not output else ""
            print(f"  [{condition_key}] iter {i:2d}/{ITERATIONS}: sim={sim_str}{empty_tag}  ({elapsed:.1f}s)")

            results.append({
                "iteration": i,
                "status": "success",
                "input_message": current_message,
                "output_message": output,
                "cosine_similarity": sim,
                "elapsed_seconds": round(elapsed, 2),
            })

            # On empty output, keep previous message so chain can recover
            if output:
                current_message = output

    return condition_key, results


def load_or_init(data_path: Path) -> dict:
    """Load existing data file or return empty structure."""
    if data_path.exists():
        return json.loads(data_path.read_text())
    return {"models": {}}


def _save_json(data_path: Path, existing: dict, all_data: dict) -> None:
    """Thread-safe save."""
    with _save_lock:
        out = dict(existing)
        out["models"] = all_data
        out["timestamp"] = datetime.now(timezone.utc).isoformat()
        data_path.write_text(json.dumps(out, indent=2))


def is_complete(iters: list[dict]) -> bool:
    """A condition is complete if it has 30 successful iterations."""
    return len(iters) >= ITERATIONS and iters[-1].get("status") == "success"


def run_experiment(
    name: str,
    data_path: Path,
    conditions: list[tuple[str, int, float, str]],
) -> None:
    """Run an experiment with 4 parallel workers."""
    print(f"\n{'='*70}")
    print(f"NVFP4 {name}")
    print(f"{'='*70}")

    existing = load_or_init(data_path)
    all_data = existing.get("models", {})
    print(f"  Loaded {len(all_data)} existing conditions from {data_path}")

    # Delete incomplete conditions so they get re-run
    incomplete = [k for k, v in all_data.items() if not is_complete(v)]
    for k in incomplete:
        del all_data[k]
        print(f"  Deleted incomplete: {k}")
    if incomplete:
        _save_json(data_path, existing, all_data)

    remaining = [(k, s, t, p) for k, s, t, p in conditions if k not in all_data]
    print(f"  Total: {len(conditions)}, Remaining: {len(remaining)}")

    if not remaining:
        print(f"  All {name} conditions already complete.")
        return

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_single_chain, key, seed, temp, prompt): key
            for key, seed, temp, prompt in remaining
        }
        for future in as_completed(futures):
            cond_key, results = future.result()
            with _save_lock:
                all_data[cond_key] = results
                completed += 1
            _save_json(data_path, existing, all_data)
            n_iters = len(results)
            last_status = results[-1]["status"] if results else "empty"
            print(f"\n  -> Saved {cond_key} ({completed}/{len(remaining)}): "
                  f"{n_iters} iters, last={last_status}")

    print(f"\n  {name} done. {completed} conditions saved.")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

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

    # Build Exp 1 conditions: 2 recipes x 5 seeds at T=0.7
    exp1_conditions = []
    for recipe, prompt in PROMPTS.items():
        for seed in SEEDS:
            if recipe == "spaghetti":
                key = f"{MODEL_LABEL}_s{seed}"
            else:
                key = f"lasagna_{MODEL_LABEL}_s{seed}"
            exp1_conditions.append((key, seed, 0.7, prompt))

    # Build Exp 2 conditions: 5 temps x 2 recipes x seeds
    exp2_conditions = []
    for recipe, prompt in PROMPTS.items():
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            for seed in seeds_for_temp:
                if recipe == "spaghetti":
                    key = f"{MODEL_LABEL}_t{temp:.1f}_s{seed}"
                else:
                    key = f"lasagna_{MODEL_LABEL}_t{temp:.1f}_s{seed}"
                exp2_conditions.append((key, seed, temp, prompt))

    run_experiment("Experiment 1: Size/Quant Drift Baseline", EXP1_DATA_PATH, exp1_conditions)
    run_experiment("Experiment 2: Temperature Ablation", EXP2_DATA_PATH, exp2_conditions)

    print(f"\n{'='*70}")
    print("All NVFP4 experiments complete!")


if __name__ == "__main__":
    main()
