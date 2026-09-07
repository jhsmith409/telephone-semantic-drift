#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Temperature ablation for Qwen3.5 models via vLLM.

5 temperatures × variable seeds = 21 runs per model, 30 iterations each.
T=0.0 gets 1 seed (deterministic), T=0.3/0.5/0.7/1.0 get 5 seeds each.
Auto-discovers the vLLM model on gpu-host-d ports 8000-8010.
Uses ThreadPoolExecutor(max_workers=4) for parallelism.
Resumable: skips condition keys already present in the data file.

Usage:
    nohup uv run python -u scripts/run_drift_experiment_temperature_qwen35.py \
        > results/drift_experiment_temperature_qwen35/<model>/run.log 2>&1 &
"""

from __future__ import annotations

import json
import math
import re
import socket
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


def sanitize_model_name(name: str) -> str:
    """Convert model name to a filesystem-safe string."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def discover_vllm(hostname: str = "gpu-host-d") -> tuple[str, int, str]:
    """Resolve hostname and probe ports 8000-8010 for a vLLM endpoint.

    Returns (host_ip, port, model_name).
    """
    print(f"Resolving {hostname}...")
    host_ip = socket.gethostbyname(hostname)
    print(f"  -> {host_ip}")

    print("Probing ports 8000-8010 for vLLM...")
    with httpx.Client(timeout=5.0) as client:
        for port in range(8000, 8011):
            try:
                resp = client.get(f"http://{host_ip}:{port}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    if models:
                        model_name = models[0]["id"]
                        print(f"  -> Found {model_name} on port {port}")
                        return host_ip, port, model_name
            except (httpx.ConnectError, httpx.ReadTimeout):
                continue

    print("ERROR: No vLLM endpoint found on gpu-host-d:8000-8010", file=sys.stderr)
    sys.exit(1)


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


def build_condition_matrix(model_label: str, model_info: dict) -> list[tuple[str, dict, int, float]]:
    """Build the condition matrix: 5 temps × variable seeds.

    T=0.0 gets 1 seed (deterministic), others get 5 seeds.
    Returns list of (condition_key, model_info, seed, temperature).
    """
    conditions = []
    for temp in TEMPERATURES:
        seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
        for seed in seeds_for_temp:
            key = f"{model_label}_t{temp:.1f}_s{seed}"
            conditions.append((key, model_info, seed, temp))
    return conditions


def main() -> None:
    # Discover vLLM model
    host_ip, port, model_name = discover_vllm("gpu-host-d")
    model_label = model_name
    safe_name = sanitize_model_name(model_name)

    model_info = {
        "model": model_name,
        "service_type": "openai",
        "host": host_ip,
        "port": port,
    }

    out_dir = Path(f"results/drift_experiment_temperature_qwen35/{safe_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "drift_data.json"

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
    conditions = build_condition_matrix(model_label, model_info)
    total = len(conditions)

    # Filter out already-completed conditions
    remaining = [(k, mi, s, t) for k, mi, s, t in conditions if k not in all_data]
    print(f"Total conditions: {total}")
    print(f"Already completed: {total - len(remaining)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Model: {model_name} on {host_ip}:{port}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Seeds: {SEEDS}")

    def _save_json() -> None:
        save_data = {
            "initial_prompt": INITIAL_PROMPT,
            "iterations": ITERATIONS,
            "prompt_style": PROMPT_STYLE,
            "temperatures": TEMPERATURES,
            "seeds": SEEDS,
            "model": model_name,
            "service_type": "openai",
            "host": host_ip,
            "port": port,
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
    print(f"SUMMARY — Mean Final Similarity by Temperature ({model_label})")
    print(f"{'='*70}")

    header = f"{'Temperature':<15}"
    header += f" {'Mean':>10} {'Std':>10} {'N seeds':>10}"
    print(header)
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    for temp in TEMPERATURES:
        seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
        finals = []
        for seed in seeds_for_temp:
            key = f"{model_label}_t{temp:.1f}_s{seed}"
            if key not in all_data:
                continue
            iters = all_data[key]
            sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
            if sims:
                finals.append(sims[-1])
        if finals:
            mean = sum(finals) / len(finals)
            std = (sum((x - mean) ** 2 for x in finals) / len(finals)) ** 0.5
            print(f"  T={temp:<11.1f} {mean:>10.4f} {std:>10.4f} {len(finals):>10d}")
        else:
            print(f"  T={temp:<11.1f} {'N/A':>10} {'N/A':>10} {'0':>10}")

    print(f"\nExperiment complete! {len(all_data)} total runs.")


if __name__ == "__main__":
    main()
