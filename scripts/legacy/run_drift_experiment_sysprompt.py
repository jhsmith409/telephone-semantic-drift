#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Run system prompt ablation drift experiment: 20 prompts × 2 emphasis × 5 seeds.

Holds the model constant (qwen3-vl:30b on vLLM/gpu-host-d) and varies the system
prompt to measure how prompt engineering affects drift stability. Auto-discovers
the vLLM endpoint on gpu-host-d ports 8000-8010.

200 runs of 30 iterations each (6,000 API calls).
Uses ThreadPoolExecutor(max_workers=4) for parallelism across conditions.
Resumable: skips condition+seed combos already present in the data file.

Usage:
    nohup uv run python -u scripts/run_drift_experiment_sysprompt.py &
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
TEMPERATURE = 0.7
SEEDS = [42, 137, 256, 512, 1024]

INITIAL_PROMPT = (
    "Step 1: Boil water. Step 2: Add pasta for 8 minutes. "
    "Step 3: Drain and serve with sauce."
)

PROMPTS = [
    {
        "id": 1,
        "name": "Baseline - None",
        "category": "Baseline",
        "text": "",
    },
    {
        "id": 2,
        "name": "Minimal Helpful",
        "category": "Baseline",
        "text": "You are a helpful assistant.",
    },
    {
        "id": 3,
        "name": "Strong Anti-Meta",
        "category": "Anti-Drift",
        "text": (
            "You are a precise recipe transmitter in a multi-agent kitchen pipeline. "
            "Your only job is to paraphrase the incoming message while preserving every "
            "single step and detail exactly. Never add, remove, or rephrase any instruction. "
            "Never add meta-commentary, explanations, or agent framing. Output only the "
            "paraphrased steps."
        ),
    },
    {
        "id": 4,
        "name": "Ultra-Constrained Fidelity",
        "category": "Anti-Drift",
        "text": (
            "Maintain exact semantic meaning at all costs. Preserve every numbered step "
            "verbatim where possible. Never add fluff, commentary, or new information. "
            "Respond with nothing but the updated numbered list."
        ),
    },
    {
        "id": 5,
        "name": "Expert VL Cooking Assistant",
        "category": "VL-Leveraged",
        "text": (
            "You are an expert visual cooking assistant. Paraphrase the recipe steps "
            "clearly and naturally while keeping every action intact. Use vivid but precise "
            "language that would work well in a visual tutorial. Do not add meta-notes or "
            "agent talk."
        ),
    },
    {
        "id": 6,
        "name": "Deterministic Relay",
        "category": "Constrained",
        "text": (
            "You are a deterministic message relay. Re-express the input using different "
            "words but identical semantics and structure. Output format must be exactly: "
            "Step 1: …\nStep 2: …\nStep 3: … Nothing else."
        ),
    },
    {
        "id": 7,
        "name": "100% Fidelity",
        "category": "Anti-Drift",
        "text": (
            "Preserve 100% of the original meaning and all three steps. Do not omit, "
            "alter, or add any detail. Never include meta-commentary."
        ),
    },
    {
        "id": 8,
        "name": "Numbered-Only Strict",
        "category": "Constrained",
        "text": (
            "Output ONLY a numbered list of the three steps. No introductions, conclusions, "
            "explanations, or extra text of any kind."
        ),
    },
    {
        "id": 9,
        "name": "Natural & Engaging",
        "category": "Creative",
        "text": (
            "Make this recipe message more natural and engaging for the next team member "
            "while preserving 100% of the original meaning and all three steps. Keep the "
            "same numbered format. No introductions or conclusions."
        ),
    },
    {
        "id": 10,
        "name": "Progressive Refinement",
        "category": "Creative",
        "text": (
            "You are part of a long-running chain of agents. Each paraphrase must be "
            "slightly clearer and easier to follow than the last while staying perfectly "
            "faithful to the original intent. Avoid repetition and any meta language."
        ),
    },
    {
        "id": 11,
        "name": "Visual Tutorial Style",
        "category": "VL-Leveraged",
        "text": (
            "As a visual cooking expert, paraphrase the steps in a way that would pair "
            "perfectly with images or a cooking video. Keep every step intact and use "
            "language that helps someone visualize the actions."
        ),
    },
    {
        "id": 12,
        "name": "Precise Transmitter",
        "category": "Anti-Drift",
        "text": (
            "You are a precise recipe transmitter. Paraphrase the message without losing "
            "or adding any information. Stay strictly factual and neutral."
        ),
    },
    {
        "id": 13,
        "name": "Zero Fluff",
        "category": "Constrained",
        "text": (
            "Never add fluff, commentary, explanations, or new information. Stick strictly "
            "to rephrasing the existing three steps only."
        ),
    },
    {
        "id": 14,
        "name": "Locked Format",
        "category": "Constrained",
        "text": (
            "Always respond in this exact format and nothing else:\n"
            "Step 1: [rephrased step]\n"
            "Step 2: [rephrased step]\n"
            "Step 3: [rephrased step]"
        ),
    },
    {
        "id": 15,
        "name": "Friendly Conversational",
        "category": "Creative",
        "text": (
            "Rephrase the recipe in a friendly, conversational tone while keeping all "
            "steps and details 100% accurate. Make it sound like helpful advice from one "
            "cook to another."
        ),
    },
    {
        "id": 16,
        "name": "Multi-Agent Relay",
        "category": "Anti-Drift",
        "text": (
            "You are part of a multi-agent recipe relay system. Pass the message forward "
            "with improved clarity but zero loss of meaning. Never reference the chain or "
            "agents."
        ),
    },
    {
        "id": 17,
        "name": "Think-Step-by-Step",
        "category": "Thinking",
        "text": (
            "Think step by step about the meaning of each instruction, then output only "
            "the final paraphrased numbered steps. Do not show your thinking."
        ),
    },
    {
        "id": 18,
        "name": "Multimodal Visual Focus",
        "category": "VL-Leveraged",
        "text": (
            "You are a multimodal cooking assistant trained on thousands of visual recipes. "
            "Paraphrase the steps to make them extremely easy to visualize and follow in a "
            "real kitchen."
        ),
    },
    {
        "id": 19,
        "name": "No Meta Ever",
        "category": "Anti-Drift",
        "text": (
            "Do not mention agents, pipelines, paraphrasing, instructions, or the task "
            "itself in any way. Just output the rephrased recipe steps cleanly."
        ),
    },
    {
        "id": 20,
        "name": "Balanced Clarity",
        "category": "Creative",
        "text": (
            "Balance natural, readable language with perfect fidelity. Make the instructions "
            "clearer and more readable for the next person without changing any content or "
            "adding new ideas."
        ),
    },
]

CATEGORY_COLORS = {
    "Baseline": "#888888",
    "Anti-Drift": "#4488ff",
    "Constrained": "#44bb44",
    "Creative": "#ff8844",
    "VL-Leveraged": "#aa44ff",
    "Thinking": "#ff4444",
}

OUT_DIR = Path("results/drift_experiment_sysprompt")


def _slug(name: str) -> str:
    """Convert prompt name to a URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def discover_vllm(hostname: str = "gpu-host-d") -> tuple[str, int, str]:
    """Resolve hostname and probe ports 8000-8010 for a vLLM endpoint.

    Returns (host_ip, port, model_name).
    """
    print(f"Resolving {hostname}...")
    host_ip = socket.gethostbyname(hostname)
    print(f"  → {host_ip}")

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
                        print(f"  → Found {model_name} on port {port}")
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
    system_prompt: str,
    seed: int,
) -> tuple[str, list[dict]]:
    """Run 30 iterations for one condition+seed, returning (key, results)."""
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
                        "system_prompt": system_prompt,
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


def build_condition_matrix(model_info: dict) -> list[tuple[str, dict, str, int]]:
    """Build the full condition matrix: 20 prompts × 2 emphasis × 5 seeds.

    Returns list of (condition_key, model_info, system_prompt, seed).
    """
    conditions = []
    for prompt in PROMPTS:
        slug = _slug(prompt["name"])
        text_1x = prompt["text"]
        text_2x = (prompt["text"] + "\n\n" + prompt["text"]) if prompt["text"] else ""

        for emphasis, text in [("1x", text_1x), ("2x", text_2x)]:
            base_key = f"{prompt['id']:02d}_{slug}_{emphasis}"
            for seed in SEEDS:
                key = f"{base_key}_s{seed}"
                conditions.append((key, model_info, text, seed))

    return conditions


def plot_line(all_data: dict, emphasis: str, out_path: Path) -> None:
    """Generate a line plot with mean ± min/max bands for one emphasis level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(16, 9))

    # Group data by prompt (across seeds)
    for prompt in PROMPTS:
        slug = _slug(prompt["name"])
        base_key = f"{prompt['id']:02d}_{slug}_{emphasis}"

        # Collect per-seed similarity arrays
        seed_curves = []
        for seed in SEEDS:
            key = f"{base_key}_s{seed}"
            if key not in all_data:
                continue
            iters = all_data[key]
            sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
            if sims:
                seed_curves.append(sims)

        if not seed_curves:
            continue

        # Align to shortest length
        min_len = min(len(c) for c in seed_curves)
        aligned = np.array([c[:min_len] for c in seed_curves])
        xs = np.arange(1, min_len + 1)

        mean_curve = aligned.mean(axis=0)
        min_curve = aligned.min(axis=0)
        max_curve = aligned.max(axis=0)

        color = CATEGORY_COLORS.get(prompt["category"], "#888888")
        final_mean = mean_curve[-1] if len(mean_curve) > 0 else 0
        label = f"{prompt['name']} ({final_mean:.2f})"

        ax.plot(xs, mean_curve, color=color, linewidth=1.5, label=label)
        ax.fill_between(xs, min_curve, max_curve, color=color, alpha=0.2)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax.set_title(
        f"System Prompt Ablation — {emphasis} Emphasis (paraphrase, t=0.7, 5 seeds)\n"
        f"20 Prompts × {ITERATIONS} Iterations",
        fontsize=13,
    )
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    # Sort legend by final mean similarity descending
    handles, labels = ax.get_legend_handles_labels()
    # Extract the numeric value from the label for sorting
    def _sort_key(label):
        m = re.search(r"\(([\d.]+)\)$", label)
        return float(m.group(1)) if m else 0
    sorted_pairs = sorted(zip(handles, labels), key=lambda p: _sort_key(p[1]), reverse=True)
    if sorted_pairs:
        handles, labels = zip(*sorted_pairs)
    ax.legend(handles, labels, fontsize=7, loc="lower left", ncol=2)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_bar_comparison(all_data: dict, out_path: Path) -> None:
    """Generate grouped bar chart comparing 1x vs 2x for all 20 prompts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    prompt_stats = []  # (name, category, mean_1x, std_1x, mean_2x, std_2x)

    for prompt in PROMPTS:
        slug = _slug(prompt["name"])

        finals = {"1x": [], "2x": []}
        for emphasis in ("1x", "2x"):
            base_key = f"{prompt['id']:02d}_{slug}_{emphasis}"
            for seed in SEEDS:
                key = f"{base_key}_s{seed}"
                if key not in all_data:
                    continue
                iters = all_data[key]
                sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                if sims:
                    finals[emphasis].append(sims[-1])

        mean_1x = np.mean(finals["1x"]) if finals["1x"] else 0
        std_1x = np.std(finals["1x"]) if finals["1x"] else 0
        mean_2x = np.mean(finals["2x"]) if finals["2x"] else 0
        std_2x = np.std(finals["2x"]) if finals["2x"] else 0

        prompt_stats.append((prompt["name"], prompt["category"], mean_1x, std_1x, mean_2x, std_2x))

    # Sort by 1x mean descending
    prompt_stats.sort(key=lambda x: x[2], reverse=True)

    names = [s[0] for s in prompt_stats]
    categories = [s[1] for s in prompt_stats]
    means_1x = [s[2] for s in prompt_stats]
    stds_1x = [s[3] for s in prompt_stats]
    means_2x = [s[4] for s in prompt_stats]
    stds_2x = [s[5] for s in prompt_stats]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    colors_1x = [CATEGORY_COLORS.get(c, "#888888") for c in categories]
    colors_2x = [CATEGORY_COLORS.get(c, "#888888") for c in categories]

    bars_1x = ax.bar(x - width / 2, means_1x, width, yerr=stds_1x, capsize=3,
                     color=colors_1x, label="1x", edgecolor="white", linewidth=0.5)
    bars_2x = ax.bar(x + width / 2, means_2x, width, yerr=stds_2x, capsize=3,
                     color=colors_2x, label="2x", alpha=0.6, hatch="//",
                     edgecolor="white", linewidth=0.5)

    # Baseline reference line (prompt #1 = first in sorted order may not be baseline)
    baseline_mean = None
    for name, _, m1x, _, _, _ in prompt_stats:
        if name == "Baseline - None":
            baseline_mean = m1x
            break
    if baseline_mean is not None:
        ax.axhline(y=baseline_mean, color="#888888", linestyle="--", alpha=0.7, label=f"Baseline ({baseline_mean:.2f})")

    ax.set_xlabel("System Prompt", fontsize=11)
    ax.set_ylabel("Mean Final Cosine Similarity (± 1 std)", fontsize=11)
    ax.set_title(
        "System Prompt Ablation — 1x vs 2x Emphasis\n"
        f"Mean Final Similarity ± std across {len(SEEDS)} seeds",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / "drift_data.json"

    # Auto-discover vLLM on gpu-host-d
    host_ip, port, model_name = discover_vllm("gpu-host-d")
    model_info = {
        "model": model_name,
        "service_type": "openai",
        "host": host_ip,
        "port": port,
    }

    # Wait for batch API
    print(f"\nWaiting for batch API at {BATCH_API_URL}...")
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
    metadata: dict = {}
    if data_path.exists():
        existing = json.loads(data_path.read_text())
        all_data = existing.get("models", {})
        metadata = {k: v for k, v in existing.items() if k != "models"}
        print(f"Loaded {len(all_data)} existing runs from {data_path}")

    # Build condition matrix
    conditions = build_condition_matrix(model_info)
    total = len(conditions)

    # Filter out already-completed conditions
    remaining = [(k, mi, sp, s) for k, mi, sp, s in conditions if k not in all_data]
    print(f"Total conditions: {total}")
    print(f"Already completed: {total - len(remaining)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Model: {model_name} on {host_ip}:{port}")

    if not remaining:
        print("\nAll conditions already completed. Generating plots...")
    else:
        # Thread-safe data access
        data_lock = threading.Lock()
        completed = [0]

        def _run_and_save(args: tuple) -> str:
            key, mi, sp, seed = args
            cond_key, results = run_single_chain(key, mi, sp, seed)

            with data_lock:
                all_data[cond_key] = results
                completed[0] += 1

                # Save after each run for resumability
                save_data = {
                    "initial_prompt": INITIAL_PROMPT,
                    "iterations": ITERATIONS,
                    "prompt_style": PROMPT_STYLE,
                    "temperature": TEMPERATURE,
                    "seeds": SEEDS,
                    "model": model_name,
                    "service_type": "openai",
                    "host": host_ip,
                    "port": port,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompts": PROMPTS,
                    "models": all_data,
                }
                data_path.write_text(json.dumps(save_data, indent=2))

                print(f"\n  ✓ Saved {cond_key} ({completed[0]}/{len(remaining)} done)")

            return cond_key

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run_and_save, args): args[0] for args in remaining}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nERROR: {key} failed: {exc}", file=sys.stderr)

    # Final save with updated timestamp
    save_data = {
        "initial_prompt": INITIAL_PROMPT,
        "iterations": ITERATIONS,
        "prompt_style": PROMPT_STYLE,
        "temperature": TEMPERATURE,
        "seeds": SEEDS,
        "model": model_name,
        "service_type": "openai",
        "host": host_ip,
        "port": port,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompts": PROMPTS,
        "models": all_data,
    }
    data_path.write_text(json.dumps(save_data, indent=2))
    print(f"\nData saved: {data_path}")

    # Generate plots
    plot_line(all_data, "1x", OUT_DIR / "drift_plot_1x.png")
    plot_line(all_data, "2x", OUT_DIR / "drift_plot_2x.png")
    plot_bar_comparison(all_data, OUT_DIR / "drift_bar_comparison.png")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY — Mean Final Similarity by Prompt")
    print(f"{'='*70}")
    print(f"{'Prompt':<35} {'Category':<14} {'1x Mean':>8} {'2x Mean':>8}")
    print(f"{'-'*35} {'-'*14} {'-'*8} {'-'*8}")

    for prompt in PROMPTS:
        slug = _slug(prompt["name"])
        finals = {"1x": [], "2x": []}
        for emphasis in ("1x", "2x"):
            base_key = f"{prompt['id']:02d}_{slug}_{emphasis}"
            for seed in SEEDS:
                key = f"{base_key}_s{seed}"
                if key not in all_data:
                    continue
                iters = all_data[key]
                sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                if sims:
                    finals[emphasis].append(sims[-1])

        mean_1x = sum(finals["1x"]) / len(finals["1x"]) if finals["1x"] else 0
        mean_2x = sum(finals["2x"]) / len(finals["2x"]) if finals["2x"] else 0
        print(f"  {prompt['name']:<33} {prompt['category']:<14} {mean_1x:>8.4f} {mean_2x:>8.4f}")

    print(f"\nExperiment complete!")


if __name__ == "__main__":
    main()
