#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Run system prompt ablation drift experiment with lasagna prompt: 20 prompts × 5 seeds.

Same as run_drift_experiment_sysprompt.py but with a 21-step lasagna recipe
instead of the 3-step pasta prompt, and no 2x emphasis level.
20 prompts × 5 seeds = 100 runs of 30 iterations each (3,000 API calls).

Usage:
    nohup uv run python -u scripts/run_drift_experiment_sysprompt_lasagna.py &
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
            "Preserve 100% of the original meaning and all steps. Do not omit, "
            "alter, or add any detail. Never include meta-commentary."
        ),
    },
    {
        "id": 8,
        "name": "Numbered-Only Strict",
        "category": "Constrained",
        "text": (
            "Output ONLY a numbered list of the steps. No introductions, conclusions, "
            "explanations, or extra text of any kind."
        ),
    },
    {
        "id": 9,
        "name": "Natural & Engaging",
        "category": "Creative",
        "text": (
            "Make this recipe message more natural and engaging for the next team member "
            "while preserving 100% of the original meaning and all steps. Keep the "
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
            "to rephrasing the existing steps only."
        ),
    },
    {
        "id": 14,
        "name": "Locked Format",
        "category": "Constrained",
        "text": (
            "Always respond in the same numbered step format as the input and nothing else. "
            "Rephrase each step but preserve the exact structure."
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

OUT_DIR = Path("results/drift_experiment_sysprompt_lasagna")


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
    """Build the condition matrix: 20 prompts × 5 seeds (1x only).

    Returns list of (condition_key, model_info, system_prompt, seed).
    """
    conditions = []
    for prompt in PROMPTS:
        slug = _slug(prompt["name"])
        base_key = f"{prompt['id']:02d}_{slug}"
        for seed in SEEDS:
            key = f"{base_key}_s{seed}"
            conditions.append((key, model_info, prompt["text"], seed))

    return conditions


def plot_line(all_data: dict, out_path: Path) -> None:
    """Generate a line plot with mean +/- min/max bands across 5 seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(16, 9))

    for prompt in PROMPTS:
        slug = _slug(prompt["name"])
        base_key = f"{prompt['id']:02d}_{slug}"

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
        f"System Prompt Ablation — Lasagna Recipe (paraphrase, t=0.7, 5 seeds)\n"
        f"20 Prompts × {ITERATIONS} Iterations",
        fontsize=13,
    )
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    handles, labels = ax.get_legend_handles_labels()
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


def plot_bar(all_data: dict, out_path: Path) -> None:
    """Generate bar chart of mean final similarity for all 20 prompts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    prompt_stats = []  # (name, category, mean, std)

    for prompt in PROMPTS:
        slug = _slug(prompt["name"])
        base_key = f"{prompt['id']:02d}_{slug}"

        finals = []
        for seed in SEEDS:
            key = f"{base_key}_s{seed}"
            if key not in all_data:
                continue
            iters = all_data[key]
            sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
            if sims:
                finals.append(sims[-1])

        mean = np.mean(finals) if finals else 0
        std = np.std(finals) if finals else 0
        prompt_stats.append((prompt["name"], prompt["category"], mean, std))

    # Sort by mean descending
    prompt_stats.sort(key=lambda x: x[2], reverse=True)

    names = [s[0] for s in prompt_stats]
    categories = [s[1] for s in prompt_stats]
    means = [s[2] for s in prompt_stats]
    stds = [s[3] for s in prompt_stats]

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = [CATEGORY_COLORS.get(c, "#888888") for c in categories]

    ax.bar(x, means, yerr=stds, capsize=3,
           color=colors, edgecolor="white", linewidth=0.5)

    # Baseline reference line
    baseline_mean = None
    for name, _, m, _ in prompt_stats:
        if name == "Baseline - None":
            baseline_mean = m
            break
    if baseline_mean is not None:
        ax.axhline(y=baseline_mean, color="#888888", linestyle="--", alpha=0.7,
                    label=f"Baseline ({baseline_mean:.2f})")

    ax.set_xlabel("System Prompt", fontsize=11)
    ax.set_ylabel("Mean Final Cosine Similarity (± 1 std)", fontsize=11)
    ax.set_title(
        f"System Prompt Ablation — Lasagna Recipe\n"
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
    if data_path.exists():
        existing = json.loads(data_path.read_text())
        all_data = existing.get("models", {})
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
        data_lock = threading.Lock()
        completed = [0]

        def _run_and_save(args: tuple) -> str:
            key, mi, sp, seed = args
            cond_key, results = run_single_chain(key, mi, sp, seed)

            with data_lock:
                all_data[cond_key] = results
                completed[0] += 1

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

    # Final save
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
    plot_line(all_data, OUT_DIR / "drift_plot.png")
    plot_bar(all_data, OUT_DIR / "drift_bar.png")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY — Mean Final Similarity by Prompt")
    print(f"{'='*60}")
    print(f"{'Prompt':<35} {'Category':<14} {'Mean':>8}")
    print(f"{'-'*35} {'-'*14} {'-'*8}")

    for prompt in PROMPTS:
        slug = _slug(prompt["name"])
        base_key = f"{prompt['id']:02d}_{slug}"
        finals = []
        for seed in SEEDS:
            key = f"{base_key}_s{seed}"
            if key not in all_data:
                continue
            iters = all_data[key]
            sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
            if sims:
                finals.append(sims[-1])

        mean = sum(finals) / len(finals) if finals else 0
        print(f"  {prompt['name']:<33} {prompt['category']:<14} {mean:>8.4f}")

    print(f"\nExperiment complete!")


if __name__ == "__main__":
    main()
