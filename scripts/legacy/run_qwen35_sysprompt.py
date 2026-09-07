#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Experiment 4: Qwen3.5 system prompt ablation.

20 system prompts × 2 recipes × 2 models (9B, 27B) × 5 seeds = 400 runs of 30 iterations.
Matches the original paper's RQ3 methodology (which used 30B AWQ + 8B q4_K_M).

Uses ThreadPoolExecutor(max_workers=2) for parallelism.
Resumable: skips condition keys already present in the data file.
Runs one model at a time to avoid Ollama model swapping.

Usage:
    nohup uv run python -u scripts/run_qwen35_sysprompt.py \
        > results/qwen35_sysprompt/run.log 2>&1 &
"""

from __future__ import annotations

import json
import math
import re
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

HOST = "<GPU-HOST-D>"
PORT = 11434

INITIAL_PROMPTS = {
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

SYSTEM_PROMPTS = [
    {"id": 1, "name": "Baseline - None", "category": "Baseline", "text": ""},
    {"id": 2, "name": "Minimal Helpful", "category": "Baseline", "text": "You are a helpful assistant."},
    {"id": 3, "name": "Strong Anti-Meta", "category": "Anti-Drift", "text": (
        "You are a precise recipe transmitter in a multi-agent kitchen pipeline. "
        "Your only job is to paraphrase the incoming message while preserving every "
        "single step and detail exactly. Never add, remove, or rephrase any instruction. "
        "Never add meta-commentary, explanations, or agent framing. Output only the "
        "paraphrased steps."
    )},
    {"id": 4, "name": "Ultra-Constrained Fidelity", "category": "Anti-Drift", "text": (
        "Maintain exact semantic meaning at all costs. Preserve every numbered step "
        "verbatim where possible. Never add fluff, commentary, or new information. "
        "Respond with nothing but the updated numbered list."
    )},
    {"id": 5, "name": "Expert VL Cooking Assistant", "category": "VL-Leveraged", "text": (
        "You are an expert visual cooking assistant. Paraphrase the recipe steps "
        "clearly and naturally while keeping every action intact. Use vivid but precise "
        "language that would work well in a visual tutorial. Do not add meta-notes or "
        "agent talk."
    )},
    {"id": 6, "name": "Deterministic Relay", "category": "Constrained", "text": (
        "You are a deterministic message relay. Re-express the input using different "
        "words but identical semantics and structure. Output format must be exactly: "
        "Step 1: …\nStep 2: …\nStep 3: … Nothing else."
    )},
    {"id": 7, "name": "100% Fidelity", "category": "Anti-Drift", "text": (
        "Preserve 100% of the original meaning and all steps. Do not omit, "
        "alter, or add any detail. Never include meta-commentary."
    )},
    {"id": 8, "name": "Numbered-Only Strict", "category": "Constrained", "text": (
        "Output ONLY a numbered list of the steps. No introductions, conclusions, "
        "explanations, or extra text of any kind."
    )},
    {"id": 9, "name": "Natural & Engaging", "category": "Creative", "text": (
        "Make this recipe message more natural and engaging for the next team member "
        "while preserving 100% of the original meaning and all steps. Keep the "
        "same numbered format. No introductions or conclusions."
    )},
    {"id": 10, "name": "Progressive Refinement", "category": "Creative", "text": (
        "You are part of a long-running chain of agents. Each paraphrase must be "
        "slightly clearer and easier to follow than the last while staying perfectly "
        "faithful to the original intent. Avoid repetition and any meta language."
    )},
    {"id": 11, "name": "Visual Tutorial Style", "category": "VL-Leveraged", "text": (
        "As a visual cooking expert, paraphrase the steps in a way that would pair "
        "perfectly with images or a cooking video. Keep every step intact and use "
        "language that helps someone visualize the actions."
    )},
    {"id": 12, "name": "Precise Transmitter", "category": "Anti-Drift", "text": (
        "You are a precise recipe transmitter. Paraphrase the message without losing "
        "or adding any information. Stay strictly factual and neutral."
    )},
    {"id": 13, "name": "Zero Fluff", "category": "Constrained", "text": (
        "Never add fluff, commentary, explanations, or new information. Stick strictly "
        "to rephrasing the existing steps only."
    )},
    {"id": 14, "name": "Locked Format", "category": "Constrained", "text": (
        "Always respond in the same numbered step format as the input and nothing else. "
        "Rephrase each step but preserve the exact structure."
    )},
    {"id": 15, "name": "Friendly Conversational", "category": "Creative", "text": (
        "Rephrase the recipe in a friendly, conversational tone while keeping all "
        "steps and details 100% accurate. Make it sound like helpful advice from one "
        "cook to another."
    )},
    {"id": 16, "name": "Multi-Agent Relay", "category": "Anti-Drift", "text": (
        "You are part of a multi-agent recipe relay system. Pass the message forward "
        "with improved clarity but zero loss of meaning. Never reference the chain or "
        "agents."
    )},
    {"id": 17, "name": "Think-Step-by-Step", "category": "Thinking", "text": (
        "Think step by step about the meaning of each instruction, then output only "
        "the final paraphrased numbered steps. Do not show your thinking."
    )},
    {"id": 18, "name": "Multimodal Visual Focus", "category": "VL-Leveraged", "text": (
        "You are a multimodal cooking assistant trained on thousands of visual recipes. "
        "Paraphrase the steps to make them extremely easy to visualize and follow in a "
        "real kitchen."
    )},
    {"id": 19, "name": "No Meta Ever", "category": "Anti-Drift", "text": (
        "Do not mention agents, pipelines, paraphrasing, instructions, or the task "
        "itself in any way. Just output the rephrased recipe steps cleanly."
    )},
    {"id": 20, "name": "Balanced Clarity", "category": "Creative", "text": (
        "Balance natural, readable language with perfect fidelity. Make the instructions "
        "clearer and more readable for the next person without changing any content or "
        "adding new ideas."
    )},
]

# Two models: medium (9B) and large (27B) — matches original paper's 8B vs 30B
MODELS = {
    "qwen3.5:9b": {"model": "qwen3.5:9b", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:27b": {"model": "qwen3.5:27b", "service_type": "ollama", "host": HOST, "port": PORT},
}

OUT_DIR = Path("results/qwen35_sysprompt")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


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
    system_prompt: str,
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


def build_condition_matrix() -> list[tuple[str, dict, str, int, str]]:
    """Build condition matrix: 2 models × 2 recipes × 20 prompts × 5 seeds = 400 runs.

    Key format: "{model_tag}_{recipe}_{id:02d}_{slug}_s{seed}"
    Returns list of (condition_key, model_info, system_prompt, seed, initial_prompt).
    """
    conditions = []
    for model_tag, model_info in MODELS.items():
        for recipe, initial_prompt in INITIAL_PROMPTS.items():
            for sp in SYSTEM_PROMPTS:
                slug = _slug(sp["name"])
                for seed in SEEDS:
                    key = f"{model_tag}_{recipe}_{sp['id']:02d}_{slug}_s{seed}"
                    conditions.append((key, model_info, sp["text"], seed, initial_prompt))
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

    remaining = [(k, mi, sp, s, ip) for k, mi, sp, s, ip in conditions if k not in all_data]
    print(f"Total conditions: {total}")
    print(f"Already completed: {total - len(remaining)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"System prompts: {len(SYSTEM_PROMPTS)}")
    print(f"Seeds: {SEEDS}")

    def _save_json() -> None:
        save_data = {
            "initial_prompts": INITIAL_PROMPTS,
            "iterations": ITERATIONS,
            "prompt_style": PROMPT_STYLE,
            "temperature": TEMPERATURE,
            "seeds": SEEDS,
            "models_config": {label: info for label, info in MODELS.items()},
            "system_prompts": SYSTEM_PROMPTS,
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
            key, mi, sp, seed, ip = args
            cond_key, results = run_single_chain(key, mi, sp, seed, ip)
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
    for model_tag in MODELS:
        for recipe in ["spaghetti", "lasagna"]:
            print(f"\n{'='*70}")
            print(f"SUMMARY — {model_tag} — {recipe.upper()}")
            print(f"{'='*70}")
            print(f"{'Prompt':<35} {'Category':<14} {'Mean':>8}")
            print(f"{'-'*35} {'-'*14} {'-'*8}")

            rows = []
            for sp in SYSTEM_PROMPTS:
                slug = _slug(sp["name"])
                finals = []
                for seed in SEEDS:
                    key = f"{model_tag}_{recipe}_{sp['id']:02d}_{slug}_s{seed}"
                    if key not in all_data:
                        continue
                    iters = all_data[key]
                    sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                    if sims:
                        finals.append(sims[-1])
                if finals:
                    mean = sum(finals) / len(finals)
                    rows.append((sp["name"], sp["category"], mean))

            rows.sort(key=lambda x: x[2], reverse=True)
            for name, cat, mean in rows:
                print(f"  {name:<33} {cat:<14} {mean:>8.4f}")

    print(f"\nExperiment complete! {len(all_data)} total runs.")


if __name__ == "__main__":
    main()
