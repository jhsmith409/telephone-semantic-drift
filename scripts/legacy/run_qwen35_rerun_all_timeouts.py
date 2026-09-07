#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Re-run ALL timed-out conditions across Exp 1, 2, and 4 (Ollama).

Scans 4 Ollama data files for conditions with <30 iterations that failed
with timeout errors, plus injects 6 missing Exp 1 spaghetti entries.
Processes everything sequentially (max_workers=1) to avoid GPU contention.

Order: Exp 1 size_quant -> Exp 2 temperature -> Exp 4 sysprompt

Usage:
    nohup uv run python -u scripts/run_qwen35_rerun_all_timeouts.py \
        > results/qwen35_rerun_all/run.log 2>&1 &
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BATCH_API_URL = "http://localhost:5001"
ITERATIONS = 30
PROMPT_STYLE = "paraphrase"
SEEDS = [42, 137, 256, 512, 1024]

HOST = "<GPU-HOST-D>"
PORT = 11434

# ── Prompts ──

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

# Exp 4 lasagna prompt (with newlines, as used in the sysprompt experiment)
EXP4_LASAGNA = (
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
)

# ── Exp 1 models (full set from run_qwen35_size_quant.py) ──

EXP1_MODELS = {
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

# ── Exp 2 models ──

EXP2_MODELS = {
    "qwen3.5:2b": {"model": "qwen3.5:2b-q4_K_M", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:9b": {"model": "qwen3.5:9b", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:27b": {"model": "qwen3.5:27b", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:35b": {"model": "qwen3.5:35b", "service_type": "ollama", "host": HOST, "port": PORT},
}
EXP2_TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]

# ── Exp 4 models + system prompts ──

EXP4_MODELS = {
    "qwen3.5:9b": {"model": "qwen3.5:9b", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:27b": {"model": "qwen3.5:27b", "service_type": "ollama", "host": HOST, "port": PORT},
}

SYSTEM_PROMPTS = {
    1: "", 2: "You are a helpful assistant.",
    3: (
        "You are a precise recipe transmitter in a multi-agent kitchen pipeline. "
        "Your only job is to paraphrase the incoming message while preserving every "
        "single step and detail exactly. Never add, remove, or rephrase any instruction. "
        "Never add meta-commentary, explanations, or agent framing. Output only the "
        "paraphrased steps."
    ),
    4: (
        "Maintain exact semantic meaning at all costs. Preserve every numbered step "
        "verbatim where possible. Never add fluff, commentary, or new information. "
        "Respond with nothing but the updated numbered list."
    ),
    5: (
        "You are an expert visual cooking assistant. Paraphrase the recipe steps "
        "clearly and naturally while keeping every action intact. Use vivid but precise "
        "language that would work well in a visual tutorial. Do not add meta-notes or "
        "agent talk."
    ),
    6: (
        "You are a deterministic message relay. Re-express the input using different "
        "words but identical semantics and structure. Output format must be exactly: "
        "Step 1: \u2026\nStep 2: \u2026\nStep 3: \u2026 Nothing else."
    ),
    7: (
        "Preserve 100% of the original meaning and all steps. Do not omit, "
        "alter, or add any detail. Never include meta-commentary."
    ),
    8: (
        "Output ONLY a numbered list of the steps. No introductions, conclusions, "
        "explanations, or extra text of any kind."
    ),
    9: (
        "Make this recipe message more natural and engaging for the next team member "
        "while preserving 100% of the original meaning and all steps. Keep the "
        "same numbered format. No introductions or conclusions."
    ),
    10: (
        "You are part of a long-running chain of agents. Each paraphrase must be "
        "slightly clearer and easier to follow than the last while staying perfectly "
        "faithful to the original intent. Avoid repetition and any meta language."
    ),
    11: (
        "As a visual cooking expert, paraphrase the steps in a way that would pair "
        "perfectly with images or a cooking video. Keep every step intact and use "
        "language that helps someone visualize the actions."
    ),
    12: (
        "You are a precise recipe transmitter. Paraphrase the message without losing "
        "or adding any information. Stay strictly factual and neutral."
    ),
    13: (
        "Never add fluff, commentary, explanations, or new information. Stick strictly "
        "to rephrasing the existing steps only."
    ),
    14: (
        "Always respond in the same numbered step format as the input and nothing else. "
        "Rephrase each step but preserve the exact structure."
    ),
    15: (
        "Rephrase the recipe in a friendly, conversational tone while keeping all "
        "steps and details 100% accurate. Make it sound like helpful advice from one "
        "cook to another."
    ),
    16: (
        "You are part of a multi-agent recipe relay system. Pass the message forward "
        "with improved clarity but zero loss of meaning. Never reference the chain or "
        "agents."
    ),
    17: (
        "Think step by step about the meaning of each instruction, then output only "
        "the final paraphrased numbered steps. Do not show your thinking."
    ),
    18: (
        "You are a multimodal cooking assistant trained on thousands of visual recipes. "
        "Paraphrase the steps to make them extremely easy to visualize and follow in a "
        "real kitchen."
    ),
    19: (
        "Do not mention agents, pipelines, paraphrasing, instructions, or the task "
        "itself in any way. Just output the rephrased recipe steps cleanly."
    ),
    20: (
        "Balance natural, readable language with perfect fidelity. Make the instructions "
        "clearer and more readable for the next person without changing any content or "
        "adding new ideas."
    ),
}

PROMPT_NAMES = {
    1: "Baseline - None", 2: "Minimal Helpful", 3: "Strong Anti-Meta",
    4: "Ultra-Constrained Fidelity", 5: "Expert VL Cooking Assistant",
    6: "Deterministic Relay", 7: "100% Fidelity", 8: "Numbered-Only Strict",
    9: "Natural & Engaging", 10: "Progressive Refinement",
    11: "Visual Tutorial Style", 12: "Precise Transmitter", 13: "Zero Fluff",
    14: "Locked Format", 15: "Friendly Conversational", 16: "Multi-Agent Relay",
    17: "Think-Step-by-Step", 18: "Multimodal Visual Focus", 19: "No Meta Ever",
    20: "Balanced Clarity",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# Build reverse lookup: slug -> system prompt id
_SLUG_TO_ID: dict[str, int] = {}
for _id, _name in PROMPT_NAMES.items():
    _SLUG_TO_ID[_slug(_name)] = _id


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
    system_prompt: str = "",
    temperature: float = 0.7,
) -> tuple[str, list[dict]]:
    print(f"\n{'='*60}")
    print(f"Condition: {condition_key}")
    print(f"  model={model_info['model']}, seed={seed}, temp={temperature}")
    if system_prompt:
        print(f"  system_prompt={system_prompt[:80]}...")
    print(f"{'='*60}")

    results = []
    current_message = initial_prompt

    with httpx.Client() as client:
        for i in range(1, ITERATIONS + 1):
            t0 = time.time()
            payload = {
                "message": current_message,
                "prompt_style": PROMPT_STYLE,
                "temperature": temperature,
                "seed": seed,
                **model_info,
            }
            if system_prompt:
                payload["system_prompt"] = system_prompt
            try:
                resp = client.post(
                    f"{BATCH_API_URL}/api/run",
                    json=payload,
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

            # Early-stop: 2 consecutive empty outputs = all remaining will be identical
            if (len(results) >= 2
                    and not results[-1]["output_message"]
                    and not results[-2]["output_message"]):
                print(f"  [{condition_key}] Early stop: 2 consecutive empty outputs at iter {i}")
                for j in range(i + 1, ITERATIONS + 1):
                    results.append({
                        "iteration": j,
                        "status": "success",
                        "input_message": current_message,
                        "output_message": "",
                        "cosine_similarity": sim,
                        "elapsed_seconds": 0,
                    })
                break

    return condition_key, results


def identify_incomplete_conditions(all_data: dict) -> list[str]:
    """Find condition keys that are incomplete (<30 successful iters)."""
    return sorted(
        k for k, v in all_data.items()
        if not v or len(v) < ITERATIONS or v[-1].get("status") != "success"
    )


# ── Exp 1 resolution ──

def resolve_exp1_condition(key: str) -> tuple[dict, int, str, str, float] | None:
    """Resolve Exp 1 key -> (model_info, seed, initial_prompt, system_prompt, temperature).

    Key formats:
      Spaghetti: "{model_label}_s{seed}"
      Lasagna:   "lasagna_{model_label}_s{seed}"
    """
    is_lasagna = key.startswith("lasagna_")
    rest = key[len("lasagna_"):] if is_lasagna else key

    # Extract seed from end
    parts = rest.rsplit("_s", 1)
    if len(parts) != 2:
        return None
    try:
        seed = int(parts[1])
    except ValueError:
        return None
    model_label = parts[0]

    if model_label not in EXP1_MODELS:
        return None

    recipe = "lasagna" if is_lasagna else "spaghetti"
    return EXP1_MODELS[model_label], seed, PROMPTS[recipe], "", 0.7


# ── Exp 2 resolution ──

def resolve_exp2_condition(key: str) -> tuple[dict, int, str, str, float] | None:
    """Resolve Exp 2 key -> (model_info, seed, initial_prompt, system_prompt, temperature).

    Key formats:
      Spaghetti: "{model_tag}_t{temp}_s{seed}"
      Lasagna:   "lasagna_{model_tag}_t{temp}_s{seed}"
    """
    is_lasagna = key.startswith("lasagna_")
    rest = key[len("lasagna_"):] if is_lasagna else key

    # Extract seed from end
    parts = rest.rsplit("_s", 1)
    if len(parts) != 2:
        return None
    try:
        seed = int(parts[1])
    except ValueError:
        return None

    # Extract temperature
    temp_parts = parts[0].rsplit("_t", 1)
    if len(temp_parts) != 2:
        return None
    model_tag = temp_parts[0]
    try:
        temperature = float(temp_parts[1])
    except ValueError:
        return None

    if model_tag not in EXP2_MODELS:
        return None

    recipe = "lasagna" if is_lasagna else "spaghetti"
    return EXP2_MODELS[model_tag], seed, PROMPTS[recipe], "", temperature


# ── Exp 4 resolution ──

def resolve_exp4_condition(key: str) -> tuple[dict, int, str, str, float] | None:
    """Resolve Exp 4 key -> (model_info, seed, initial_prompt, system_prompt, temperature).

    Key format: "{model_tag}_{recipe}_{id:02d}_{slug}_s{seed}"
    """
    # Try each model prefix
    model_tag = None
    rest = None
    for tag in EXP4_MODELS:
        prefix = f"{tag}_"
        if key.startswith(prefix):
            model_tag = tag
            rest = key[len(prefix):]
            break

    if model_tag is None or rest is None:
        return None

    # Extract seed from end
    parts = rest.rsplit("_s", 1)
    if len(parts) != 2:
        return None
    try:
        seed = int(parts[1])
    except ValueError:
        return None
    rest = parts[0]

    # Extract recipe
    recipe = None
    for r in ("spaghetti", "lasagna"):
        recipe_prefix = f"{r}_"
        if rest.startswith(recipe_prefix):
            recipe = r
            rest = rest[len(recipe_prefix):]
            break

    if recipe is None:
        return None

    # Extract prompt ID: "{id:02d}_{slug}"
    id_match = re.match(r"^(\d{2})_(.+)$", rest)
    if not id_match:
        return None
    prompt_id = int(id_match.group(1))

    if prompt_id not in SYSTEM_PROMPTS:
        print(f"  WARNING: prompt_id {prompt_id} not found for key {key}")
        return None

    system_prompt = SYSTEM_PROMPTS[prompt_id]
    model_info = EXP4_MODELS[model_tag]

    # Use correct initial prompt for recipe
    if recipe == "lasagna":
        initial_prompt = EXP4_LASAGNA
    else:
        initial_prompt = PROMPTS["spaghetti"]

    return model_info, seed, initial_prompt, system_prompt, 0.7


# ── Missing Exp 1 spaghetti entries ──

MISSING_EXP1_SPAGHETTI = [
    ("qwen3.5:27b-bf16_s1024", "qwen3.5:27b-bf16", 1024),
    ("qwen3.5:27b-bf16_s137", "qwen3.5:27b-bf16", 137),
    ("qwen3.5:27b-bf16_s256", "qwen3.5:27b-bf16", 256),
    ("qwen3.5:27b-bf16_s512", "qwen3.5:27b-bf16", 512),
    ("qwen3.5:27b-q4_K_M_s137", "qwen3.5:27b-q4_K_M", 137),
    ("qwen3.5:27b-q4_K_M_s42", "qwen3.5:27b-q4_K_M", 42),
]


def process_dataset(
    data_path: Path,
    resolve_fn,
    dataset_name: str,
    extra_conditions: list[tuple[str, dict, int, str, str, float]] | None = None,
) -> int:
    """Process a single dataset: identify timeouts, delete, re-run, save.

    Returns number of conditions re-run.
    """
    if not data_path.exists():
        print(f"  SKIP: {data_path} not found")
        return 0

    existing = json.loads(data_path.read_text())
    all_data = existing.get("models", {})
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"  Loaded {len(all_data)} conditions from {data_path}")

    timeout_keys = identify_incomplete_conditions(all_data)
    print(f"  Found {len(timeout_keys)} incomplete conditions:")
    for k in timeout_keys:
        n = len(all_data.get(k, []))
        print(f"    {k}: had {n} iters")

    # Collect extra missing conditions
    extra_keys = []
    if extra_conditions:
        for key, model_info, seed, prompt, sysprompt, temp in extra_conditions:
            if key not in all_data:
                extra_keys.append(key)
                print(f"    {key}: MISSING (will inject)")

    total_to_run = len(timeout_keys) + len(extra_keys)
    if total_to_run == 0:
        print("  Nothing to re-run.")
        return 0

    # Delete timed-out entries
    for key in timeout_keys:
        del all_data[key]
    print(f"\n  Removed {len(timeout_keys)} incomplete entries. "
          f"Total to run: {total_to_run} (sequential)...\n")

    def save_json() -> None:
        save_data = dict(existing)
        save_data["models"] = all_data
        save_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data_path.write_text(json.dumps(save_data, indent=2))

    # Save with deletions first
    save_json()

    completed = 0

    # Re-run timeout conditions
    for key in timeout_keys:
        resolved = resolve_fn(key)
        if resolved is None:
            print(f"  WARNING: Could not resolve {key}, skipping")
            continue

        model_info, seed, initial_prompt, system_prompt, temperature = resolved
        cond_key, results = run_single_chain(
            key, model_info, seed, initial_prompt, system_prompt, temperature
        )

        all_data[cond_key] = results
        completed += 1
        save_json()
        n_iters = len(results)
        last_status = results[-1]["status"] if results else "empty"
        print(f"\n  -> Saved {cond_key} ({completed}/{total_to_run}): "
              f"{n_iters} iters, last={last_status}")

    # Run extra missing conditions
    if extra_conditions:
        for key, model_info, seed, initial_prompt, system_prompt, temperature in extra_conditions:
            if key in all_data:
                continue  # Already exists (maybe completed above)
            cond_key, results = run_single_chain(
                key, model_info, seed, initial_prompt, system_prompt, temperature
            )
            all_data[cond_key] = results
            completed += 1
            save_json()
            n_iters = len(results)
            last_status = results[-1]["status"] if results else "empty"
            print(f"\n  -> Saved {cond_key} ({completed}/{total_to_run}): "
                  f"{n_iters} iters, last={last_status}")

    print(f"\n  Done {dataset_name}. Re-ran {completed} conditions.")
    print(f"  Total conditions in file: {len(all_data)}")
    return completed


def main() -> None:
    Path("results/qwen35_rerun_all").mkdir(parents=True, exist_ok=True)

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

    total_rerun = 0

    # ── Exp 1: size_quant (timeouts + 6 missing spaghetti) ──
    missing_exp1 = [
        (key, EXP1_MODELS[model_label], seed, PROMPTS["spaghetti"], "", 0.7)
        for key, model_label, seed in MISSING_EXP1_SPAGHETTI
    ]
    total_rerun += process_dataset(
        data_path=Path("results/qwen35_size_quant/drift_data.json"),
        resolve_fn=resolve_exp1_condition,
        dataset_name="Exp 1 size_quant (timeouts + missing)",
        extra_conditions=missing_exp1,
    )

    # ── Exp 2: temperature ──
    total_rerun += process_dataset(
        data_path=Path("results/qwen35_temperature/drift_data.json"),
        resolve_fn=resolve_exp2_condition,
        dataset_name="Exp 2 temperature",
    )

    # ── Exp 4: sysprompt (timeouts + missing conditions for 9b/27b) ──
    missing_exp4 = []
    for model_tag, model_info in EXP4_MODELS.items():
        for recipe in ("spaghetti", "lasagna"):
            for prompt_id, system_prompt in SYSTEM_PROMPTS.items():
                slug = _slug(PROMPT_NAMES[prompt_id])
                for seed in SEEDS:
                    key = f"{model_tag}_{recipe}_{prompt_id:02d}_{slug}_s{seed}"
                    initial_prompt = EXP4_LASAGNA if recipe == "lasagna" else PROMPTS["spaghetti"]
                    missing_exp4.append((key, model_info, seed, initial_prompt, system_prompt, 0.7))

    total_rerun += process_dataset(
        data_path=Path("results/qwen35_sysprompt/drift_data.json"),
        resolve_fn=resolve_exp4_condition,
        dataset_name="Exp 4 sysprompt (timeouts + missing)",
        extra_conditions=missing_exp4,
    )

    print(f"\n{'='*70}")
    print(f"All done! Re-ran {total_rerun} total conditions.")


if __name__ == "__main__":
    main()
