#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Re-run 27b timed-out conditions from Exp 4 (sysprompt) and Exp 1 (lasagna).

Identifies 27b entries with <30 iterations that failed with timeout errors,
removes them from the data, then re-runs them sequentially (max_workers=1)
to avoid GPU contention.

Handles two datasets:
  - results/qwen35_sysprompt/drift_data.json (Exp 4: 198 27b timeouts)
  - results/qwen35_size_quant/drift_data.json (Exp 1: 15 27b lasagna timeouts)

Usage:
    uv run python -u scripts/run_qwen35_rerun_27b_timeouts.py
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
TEMPERATURE = 0.7
SEEDS = [42, 137, 256, 512, 1024]

HOST = "<GPU-HOST-D>"
PORT = 11434

# ── Prompts (shared across both experiments) ──

PROMPTS = {
    "spaghetti": (
        "Step 1: Boil water. Step 2: Add pasta for 8 minutes. "
        "Step 3: Drain and serve with sauce."
    ),
    "lasagna": (
        "Follow these exact steps to prepare a classic homemade beef lasagna that serves 8-10 people: "
        "Step 1: Preheat your oven to 375°F (190°C) and lightly grease a 9×13 inch baking dish. "
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

# Exp 4 also has a different lasagna prompt (with newlines) — use from stored data
EXP4_LASAGNA = (
    "Follow these exact steps to prepare a classic homemade beef lasagna that serves 8-10 people:\n\n"
    "Step 1: Preheat your oven to 375°F (190°C) and lightly grease a 9x13 inch baking dish.\n"
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

# ── System prompts (from Exp 4) ──

SYSTEM_PROMPTS = {
    1: "",
    2: "You are a helpful assistant.",
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
        "Step 1: …\nStep 2: …\nStep 3: … Nothing else."
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

# ── Exp 1 models (for lasagna keys) ──

EXP1_MODELS = {
    "qwen3.5:27b-q4_K_M": {"model": "qwen3.5:27b", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:27b-q8_0": {"model": "qwen3.5:27b-q8_0", "service_type": "ollama", "host": HOST, "port": PORT},
    "qwen3.5:27b-bf16": {"model": "qwen3.5:27b-bf16", "service_type": "ollama", "host": HOST, "port": PORT},
}

# ── Exp 4 models ──

EXP4_MODELS = {
    "qwen3.5:27b": {"model": "qwen3.5:27b", "service_type": "ollama", "host": HOST, "port": PORT},
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# Build reverse lookup: slug -> system prompt id
_SLUG_TO_ID: dict[str, int] = {}
_PROMPT_NAMES = {
    1: "Baseline - None", 2: "Minimal Helpful", 3: "Strong Anti-Meta",
    4: "Ultra-Constrained Fidelity", 5: "Expert VL Cooking Assistant",
    6: "Deterministic Relay", 7: "100% Fidelity", 8: "Numbered-Only Strict",
    9: "Natural & Engaging", 10: "Progressive Refinement",
    11: "Visual Tutorial Style", 12: "Precise Transmitter", 13: "Zero Fluff",
    14: "Locked Format", 15: "Friendly Conversational", 16: "Multi-Agent Relay",
    17: "Think-Step-by-Step", 18: "Multimodal Visual Focus", 19: "No Meta Ever",
    20: "Balanced Clarity",
}
for _id, _name in _PROMPT_NAMES.items():
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
) -> tuple[str, list[dict]]:
    print(f"\n{'='*60}")
    print(f"Condition: {condition_key}")
    print(f"{'='*60}")

    results = []
    current_message = initial_prompt

    with httpx.Client() as client:
        for i in range(1, ITERATIONS + 1):
            t0 = time.time()
            payload = {
                "message": current_message,
                "prompt_style": PROMPT_STYLE,
                "temperature": TEMPERATURE,
                "seed": seed,
                **model_info,
            }
            if system_prompt:
                payload["system_prompt"] = system_prompt
            try:
                resp = client.post(
                    f"{BATCH_API_URL}/api/run",
                    json=payload,
                    timeout=600.0,
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
    """Find 27b condition keys that failed due to timeout."""
    timeout_keys = []
    for key, iters in all_data.items():
        if "27b" not in key:
            continue
        if len(iters) >= ITERATIONS:
            continue
        if not iters:
            continue
        last = iters[-1]
        error = last.get("error", "")
        if "timed out" in error or "timed_out" in error:
            timeout_keys.append(key)
    return sorted(timeout_keys)


def resolve_exp4_condition(key: str) -> tuple[dict, int, str, str] | None:
    """Resolve Exp 4 key -> (model_info, seed, initial_prompt, system_prompt).

    Key format: "qwen3.5:27b_{recipe}_{id:02d}_{slug}_s{seed}"
    """
    # Strip model prefix
    prefix = "qwen3.5:27b_"
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]

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
    for recipe in ("spaghetti", "lasagna"):
        recipe_prefix = f"{recipe}_"
        if rest.startswith(recipe_prefix):
            slug_part = rest[len(recipe_prefix):]
            break
    else:
        return None

    # Extract prompt ID and slug: "{id:02d}_{slug}"
    id_match = re.match(r"^(\d{2})_(.+)$", slug_part)
    if not id_match:
        return None
    prompt_id = int(id_match.group(1))
    slug = id_match.group(2)

    if prompt_id not in SYSTEM_PROMPTS:
        print(f"  WARNING: prompt_id {prompt_id} not found for key {key}")
        return None

    system_prompt = SYSTEM_PROMPTS[prompt_id]
    model_info = EXP4_MODELS["qwen3.5:27b"]

    # Use the correct initial prompt for the recipe
    if recipe == "lasagna":
        initial_prompt = EXP4_LASAGNA
    else:
        initial_prompt = PROMPTS["spaghetti"]

    return model_info, seed, initial_prompt, system_prompt


def resolve_exp1_condition(key: str) -> tuple[dict, int, str, str] | None:
    """Resolve Exp 1 key -> (model_info, seed, initial_prompt, system_prompt).

    Key format: "lasagna_qwen3.5:27b-{quant}_s{seed}"
    """
    if not key.startswith("lasagna_"):
        return None
    rest = key[len("lasagna_"):]

    # Extract seed
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

    return EXP1_MODELS[model_label], seed, PROMPTS["lasagna"], ""


def process_dataset(
    data_path: Path,
    resolve_fn,
    dataset_name: str,
) -> None:
    """Process a single dataset: identify timeouts, delete, re-run, save."""
    if not data_path.exists():
        print(f"  SKIP: {data_path} not found")
        return

    existing = json.loads(data_path.read_text())
    all_data = existing.get("models", {})
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"  Loaded {len(all_data)} conditions from {data_path}")

    timeout_keys = identify_timeout_conditions(all_data)
    print(f"  Found {len(timeout_keys)} 27b timed-out conditions:")
    for k in timeout_keys:
        n = len(all_data[k])
        print(f"    {k}: had {n} iters")

    if not timeout_keys:
        print("  Nothing to re-run.")
        return

    # Delete timed-out entries
    for key in timeout_keys:
        del all_data[key]
    print(f"\n  Removed {len(timeout_keys)} incomplete entries. Running sequentially...\n")

    def save_json() -> None:
        save_data = dict(existing)
        save_data["models"] = all_data
        save_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data_path.write_text(json.dumps(save_data, indent=2))

    # Save with deletions first
    save_json()

    # Run each one sequentially
    completed = 0
    for key in timeout_keys:
        resolved = resolve_fn(key)
        if resolved is None:
            print(f"  WARNING: Could not resolve {key}, skipping")
            continue

        model_info, seed, initial_prompt, system_prompt = resolved
        cond_key, results = run_single_chain(
            key, model_info, seed, initial_prompt, system_prompt
        )

        all_data[cond_key] = results
        completed += 1
        save_json()
        n_iters = len(results)
        last_status = results[-1]["status"] if results else "empty"
        print(f"\n  -> Saved {cond_key} ({completed}/{len(timeout_keys)}): "
              f"{n_iters} iters, last={last_status}")

    print(f"\n  Done {dataset_name}. Re-ran {completed} conditions.")
    print(f"  Total conditions in file: {len(all_data)}")


def main() -> None:
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

    # Process Exp 4 sysprompt first (198 conditions, ~15h)
    process_dataset(
        data_path=Path("results/qwen35_sysprompt/drift_data.json"),
        resolve_fn=resolve_exp4_condition,
        dataset_name="Exp 4 sysprompt (27b timeouts)",
    )

    # Process Exp 1 lasagna (15 conditions, ~1h)
    process_dataset(
        data_path=Path("results/qwen35_size_quant/drift_data.json"),
        resolve_fn=resolve_exp1_condition,
        dataset_name="Exp 1 lasagna (27b timeouts)",
    )

    print(f"\n{'='*70}")
    print("All done!")


if __name__ == "__main__":
    main()
