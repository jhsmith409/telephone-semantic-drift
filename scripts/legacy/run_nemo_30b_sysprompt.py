#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Nemotron-3-Nano-30B NVFP4 system prompt experiment on sglang.

Runs the same 20 system prompts x 2 recipes x 5 seeds as the other quant
variants, enabling cross-quantization comparison on prompt sensitivity.

20 prompts x 2 recipes x 5 seeds x 30 iters = 6,000 API calls

vLLM endpoint: <GPU-HOST-C>:8005 (OpenAI-compatible, separate GPU).

Usage:
    nohup uv run python -u scripts/run_nemo_30b_sysprompt.py \
        > results/qwen35_nemo/run_sysprompt.log 2>&1 &
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
SEEDS = [42, 137, 256, 512, 1024]
MAX_WORKERS = 1

_save_lock = threading.Lock()

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
    "model": "model",
    "service_type": "openai",
    "host": "<GPU-HOST-C>",
    "port": 8005,
}

# Same 20 system prompts as Exp 3c and Exp 4
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

DATA_PATH = Path("results/qwen35_nemo/sysprompt_drift_data.json")


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
    seed: int,
    initial_prompt: str,
    system_prompt: str = "",
) -> tuple[str, list[dict]]:
    print(f"\n{'='*60}")
    print(f"Condition: {condition_key}")
    print(f"  seed={seed}")
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
                "temperature": 0.7,
                "seed": seed,
                **MODEL_INFO,
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


def main() -> None:
    Path("results/qwen35_nemo").mkdir(parents=True, exist_ok=True)

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

    # Load existing data (for resumability)
    if DATA_PATH.exists():
        existing = json.loads(DATA_PATH.read_text())
    else:
        existing = {
            "initial_prompts": PROMPTS,
            "models": {},
        }
    all_data = existing.get("models", {})
    print(f"Loaded {len(all_data)} existing conditions from {DATA_PATH}")

    # Delete incomplete conditions so they get re-run
    incomplete = [k for k, v in all_data.items()
                  if len(v) < ITERATIONS or v[-1].get("status") != "success"]
    for k in incomplete:
        del all_data[k]
        print(f"  Deleted incomplete: {k}")

    # Build conditions: 20 prompts x 2 recipes x 5 seeds = 200
    conditions = []
    for recipe in ("spaghetti", "lasagna"):
        for prompt_id, system_prompt in SYSTEM_PROMPTS.items():
            slug = _slug(PROMPT_NAMES[prompt_id])
            for seed in SEEDS:
                key = f"{recipe}_{prompt_id:02d}_{slug}_s{seed}"
                conditions.append((key, seed, PROMPTS[recipe], system_prompt))

    remaining = [(k, s, p, sp) for k, s, p, sp in conditions if k not in all_data]
    print(f"Total: {len(conditions)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All Nemotron sysprompt conditions already complete.")
        return

    def save_json() -> None:
        with _save_lock:
            save = dict(existing)
            save["models"] = all_data
            save["timestamp"] = datetime.now(timezone.utc).isoformat()
            DATA_PATH.write_text(json.dumps(save, indent=2))

    # Save after deleting incomplete entries
    if incomplete:
        save_json()

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_single_chain, key, seed, prompt, sysprompt): key
            for key, seed, prompt, sysprompt in remaining
        }
        for future in as_completed(futures):
            cond_key, results = future.result()
            with _save_lock:
                all_data[cond_key] = results
                completed += 1
            save_json()
            n_iters = len(results)
            last_status = results[-1]["status"] if results else "empty"
            print(f"\n  -> Saved {cond_key} ({completed}/{len(remaining)}): "
                  f"{n_iters} iters, last={last_status}")

    print(f"\n{'='*70}")
    print(f"Nemotron sysprompt experiment complete! {completed} conditions saved.")


if __name__ == "__main__":
    main()
