#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Master test script — defines a test matrix, calls the batch API, saves results as JSON."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import httpx

# ─── Configuration (env vars with defaults) ─────────────────

BATCH_API_URL = os.environ.get("BATCH_API_URL", "http://localhost:5001").rstrip("/")
APP_API_KEY = os.environ.get("APP_API_KEY", "")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))

# ─── Test Matrix (easily customizable) ──────────────────────

PROMPTS = {
    "fox": "The quick brown fox jumps over the lazy dog",
    "hamlet": "To be or not to be, that is the question",
    "science": "Water is composed of two hydrogen atoms and one oxygen atom",
}

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]

PROMPT_STYLES = ["repeat", "paraphrase", "exaggerate", "improv"]

# Empty list = use all discovered models; otherwise filter to these names
MODEL_FILTER: list[str] = []

# ─── Helpers ─────────────────────────────────────────────────


def api_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if APP_API_KEY:
        headers["Authorization"] = f"Bearer {APP_API_KEY}"
    return headers


def discover_models(client: httpx.Client) -> list[dict]:
    """Fetch available services/models from the batch API."""
    resp = client.get(f"{BATCH_API_URL}/api/services", headers=api_headers())
    resp.raise_for_status()
    data = resp.json()
    services = data.get("services", [])

    models = []
    for svc in services:
        for model_name in svc.get("models", []):
            if not MODEL_FILTER or model_name in MODEL_FILTER:
                models.append({
                    "service_type": svc["service_type"],
                    "host": svc["host"],
                    "port": svc["port"],
                    "model": model_name,
                })
    return models


def sanitize_filename(name: str) -> str:
    """Make a string safe for use in filenames."""
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def run_single_test(
    client: httpx.Client,
    test_id: int,
    prompt_label: str,
    prompt_text: str,
    model_info: dict,
    temperature: float,
    prompt_style: str,
) -> dict:
    """Run a single test via the batch API and return the result dict."""
    request_body = {
        "message": prompt_text,
        "model": model_info["model"],
        "service_type": model_info["service_type"],
        "host": model_info["host"],
        "port": model_info["port"],
        "prompt_style": prompt_style,
        "temperature": temperature,
    }

    try:
        resp = client.post(
            f"{BATCH_API_URL}/api/run",
            headers=api_headers(),
            json=request_body,
            timeout=300.0,
        )
        api_result = resp.json()
    except Exception as exc:
        api_result = {"status": "error", "error": str(exc)}

    return {
        "test_id": test_id,
        "prompt_label": prompt_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": request_body,
        "response": api_result,
    }


# ─── Main ────────────────────────────────────────────────────


def main() -> None:
    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S')}"
    batch_dir = RESULTS_DIR / batch_id
    individual_dir = batch_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch: {batch_id}")
    print(f"API:   {BATCH_API_URL}")
    print(f"Dir:   {batch_dir}")
    print()

    with httpx.Client() as client:
        # Discover models
        print("Discovering models...")
        models = discover_models(client)
        if not models:
            print("No models found. Is the batch API running?", file=sys.stderr)
            sys.exit(1)

        model_names = sorted({m["model"] for m in models})
        print(f"Found {len(models)} model endpoints: {', '.join(model_names)}")

        # Build Cartesian product
        combinations = list(product(
            PROMPTS.items(),
            models,
            TEMPERATURES,
            PROMPT_STYLES,
        ))
        total = len(combinations)
        print(f"Total tests: {total}")
        print()

        succeeded = 0
        failed = 0
        failures = []

        for i, ((prompt_label, prompt_text), model_info, temp, style) in enumerate(combinations, 1):
            model_safe = sanitize_filename(model_info["model"])
            filename = f"{i:04d}_{model_safe}_{style}_t{temp}_{prompt_label}.json"

            print(f"[{i}/{total}] {model_info['model']} | {style} | t={temp} | {prompt_label}...", end=" ", flush=True)

            result = run_single_test(
                client, i, prompt_label, prompt_text, model_info, temp, style,
            )

            # Save individual result
            result_path = individual_dir / filename
            result_path.write_text(json.dumps(result, indent=2))

            status = result["response"].get("status", "error")
            if status == "success":
                succeeded += 1
                elapsed = result["response"].get("elapsed_seconds", "?")
                similarity = result["response"].get("cosine_similarity")
                sim_str = f"sim={similarity:.3f}" if similarity is not None else "sim=N/A"
                print(f"OK ({elapsed}s, {sim_str})")
            else:
                failed += 1
                error = result["response"].get("error", "unknown")
                failures.append({"test_id": i, "filename": filename, "error": error})
                print(f"FAIL: {error}")

        # Write summary
        summary = {
            "batch_id": batch_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "dimensions": {
                "prompts": list(PROMPTS.keys()),
                "models": model_names,
                "temperatures": TEMPERATURES,
                "prompt_styles": PROMPT_STYLES,
            },
            "failures": failures,
        }
        summary_path = batch_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print()
        print(f"Done! {succeeded}/{total} succeeded, {failed} failed")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
