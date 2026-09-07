#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Paper 3 shared machinery: endpoints, chain runner, embedding, save/resume.

Calls the inference backends *directly* with the `openai` client (never via the
LiteLLM aliases, which are ambiguous). Chain protocol is identical to Papers 1-2:
30 iterations of "Paraphrase this to the next agent: {message}", cosine
similarity of each output against the original prompt.

Data files use Paper 2's schema so the old analysis code still loads them:
    {"models": {condition_key: [iteration records...]}, "timestamp": ...}
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI

# Import prompts / system prompts from the Paper 2 script rather than retyping.
# (That module guards all of its work behind `if __name__ == "__main__"`, so
# importing it is side-effect free.)
_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "legacy"))  # Paper 1/2 run scripts (prompt definitions)
from run_qwen35_nvfp4_sysprompt import (  # noqa: E402
    PROMPTS,
    PROMPT_NAMES,
    SYSTEM_PROMPTS,
    _slug,
)

__all__ = [
    "PROMPTS", "PROMPT_NAMES", "SYSTEM_PROMPTS", "_slug",
    "ENDPOINTS", "Endpoint", "SEEDS", "TEMPERATURES", "ITERATIONS",
    "USER_TEMPLATE", "run_single_chain", "run_conditions",
    "data_path", "log_dir", "is_complete", "load_or_init",
]

# ---------------------------------------------------------------- constants

ITERATIONS = 30
SEEDS = [42, 137, 256, 512, 1024]
TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]
MAX_TOKENS = 16384
REQUEST_TIMEOUT = 600.0
RETRIES = 1

# PROMPT_STYLES["paraphrase"] from src/telephone/game_logic.py
USER_TEMPLATE = "Paraphrase this to the next agent: {message}"

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "paper3"

EMBED_HOST = os.environ.get("EMBEDDING_HOST", "<GPU-HOST-D>")
EMBED_PORT = os.environ.get("EMBEDDING_PORT", "8002")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_FALLBACK_HOST = "<EMBED-HOST>"


@dataclass(frozen=True)
class Endpoint:
    label: str
    host: str
    port: int
    model: str
    backend: str
    max_workers: int


ENDPOINTS: dict[str, Endpoint] = {
    "qwen3.6-35b-nvfp4": Endpoint(
        "qwen3.6-35b-nvfp4", "<GPU-HOST-C>", 8005, "qwen3.6-moe-nvfp4", "vllm", 8),
    "qwen3.6-35b-nvfp4-gpu-host-d": Endpoint(
        "qwen3.6-35b-nvfp4-gpu-host-d", "<GPU-HOST-D>", 8005, "qwen3.6-moe-nvfp4", "vllm", 8),
    "qwen3.8-27b-nvfp4": Endpoint(
        "qwen3.8-27b-nvfp4", "<GPU-HOST-A>", 8005, "qwen3.8-27b-nvfp4", "vllm", 4),
    "qwen3.8-flash-next-nvfp4": Endpoint(
        "qwen3.8-flash-next-nvfp4", "<GPU-HOST-D>", 8006, "qwen3.8-flash-next", "sglang", 4),
    "qwen3.8-27b-q4kxl": Endpoint(
        "qwen3.8-27b-q4kxl", "<GPU-HOST-B>", 8080, "qwen3.8-27b", "llamacpp", 1),
    "qwen3.8-27b-q4kxl-mtp": Endpoint(
        "qwen3.8-27b-q4kxl-mtp", "<GPU-HOST-B>", 8080, "qwen3.8-27b-mtp", "llamacpp", 1),
    "qwen3.6-35b-q4kxl": Endpoint(
        "qwen3.6-35b-q4kxl", "<GPU-HOST-B>", 8080, "qwen3.6-35b", "llamacpp", 1),
    "qwen3.6-27b-q4kxl": Endpoint(
        "qwen3.6-27b-q4kxl", "<GPU-HOST-B>", 8080, "qwen3.6-27b", "llamacpp", 1),
}

# Hard guard: only these served ids may ever be addressed on the gpu-host-b box.
GPU_HOST_B_HOST = "<GPU-HOST-B>"
GPU_HOST_B_ALLOWED_MODELS = {
    "qwen3.8-27b", "qwen3.8-27b-mtp", "qwen3.6-35b", "qwen3.6-27b",
}
GPU_HOST_B_MAX_WORKERS = 1

_save_lock = threading.Lock()
_client_lock = threading.Lock()
_clients: dict[str, OpenAI] = {}


def log_dir(label: str) -> Path:
    d = RESULTS_ROOT / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_path(label: str, exp: str) -> Path:
    return log_dir(label) / f"{exp}.json"


# ---------------------------------------------------------------- clients

def get_client(ep: Endpoint) -> OpenAI:
    if ep.host == GPU_HOST_B_HOST and ep.model not in GPU_HOST_B_ALLOWED_MODELS:
        raise RuntimeError(
            f"refusing to send chat requests to gpu-host-b model id {ep.model!r}; "
            f"allowed: {sorted(GPU_HOST_B_ALLOWED_MODELS)}")
    key = f"{ep.host}:{ep.port}"
    with _client_lock:
        if key not in _clients:
            _clients[key] = OpenAI(
                base_url=f"http://{ep.host}:{ep.port}/v1",
                api_key="EMPTY",
                timeout=REQUEST_TIMEOUT,
                max_retries=0,
            )
        return _clients[key]


# ---------------------------------------------------------------- embedding

def _embed(client: httpx.Client, host: str, texts: list[str]) -> list[list[float]]:
    resp = client.post(
        f"http://{host}:{EMBED_PORT}/v1/embeddings",
        json={"input": texts, "model": EMBED_MODEL},
        timeout=60.0,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def get_similarity(client: httpx.Client, text_a: str, text_b: str) -> float | None:
    for host in (EMBED_HOST, EMBED_FALLBACK_HOST):
        try:
            va, vb = _embed(client, host, [text_a, text_b])
            return _cosine(va, vb)
        except Exception as exc:  # noqa: BLE001
            last = exc
    print(f"  [embedding error: {last}]", flush=True)
    return None


# ---------------------------------------------------------------- chain loop

def _call_model(
    ep: Endpoint,
    message: str,
    seed: int,
    temperature: float,
    system_prompt: str,
    thinking: bool,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(message=message)})

    kwargs: dict = {
        "model": ep.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
    }
    if thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    return get_client(ep).chat.completions.create(**kwargs)


def run_single_chain(
    ep: Endpoint,
    condition_key: str,
    seed: int,
    temperature: float,
    initial_prompt: str,
    system_prompt: str = "",
    thinking: bool = False,
    iterations: int = ITERATIONS,
) -> tuple[str, list[dict]]:
    print(f"\n=== {condition_key}  (seed={seed} T={temperature} think={thinking})", flush=True)

    records: list[dict] = []
    current_message = initial_prompt

    with httpx.Client() as hclient:
        for i in range(1, iterations + 1):
            t0 = time.time()
            resp = None
            err = None
            for attempt in range(RETRIES + 1):
                try:
                    resp = _call_model(
                        ep, current_message, seed, temperature, system_prompt, thinking)
                    err = None
                    break
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)
                    if attempt < RETRIES:
                        print(f"  [{condition_key}] iter {i}: error, retry in 60s: {err}",
                              flush=True)
                        time.sleep(60)

            elapsed = time.time() - t0

            if resp is None:
                print(f"  [{condition_key}] iter {i:2d}/{iterations}: FAIL ({err})", flush=True)
                records.append({
                    "iteration": i,
                    "status": "error",
                    "error": err,
                    "input_message": current_message,
                    "output_message": "",
                    "cosine_similarity": None,
                    "elapsed_seconds": round(elapsed, 2),
                })
                break

            choice = resp.choices[0]
            output = choice.message.content or ""
            # vLLM returns `reasoning`, sglang returns `reasoning_content`
            reasoning = (getattr(choice.message, "reasoning_content", None)
                         or getattr(choice.message, "reasoning", None) or "")
            usage = resp.usage
            _details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(_details, "reasoning_tokens", None) if _details else None
            sim = get_similarity(hclient, initial_prompt, output)

            sim_str = f"{sim:.4f}" if sim is not None else "N/A"
            tag = " [EMPTY]" if not output else ""
            print(f"  [{condition_key}] iter {i:2d}/{iterations}: sim={sim_str}{tag} "
                  f"({elapsed:.1f}s, {getattr(usage, 'completion_tokens', None)} tok, "
                  f"rc={len(reasoning)})", flush=True)

            records.append({
                "iteration": i,
                "status": "success",
                "input_message": current_message,
                "output_message": output,
                "cosine_similarity": sim,
                "elapsed_seconds": round(elapsed, 2),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_chars": len(reasoning),
                "reasoning_tokens": reasoning_tokens,
                "finish_reason": choice.finish_reason,
                "output_words": len(output.split()),
            })

            # Paper 2 rule: on empty output keep the previous message.
            if output:
                current_message = output

    return condition_key, records


# ---------------------------------------------------------------- persistence

def load_or_init(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  WARNING: {path} unreadable, starting fresh", flush=True)
    return {"models": {}}


def save_json(path: Path, existing: dict, all_data: dict) -> None:
    with _save_lock:
        out = dict(existing)
        out["models"] = all_data
        out["timestamp"] = datetime.now(timezone.utc).isoformat()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(path)


def is_complete(iters: list[dict], iterations: int = ITERATIONS) -> bool:
    return len(iters) >= iterations and iters[-1].get("status") == "success"


def run_conditions(
    ep: Endpoint,
    path: Path,
    conditions: list[dict],
    workers: int,
    iterations: int = ITERATIONS,
) -> None:
    """conditions: list of dicts with keys key, seed, temperature, prompt,
    system_prompt, thinking."""
    if ep.host == GPU_HOST_B_HOST and workers > GPU_HOST_B_MAX_WORKERS:
        print(f"Clamping workers {workers} -> {GPU_HOST_B_MAX_WORKERS} (gpu-host-b guard)", flush=True)
        workers = GPU_HOST_B_MAX_WORKERS

    existing = load_or_init(path)
    all_data = existing.get("models", {})
    print(f"Loaded {len(all_data)} existing conditions from {path}", flush=True)

    dropped = [k for k, v in all_data.items() if not is_complete(v, iterations)]
    for k in dropped:
        del all_data[k]
    if dropped:
        print(f"Dropped {len(dropped)} incomplete conditions for re-run", flush=True)
        save_json(path, existing, all_data)

    remaining = [c for c in conditions if c["key"] not in all_data]
    print(f"Total {len(conditions)}, remaining {len(remaining)}, workers {workers}", flush=True)
    if not remaining:
        print("Nothing to do.", flush=True)
        return

    done = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_single_chain, ep, c["key"], c["seed"], c["temperature"],
                c["prompt"], c.get("system_prompt", ""), c.get("thinking", False),
                iterations,
            ): c["key"]
            for c in remaining
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                cond_key, records = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  !! chain {key} raised: {exc}", flush=True)
                continue
            with _save_lock:
                all_data[cond_key] = records
                done += 1
            save_json(path, existing, all_data)
            mins = (time.time() - t_start) / 60
            print(f"-> saved {cond_key} ({done}/{len(remaining)}): {len(records)} iters, "
                  f"last={records[-1]['status'] if records else 'empty'} "
                  f"[{mins:.1f} min elapsed]", flush=True)

    print(f"Done: {done}/{len(remaining)} conditions in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
