#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Paper 3 experiment runner.

Usage:
    uv run python -u scripts/paper3/run.py --label qwen3.6-35b-nvfp4 \
        --exp baseline --recipes spaghetti,lasagna --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ENDPOINTS, ITERATIONS, PROMPTS, PROMPT_NAMES, SEEDS, SYSTEM_PROMPTS,
    TEMPERATURES, _slug, data_path, is_complete, load_or_init, run_conditions,
)
from concrete_prompts import CONCRETE_PROMPTS  # noqa: E402
from domain_prompts import DOMAIN_ORDER, DOMAIN_PROMPTS  # noqa: E402

# Recipe dictionaries by knowledge domain. Condition keys embed the recipe name,
# so keys stay unique across domains even inside one data file.
DOMAINS = {
    "recipe": PROMPTS,
    "concrete": CONCRETE_PROMPTS,
    "breadth": DOMAIN_PROMPTS,
}
DEFAULT_RECIPES = {
    "recipe": "spaghetti,lasagna",
    "concrete": "concrete_short,concrete_long",
    "breadth": ",".join(DOMAIN_ORDER),
}


def build_conditions(args) -> list[dict]:
    label = args.label
    conds: list[dict] = []

    prompts = DOMAINS[args.domain]

    for recipe in args.recipes:
        prompt = prompts[recipe]

        if args.exp == "baseline":
            for seed in args.seeds:
                conds.append({
                    "key": f"{label}_{recipe}_s{seed}",
                    "seed": seed, "temperature": 0.7, "prompt": prompt,
                    "system_prompt": "", "thinking": False,
                })

        elif args.exp == "thinking":
            for seed in args.seeds:
                conds.append({
                    "key": f"{label}_{recipe}_think_s{seed}",
                    "seed": seed, "temperature": 0.7, "prompt": prompt,
                    "system_prompt": "", "thinking": True,
                })

        elif args.exp == "temperature":
            for temp in args.temps:
                # T=0.7 duplicates baseline -> reuse those keys, do not rerun.
                if abs(temp - 0.7) < 1e-9:
                    continue
                seeds = [args.seeds[0]] if (temp == 0.0 and not args.t0_all_seeds) else args.seeds
                for seed in seeds:
                    conds.append({
                        "key": f"{label}_{recipe}_t{temp:.1f}_s{seed}",
                        "seed": seed, "temperature": temp, "prompt": prompt,
                        "system_prompt": "", "thinking": False,
                    })

        elif args.exp == "sysprompt":
            think_sfx = "_think" if args.thinking else ""
            for pid in args.prompt_ids:
                slug = _slug(PROMPT_NAMES[pid])
                for seed in args.seeds:
                    conds.append({
                        "key": f"{label}_{recipe}_{pid:02d}_{slug}{think_sfx}_s{seed}",
                        "seed": seed, "temperature": 0.7, "prompt": prompt,
                        "system_prompt": SYSTEM_PROMPTS[pid],
                        "thinking": args.thinking,
                    })

    return conds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, choices=sorted(ENDPOINTS))
    p.add_argument("--exp", required=True,
                   choices=["baseline", "thinking", "temperature", "sysprompt"])
    p.add_argument("--domain", default="recipe", choices=sorted(DOMAINS),
                   help="knowledge domain supplying the recipe dictionary")
    p.add_argument("--recipes", default=None,
                   help="default: both recipes of the selected --domain")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--temps", default=",".join(f"{t}" for t in TEMPERATURES))
    p.add_argument("--prompt-ids", default=",".join(str(i) for i in sorted(SYSTEM_PROMPTS)))
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--t0-all-seeds", action="store_true",
                   help="run every seed at T=0 (default: one seed, Papers 1-2 convention)")
    p.add_argument("--out-name", default=None,
                   help="override output file stem (default: the --exp name, "
                        "suffixed with the domain for non-recipe domains)")
    p.add_argument("--thinking", action="store_true",
                   help="thinking ON; only meaningful with --exp sysprompt "
                        "(--exp thinking already implies it). Keys get a _think suffix.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.recipes is None:
        args.recipes = DEFAULT_RECIPES[args.domain]
    args.recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    unknown = [r for r in args.recipes if r not in DOMAINS[args.domain]]
    if unknown:
        p.error(f"unknown recipe(s) {unknown} for --domain {args.domain}; "
                f"available: {sorted(DOMAINS[args.domain])}")
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.temps = [float(t) for t in args.temps.split(",") if t.strip()]
    args.prompt_ids = [int(i) for i in args.prompt_ids.split(",") if i.strip()]

    if args.thinking and args.exp != "sysprompt":
        p.error("--thinking is only supported with --exp sysprompt "
                "(use --exp thinking for the plain thinking experiment)")

    ep = ENDPOINTS[args.label]
    workers = args.workers or ep.max_workers
    if workers > ep.max_workers:
        print(f"Clamping workers {workers} -> {ep.max_workers} (endpoint limit)", flush=True)
        workers = ep.max_workers

    conds = build_conditions(args)
    default_stem = args.exp if args.domain == "recipe" else f"{args.exp}_{args.domain}"
    path = data_path(args.label, args.out_name or default_stem)

    print(f"label={args.label} exp={args.exp} domain={args.domain} backend={ep.backend} "
          f"{ep.host}:{ep.port} model={ep.model}", flush=True)
    print(f"conditions={len(conds)} iterations={args.iterations} workers={workers}", flush=True)
    print(f"data={path}", flush=True)

    if args.dry_run:
        # Read-only preview: never writes, never deletes.
        existing = load_or_init(path).get("models", {})
        complete = {k for k, v in existing.items() if is_complete(v, args.iterations)}
        incomplete = set(existing) - complete
        missing = [c for c in conds if c["key"] not in complete]
        print(f"file has {len(existing)} keys: {len(complete)} complete, "
              f"{len(incomplete)} incomplete", flush=True)
        if incomplete:
            print("  incomplete (would be dropped + re-run if in the request):", flush=True)
            for k in sorted(incomplete):
                print(f"    {k}  ({len(existing[k])} iters)")
        print(f"requested {len(conds)}, ALREADY DONE "
              f"{len(conds) - len(missing)}, WOULD RUN {len(missing)}", flush=True)
        for c in missing:
            print(f"  + {c['key']}  T={c['temperature']} think={c['thinking']} "
                  f"sys={len(c['system_prompt'])}ch")
        return

    run_conditions(ep, path, conds, workers, args.iterations)


if __name__ == "__main__":
    main()
