#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Paper 3 smoke test: short chains per label x recipe, prints a timing table.

Results are NOT written into the experiment data files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ENDPOINTS, PROMPTS, run_single_chain  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=",".join(sorted(ENDPOINTS)))
    p.add_argument("--recipes", default="spaghetti,lasagna")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    recipes = [x.strip() for x in args.recipes.split(",") if x.strip()]

    rows = []
    for label in labels:
        ep = ENDPOINTS[label]
        for recipe in recipes:
            try:
                _, recs = run_single_chain(
                    ep, f"SMOKE_{label}_{recipe}", 42, 0.7, PROMPTS[recipe],
                    thinking=args.thinking, iterations=args.iterations)
            except Exception as exc:  # noqa: BLE001
                print(f"!! {label}/{recipe} failed: {exc}", flush=True)
                rows.append((label, recipe, None, None, None, f"ERROR {exc}"))
                continue
            ok = [r for r in recs if r["status"] == "success"]
            if not ok:
                rows.append((label, recipe, None, None, None, "all iterations failed"))
                continue
            s_iter = sum(r["elapsed_seconds"] for r in ok) / len(ok)
            toks = [r["completion_tokens"] for r in ok if r["completion_tokens"] is not None]
            tok_iter = sum(toks) / len(toks) if toks else None
            rc = any(r["reasoning_chars"] > 0 for r in ok)
            note = f"{len(ok)}/{args.iterations} ok"
            if any(not r["output_message"] for r in ok):
                note += ", EMPTY output seen"
            rows.append((label, recipe, s_iter, tok_iter, rc, note))

    print("\n\n" + "=" * 100)
    print("SMOKE TIMING TABLE")
    print("=" * 100)
    print(f"{'label':<26} {'recipe':<10} {'s/iter':>9} {'tok/iter':>9} {'reason':>7}  note")
    for label, recipe, s, t, rc, note in rows:
        s_s = f"{s:.1f}" if s is not None else "-"
        t_s = f"{t:.0f}" if t is not None else "-"
        rc_s = ("yes" if rc else "no") if rc is not None else "-"
        print(f"{label:<26} {recipe:<10} {s_s:>9} {t_s:>9} {rc_s:>7}  {note}")


if __name__ == "__main__":
    main()
