#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Analyze Qwen 3.5 drift experiment results and generate plots.

Reads results/drift_qwen35/runs.jsonl, produces:
  - results/drift_qwen35/plots/category_comparison.png
  - results/drift_qwen35/plots/temperature_effect.png
  - results/drift_qwen35/plots/seed_variance.png
  - results/drift_qwen35/plots/prompt_complexity.png
  - results/drift_qwen35/plots/system_prompts_ranked.png
  - results/drift_qwen35/summary_stats.md

Usage:
    uv run python scripts/analyze_drift_qwen35.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS_PATH = Path("results/drift_qwen35/runs.jsonl")
PLOTS_DIR = Path("results/drift_qwen35/plots")
SUMMARY_PATH = Path("results/drift_qwen35/summary_stats.md")

ITERATIONS = 30

CATEGORY_COLORS = {
    "baseline": "#888888",
    "anti-drift": "#2196F3",
    "constrained": "#4CAF50",
    "creative": "#FF9800",
    "vl-leveraged": "#9C27B0",
    "thinking": "#E91E63",
}

CATEGORY_ORDER = ["baseline", "anti-drift", "constrained", "creative", "vl-leveraged", "thinking"]


def load_runs() -> list[dict]:
    """Load all run records from runs.jsonl."""
    if not RUNS_PATH.exists():
        print(f"ERROR: {RUNS_PATH} not found", file=sys.stderr)
        sys.exit(1)
    runs = []
    with open(RUNS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    print(f"Loaded {len(runs)} runs from {RUNS_PATH}")
    return runs


def get_similarity_curve(run: dict) -> list[float | None]:
    """Extract the 30-step similarity curve from a run."""
    sims = [None] * ITERATIONS
    for it in run["iterations"]:
        idx = it["iteration"] - 1
        if 0 <= idx < ITERATIONS:
            sims[idx] = it["cosine_similarity"]
    return sims


def get_final_similarity(run: dict) -> float | None:
    """Get the last non-None similarity value."""
    for it in reversed(run["iterations"]):
        if it["cosine_similarity"] is not None:
            return it["cosine_similarity"]
    return None


def mean_curve(curves: list[list[float | None]]) -> tuple[list[float], list[float], list[float]]:
    """Compute mean and std of similarity curves, skipping None values."""
    means = []
    stds = []
    xs = []
    for i in range(ITERATIONS):
        vals = [c[i] for c in curves if c[i] is not None]
        if vals:
            xs.append(i + 1)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return xs, means, stds


# ─── Plot 1: Category comparison ─────────────────────────────

def plot_category_comparison(runs: list[dict], temperature: float = 0.7) -> None:
    """One line per system prompt category (mean ± std), at selected temperature."""
    by_cat: dict[str, list[list[float | None]]] = defaultdict(list)
    for r in runs:
        if r["temperature"] == temperature:
            by_cat[r["system_prompt_category"]].append(get_similarity_curve(r))

    fig, ax = plt.subplots(figsize=(12, 7))
    for cat in CATEGORY_ORDER:
        if cat not in by_cat:
            continue
        xs, means, stds = mean_curve(by_cat[cat])
        color = CATEGORY_COLORS.get(cat, "#333333")
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(xs, means, marker="o", markersize=3, linewidth=1.8, label=cat, color=color)
        ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr, alpha=0.15, color=color)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax.set_title(f"System Prompt Category vs Semantic Drift (t={temperature})", fontsize=13)
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    path = PLOTS_DIR / "category_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 2: Temperature effect ──────────────────────────────

def plot_temperature_effect(runs: list[dict]) -> None:
    """One line per temperature, averaging across all system prompts."""
    by_temp: dict[float, list[list[float | None]]] = defaultdict(list)
    for r in runs:
        by_temp[r["temperature"]].append(get_similarity_curve(r))

    fig, ax = plt.subplots(figsize=(12, 7))
    temps_sorted = sorted(by_temp.keys())
    cmap = plt.cm.coolwarm
    for idx, temp in enumerate(temps_sorted):
        xs, means, stds = mean_curve(by_temp[temp])
        color = cmap(idx / max(len(temps_sorted) - 1, 1))
        ax.plot(xs, means, marker="o", markersize=3, linewidth=1.8,
                label=f"t={temp}", color=color)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr, alpha=0.1, color=color)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax.set_title("Temperature Effect on Semantic Drift", fontsize=13)
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    path = PLOTS_DIR / "temperature_effect.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 3: Seed variance ───────────────────────────────────

def plot_seed_variance(runs: list[dict], temperature: float = 0.7) -> None:
    """Box plot of final similarity per system prompt at given temperature."""
    by_sp: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r["temperature"] == temperature:
            sim = get_final_similarity(r)
            if sim is not None:
                by_sp[r["system_prompt_key"]].append(sim)

    # Sort by median
    sp_sorted = sorted(by_sp.keys(), key=lambda k: np.median(by_sp[k]), reverse=True)
    data = [by_sp[sp] for sp in sp_sorted]

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6)

    # Color by category — derive from run data
    sp_to_cat: dict[str, str] = {}
    for r in runs:
        sp_to_cat[r["system_prompt_key"]] = r["system_prompt_category"]

    for i, sp in enumerate(sp_sorted):
        cat = sp_to_cat.get(sp, "baseline")
        color = CATEGORY_COLORS.get(cat, "#888888")
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.5)

    ax.set_xticklabels(sp_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final Cosine Similarity", fontsize=12)
    ax.set_title(f"Seed Variance per System Prompt (t={temperature}, 5 seeds)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    path = PLOTS_DIR / "seed_variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 4: Spaghetti vs Lasagna ────────────────────────────

def plot_prompt_complexity(runs: list[dict], temperature: float = 0.7) -> None:
    """Grouped bar chart: short vs long prompt per category."""
    # category → prompt_label → [final sims]
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["temperature"] == temperature:
            sim = get_final_similarity(r)
            if sim is not None:
                data[r["system_prompt_category"]][r["input_prompt_label"]].append(sim)

    cats = [c for c in CATEGORY_ORDER if c in data]
    x = np.arange(len(cats))
    width = 0.35

    spaghetti_means = [np.mean(data[c].get("spaghetti", [0])) for c in cats]
    spaghetti_stds = [np.std(data[c].get("spaghetti", [0])) for c in cats]
    lasagna_means = [np.mean(data[c].get("lasagna", [0])) for c in cats]
    lasagna_stds = [np.std(data[c].get("lasagna", [0])) for c in cats]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, spaghetti_means, width, yerr=spaghetti_stds,
           label="Spaghetti (3 steps)", color="#42A5F5", capsize=3)
    ax.bar(x + width / 2, lasagna_means, width, yerr=lasagna_stds,
           label="Lasagna (21 steps)", color="#EF5350", capsize=3)

    ax.set_xlabel("System Prompt Category", fontsize=12)
    ax.set_ylabel("Mean Final Cosine Similarity", fontsize=12)
    ax.set_title(f"Short vs Long Prompt by Category (t={temperature})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    path = PLOTS_DIR / "prompt_complexity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 5: All 20 system prompts ranked ────────────────────

def plot_system_prompts_ranked(runs: list[dict], temperature: float = 0.7) -> None:
    """Horizontal bar chart of all system prompts, color-coded by category."""
    by_sp: dict[str, list[float]] = defaultdict(list)
    sp_cat: dict[str, str] = {}
    for r in runs:
        if r["temperature"] == temperature:
            sim = get_final_similarity(r)
            if sim is not None:
                by_sp[r["system_prompt_key"]].append(sim)
                sp_cat[r["system_prompt_key"]] = r["system_prompt_category"]

    # Sort by mean descending
    sp_sorted = sorted(by_sp.keys(), key=lambda k: np.mean(by_sp[k]), reverse=True)
    means = [np.mean(by_sp[sp]) for sp in sp_sorted]
    stds = [np.std(by_sp[sp]) for sp in sp_sorted]
    colors = [CATEGORY_COLORS.get(sp_cat.get(sp, "baseline"), "#888888") for sp in sp_sorted]

    fig, ax = plt.subplots(figsize=(12, 9))
    y_pos = np.arange(len(sp_sorted))
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.7, capsize=3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sp_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Final Cosine Similarity", fontsize=12)
    ax.set_title(f"System Prompts Ranked by Drift Resistance (t={temperature})", fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS[cat], alpha=0.7, label=cat)
        for cat in CATEGORY_ORDER if cat in set(sp_cat.values())
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    path = PLOTS_DIR / "system_prompts_ranked.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Summary stats table ─────────────────────────────────────

def write_summary_stats(runs: list[dict]) -> None:
    """Write summary statistics as a markdown table."""
    # Group by (system_prompt_key, temperature) → [final sims]
    by_sp_temp: dict[tuple[str, float], list[float]] = defaultdict(list)
    sp_cat: dict[str, str] = {}
    for r in runs:
        sim = get_final_similarity(r)
        if sim is not None:
            by_sp_temp[(r["system_prompt_key"], r["temperature"])].append(sim)
            sp_cat[r["system_prompt_key"]] = r["system_prompt_category"]

    # Also group by (category, temperature)
    by_cat_temp: dict[tuple[str, float], list[float]] = defaultdict(list)
    for (sp, temp), sims in by_sp_temp.items():
        by_cat_temp[(sp_cat[sp], temp)].extend(sims)

    lines = ["# Qwen 3.5 Drift Experiment — Summary Statistics\n"]

    # Per-category summary at each temperature
    lines.append("## Category Averages\n")
    temps = sorted(set(t for _, t in by_cat_temp))
    header = "| Category | " + " | ".join(f"t={t}" for t in temps) + " |"
    sep = "|---|" + "|".join("---" for _ in temps) + "|"
    lines.append(header)
    lines.append(sep)
    for cat in CATEGORY_ORDER:
        row = f"| {cat} |"
        for t in temps:
            sims = by_cat_temp.get((cat, t), [])
            if sims:
                row += f" {np.mean(sims):.3f} ± {np.std(sims):.3f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")

    # Per system-prompt at t=0.7
    lines.append("## Per System Prompt (t=0.7)\n")
    lines.append("| System Prompt | Category | Mean | Std | Min | Max | N |")
    lines.append("|---|---|---|---|---|---|---|")
    sp_at_07 = {sp: sims for (sp, t), sims in by_sp_temp.items() if t == 0.7}
    for sp in sorted(sp_at_07, key=lambda k: np.mean(sp_at_07[k]), reverse=True):
        sims = sp_at_07[sp]
        cat = sp_cat.get(sp, "?")
        lines.append(
            f"| {sp} | {cat} | {np.mean(sims):.3f} | {np.std(sims):.3f} "
            f"| {min(sims):.3f} | {max(sims):.3f} | {len(sims)} |"
        )

    lines.append("")

    # Overall stats
    all_finals = [get_final_similarity(r) for r in runs]
    all_finals = [s for s in all_finals if s is not None]
    if all_finals:
        lines.append("## Overall\n")
        lines.append(f"- Total runs: {len(runs)}")
        lines.append(f"- Runs with final similarity: {len(all_finals)}")
        lines.append(f"- Global mean final similarity: {np.mean(all_finals):.3f}")
        lines.append(f"- Global std: {np.std(all_finals):.3f}")
        lines.append(f"- Min: {min(all_finals):.3f}, Max: {max(all_finals):.3f}")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {SUMMARY_PATH}")


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    runs = load_runs()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_category_comparison(runs)
    plot_temperature_effect(runs)
    plot_seed_variance(runs)
    plot_prompt_complexity(runs)
    plot_system_prompts_ranked(runs)

    print("\nWriting summary stats...")
    write_summary_stats(runs)

    print("\nDone!")


if __name__ == "__main__":
    main()
