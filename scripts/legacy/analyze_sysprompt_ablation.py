#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Analyze system prompt ablation experiments across 3 datasets and 2 models.

Loads:
  1. 30b spaghetti (1x/2x emphasis) — results/drift_experiment_sysprompt/drift_data.json
  2. 30b lasagna — results/drift_experiment_sysprompt_lasagna/drift_data.json
  3. 8b spaghetti + lasagna — results/drift_experiment_sysprompt_qwen3_8b/drift_data.json

Generates 5 plots (150 DPI → results/, blog-sized → j8web/images/):
  1. Hero line plot — 30b lasagna, 20 prompts, mean ± min/max band
  2. Complexity threshold bar chart — 30b spaghetti 1x vs lasagna
  3. Cross-model slope chart — 30b vs 8b lasagna rankings
  4. 8b spaghetti seed variance — most variable prompts, individual seeds
  5. Category heatmap — 6 categories × 4 conditions

Usage:
    uv run python scripts/analyze_sysprompt_ablation.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr

# --- Paths ---
SPAG30B_DATA = Path("results/drift_experiment_sysprompt/drift_data.json")
LAS30B_DATA = Path("results/drift_experiment_sysprompt_lasagna/drift_data.json")
DATA_8B = Path("results/drift_experiment_sysprompt_qwen3_8b/drift_data.json")
OUT_DIR = Path("results/sysprompt_ablation_analysis")
BLOG_IMG_DIR = Path("../j8web/images/llm-telephone-drift-sysprompt")

ITERATIONS = 30
SEEDS = [42, 137, 256, 512, 1024]

# --- 20 system prompts (same across all experiments) ---
PROMPTS = [
    {"id": 1,  "name": "Baseline - None",              "category": "Baseline"},
    {"id": 2,  "name": "Minimal Helpful",               "category": "Baseline"},
    {"id": 3,  "name": "Strong Anti-Meta",              "category": "Anti-Drift"},
    {"id": 4,  "name": "Ultra-Constrained Fidelity",    "category": "Anti-Drift"},
    {"id": 5,  "name": "Expert VL Cooking Assistant",   "category": "VL-Leveraged"},
    {"id": 6,  "name": "Deterministic Relay",           "category": "Constrained"},
    {"id": 7,  "name": "100% Fidelity",                 "category": "Anti-Drift"},
    {"id": 8,  "name": "Numbered-Only Strict",          "category": "Constrained"},
    {"id": 9,  "name": "Natural & Engaging",            "category": "Creative"},
    {"id": 10, "name": "Progressive Refinement",        "category": "Creative"},
    {"id": 11, "name": "Visual Tutorial Style",         "category": "VL-Leveraged"},
    {"id": 12, "name": "Precise Transmitter",           "category": "Anti-Drift"},
    {"id": 13, "name": "Zero Fluff",                    "category": "Constrained"},
    {"id": 14, "name": "Locked Format",                 "category": "Constrained"},
    {"id": 15, "name": "Friendly Conversational",       "category": "Creative"},
    {"id": 16, "name": "Multi-Agent Relay",             "category": "Anti-Drift"},
    {"id": 17, "name": "Think-Step-by-Step",            "category": "Thinking"},
    {"id": 18, "name": "Multimodal Visual Focus",       "category": "VL-Leveraged"},
    {"id": 19, "name": "No Meta Ever",                  "category": "Anti-Drift"},
    {"id": 20, "name": "Balanced Clarity",              "category": "Creative"},
]

CATEGORY_COLORS = {
    "Baseline":     "#888888",
    "Anti-Drift":   "#4488ff",
    "Constrained":  "#44bb44",
    "Creative":     "#ff8844",
    "VL-Leveraged": "#aa44ff",
    "Thinking":     "#ff4444",
}

CATEGORY_ORDER = ["Baseline", "Anti-Drift", "Constrained", "Creative", "VL-Leveraged", "Thinking"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_data() -> tuple[dict, dict, dict]:
    """Load all 3 datasets. Returns (spag30b_models, las30b_models, data8b_models)."""
    spag30b = json.loads(SPAG30B_DATA.read_text())
    las30b = json.loads(LAS30B_DATA.read_text())
    d8b = json.loads(DATA_8B.read_text())
    return spag30b["models"], las30b["models"], d8b["models"]


def get_seed_curves(models: dict, key_pattern: str) -> list[list[float]]:
    """Get per-seed similarity curves matching key_pattern (with {seed} placeholder)."""
    curves = []
    for seed in SEEDS:
        key = key_pattern.format(seed=seed)
        if key not in models:
            continue
        iters = models[key]
        sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
        if sims:
            curves.append(sims)
    return curves


def get_finals(models: dict, key_pattern: str) -> list[float]:
    """Get final cosine similarity for each seed matching key_pattern."""
    finals = []
    for curve in get_seed_curves(models, key_pattern):
        finals.append(curve[-1])
    return finals


def prompt_stats(models: dict, key_fn) -> list[dict]:
    """Compute per-prompt stats. key_fn(prompt) -> key_pattern with {seed}."""
    stats = []
    for p in PROMPTS:
        pattern = key_fn(p)
        finals = get_finals(models, pattern)
        mean = float(np.mean(finals)) if finals else 0.0
        std = float(np.std(finals)) if finals else 0.0
        stats.append({
            "name": p["name"],
            "id": p["id"],
            "category": p["category"],
            "mean": mean,
            "std": std,
            "n_seeds": len(finals),
            "finals": finals,
        })
    return stats


def _save_plot(fig: plt.Figure, name: str, *, figsize: tuple | None = None) -> None:
    """Save to results/ at 150 DPI and blog/ at 100 DPI."""
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  {name}: {path}")

    blog_path = BLOG_IMG_DIR / name
    fig.savefig(blog_path, dpi=100, bbox_inches="tight")
    print(f"  {name} (blog): {blog_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Hero line plot — 30b lasagna, 20 prompts, mean ± band
# ──────────────────────────────────────────────────────────────────────
def plot_hero_lasagna(las30b: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))

    plot_items = []  # (final_mean, name, category, mean_curve, min_curve, max_curve)

    for p in PROMPTS:
        slug = _slug(p["name"])
        pattern = f"{p['id']:02d}_{slug}_s{{seed}}"
        curves = get_seed_curves(las30b, pattern)
        if not curves:
            continue

        min_len = min(len(c) for c in curves)
        aligned = np.array([c[:min_len] for c in curves])
        xs = np.arange(1, min_len + 1)
        mean_curve = aligned.mean(axis=0)
        min_curve = aligned.min(axis=0)
        max_curve = aligned.max(axis=0)
        final_mean = float(mean_curve[-1])

        plot_items.append((final_mean, p["name"], p["category"], xs, mean_curve, min_curve, max_curve))

    # Sort by final mean descending for legend
    plot_items.sort(key=lambda x: x[0], reverse=True)

    for final_mean, name, category, xs, mean_curve, min_curve, max_curve in plot_items:
        color = CATEGORY_COLORS.get(category, "#888888")
        ax.plot(xs, mean_curve, color=color, linewidth=1.5,
                label=f"{name} ({final_mean:.3f})")
        ax.fill_between(xs, min_curve, max_curve, color=color, alpha=0.12)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax.set_title(
        "System Prompt Ablation — 30B AWQ, Lasagna Recipe (21 steps)\n"
        "20 Prompts, mean across 5 seeds, shaded = min/max",
        fontsize=13,
    )
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    # Category legend patches
    cat_handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    cat_legend = ax.legend(handles=cat_handles, fontsize=8, loc="upper right", title="Category")
    ax.add_artist(cat_legend)

    # Sorted prompt legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6.5, loc="lower left", ncol=2,
              title="System Prompt (final mean sim)")

    _save_plot(fig, "hero_lasagna_30b.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Complexity threshold — 30b spaghetti 1x vs lasagna grouped bars
# ──────────────────────────────────────────────────────────────────────
def plot_complexity_threshold(spag30b: dict, las30b: dict) -> None:
    spag_stats = prompt_stats(
        spag30b, lambda p: f"{p['id']:02d}_{_slug(p['name'])}_1x_s{{seed}}"
    )
    las_stats = prompt_stats(
        las30b, lambda p: f"{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
    )

    # Sort by lasagna mean descending
    las_order = sorted(range(len(las_stats)), key=lambda i: las_stats[i]["mean"], reverse=True)

    names = [las_stats[i]["name"] for i in las_order]
    cats = [las_stats[i]["category"] for i in las_order]
    las_means = [las_stats[i]["mean"] for i in las_order]
    las_stds = [las_stats[i]["std"] for i in las_order]

    # Match spaghetti stats by name
    spag_by_name = {s["name"]: s for s in spag_stats}
    spag_means = [spag_by_name[n]["mean"] for n in names]
    spag_stds = [spag_by_name[n]["std"] for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 8))

    # Spaghetti bars (solid)
    colors = [CATEGORY_COLORS.get(c, "#888888") for c in cats]
    ax.bar(x - width / 2, spag_means, width, yerr=spag_stds, capsize=2,
           color=colors, edgecolor="white", linewidth=0.5, label="Spaghetti (3 steps)")

    # Lasagna bars (hatched)
    ax.bar(x + width / 2, las_means, width, yerr=las_stds, capsize=2,
           color=colors, edgecolor="white", linewidth=0.5, hatch="//", alpha=0.85,
           label="Lasagna (21 steps)")

    # Baselines
    spag_baseline = spag_by_name["Baseline - None"]["mean"]
    las_baseline = next(s["mean"] for s in las_stats if s["name"] == "Baseline - None")
    ax.axhline(y=spag_baseline, color="#aaaaaa", linestyle=":", alpha=0.6, linewidth=1)
    ax.axhline(y=las_baseline, color="#aaaaaa", linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xlabel("System Prompt (sorted by lasagna mean)", fontsize=11)
    ax.set_ylabel("Mean Final Cosine Similarity (5 seeds)", fontsize=11)
    ax.set_title(
        "Complexity Threshold — 30B AWQ\n"
        "Spaghetti (3 steps, solid) vs Lasagna (21 steps, hatched)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7.5)
    ax.set_ylim(0.75, 1.0)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save_plot(fig, "complexity_threshold.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Cross-model slope chart — 30b vs 8b lasagna rankings
# ──────────────────────────────────────────────────────────────────────
def plot_cross_model_ranking(las30b: dict, d8b: dict) -> float:
    stats_30b = prompt_stats(
        las30b, lambda p: f"{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
    )
    stats_8b = prompt_stats(
        d8b, lambda p: f"lasagna_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
    )

    # Sort each by mean descending
    order_30b = sorted(stats_30b, key=lambda s: s["mean"], reverse=True)
    order_8b = sorted(stats_8b, key=lambda s: s["mean"], reverse=True)

    rank_30b = {s["name"]: i for i, s in enumerate(order_30b)}
    rank_8b = {s["name"]: i for i, s in enumerate(order_8b)}

    # Spearman
    names = [p["name"] for p in PROMPTS]
    r30 = [rank_30b[n] for n in names]
    r8 = [rank_8b[n] for n in names]
    rho, pval = spearmanr(r30, r8)

    n = len(PROMPTS)
    fig, ax = plt.subplots(figsize=(10, 12))

    left_x, right_x = 0.2, 0.8

    for p in PROMPTS:
        name = p["name"]
        y_left = rank_30b[name]
        y_right = rank_8b[name]
        color = CATEGORY_COLORS.get(p["category"], "#888888")

        ax.plot([left_x, right_x], [y_left, y_right],
                color=color, linewidth=1.8, alpha=0.7)
        ax.plot(left_x, y_left, "o", color=color, markersize=7, zorder=5)
        ax.plot(right_x, y_right, "o", color=color, markersize=7, zorder=5)

        # Labels
        mean_30b = next(s["mean"] for s in stats_30b if s["name"] == name)
        mean_8b = next(s["mean"] for s in stats_8b if s["name"] == name)

        ax.text(left_x - 0.02, y_left, f"{name} ({mean_30b:.3f})",
                ha="right", va="center", fontsize=7, color=color)
        ax.text(right_x + 0.02, y_right, f"{name} ({mean_8b:.3f})",
                ha="left", va="center", fontsize=7, color=color)

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels(["30B AWQ (vLLM)", "8B q4_K_M (Ollama)"], fontsize=12, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{i+1}" for i in range(n)], fontsize=8)
    ax.set_ylabel("Rank (1 = best)", fontsize=11)
    ax.set_title(
        f"System Prompt Rankings: 30B vs 8B on Lasagna\n"
        f"Spearman \u03c1 = {rho:.3f} (p = {pval:.3f}) — rankings essentially uncorrelated",
        fontsize=13,
    )
    ax.grid(True, axis="y", alpha=0.15)

    # Category legend
    cat_handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    ax.legend(handles=cat_handles, fontsize=8, loc="lower center", ncol=3)

    fig.tight_layout()
    _save_plot(fig, "cross_model_ranking.png")

    return rho


# ──────────────────────────────────────────────────────────────────────
# Plot 4: 8b spaghetti seed variance — top 6 most variable prompts
# ──────────────────────────────────────────────────────────────────────
def plot_seed_variance(d8b: dict) -> None:
    # Compute per-prompt spread
    prompt_spreads = []
    for p in PROMPTS:
        slug = _slug(p["name"])
        pattern = f"spaghetti_{p['id']:02d}_{slug}_s{{seed}}"
        curves = get_seed_curves(d8b, pattern)
        finals = [c[-1] for c in curves if c]
        if finals:
            spread = max(finals) - min(finals)
            prompt_spreads.append((spread, p, curves))

    # Top 6 most variable
    prompt_spreads.sort(key=lambda x: x[0], reverse=True)
    show = prompt_spreads[:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)

    for idx, (spread, p, curves) in enumerate(show):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        color = CATEGORY_COLORS.get(p["category"], "#888888")

        for i, curve in enumerate(curves):
            xs = np.arange(1, len(curve) + 1)
            ax.plot(xs, curve, color=color, linewidth=1.2, alpha=0.6,
                    marker="o", markersize=1.5, label=f"seed {SEEDS[i]}")

        finals = [c[-1] for c in curves if c]
        ax.set_title(f"{p['name']}\nspread={spread:.3f} [{min(finals):.2f}\u2013{max(finals):.2f}]",
                     fontsize=9, fontweight="bold")
        ax.set_xlim(0.5, ITERATIONS + 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.set_xticks([1, 10, 20, 30])
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower left")

    fig.suptitle(
        "Seed Variance on 8B q4_K_M — Spaghetti Recipe\n"
        "6 Most Variable System Prompts (5 seeds each)",
        fontsize=13,
    )
    fig.tight_layout()
    _save_plot(fig, "seed_variance_8b.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 5: Category heatmap — 6 categories × 4 conditions
# ──────────────────────────────────────────────────────────────────────
def plot_category_heatmap(spag30b: dict, las30b: dict, d8b: dict) -> dict:
    conditions = ["30B\nSpaghetti", "30B\nLasagna", "8B\nSpaghetti", "8B\nLasagna"]

    # Compute category means for each condition
    cat_means = {}  # category -> [4 means]
    for cat in CATEGORY_ORDER:
        prompts_in_cat = [p for p in PROMPTS if p["category"] == cat]
        means_per_cond = []

        # 30b spaghetti 1x
        vals = []
        for p in prompts_in_cat:
            f = get_finals(spag30b, f"{p['id']:02d}_{_slug(p['name'])}_1x_s{{seed}}")
            if f:
                vals.append(float(np.mean(f)))
        means_per_cond.append(float(np.mean(vals)) if vals else 0)

        # 30b lasagna
        vals = []
        for p in prompts_in_cat:
            f = get_finals(las30b, f"{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")
            if f:
                vals.append(float(np.mean(f)))
        means_per_cond.append(float(np.mean(vals)) if vals else 0)

        # 8b spaghetti
        vals = []
        for p in prompts_in_cat:
            f = get_finals(d8b, f"spaghetti_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")
            if f:
                vals.append(float(np.mean(f)))
        means_per_cond.append(float(np.mean(vals)) if vals else 0)

        # 8b lasagna
        vals = []
        for p in prompts_in_cat:
            f = get_finals(d8b, f"lasagna_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")
            if f:
                vals.append(float(np.mean(f)))
        means_per_cond.append(float(np.mean(vals)) if vals else 0)

        cat_means[cat] = means_per_cond

    grid = np.array([cat_means[c] for c in CATEGORY_ORDER])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=0.75, vmax=0.95)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels(CATEGORY_ORDER, fontsize=11)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Prompt Category", fontsize=12)
    ax.set_title(
        "Mean Final Similarity by Category and Condition\n"
        "6 Categories \u00d7 4 Conditions (2 models \u00d7 2 recipes)",
        fontsize=13,
    )

    # Annotate cells
    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(conditions)):
            val = grid[i, j]
            text_color = "white" if val < 0.83 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Final Cosine Similarity")
    fig.tight_layout()
    _save_plot(fig, "category_heatmap.png")

    return cat_means


# ──────────────────────────────────────────────────────────────────────
# Console summary + JSON export
# ──────────────────────────────────────────────────────────────────────
def print_summary(
    spag30b: dict, las30b: dict, d8b: dict, rho: float, cat_means: dict
) -> None:
    print(f"\n{'='*80}")
    print("SYSTEM PROMPT ABLATION SUMMARY")
    print(f"{'='*80}")

    # --- 30b lasagna (main result) ---
    print(f"\n--- 30B AWQ Lasagna (sorted by mean) ---")
    las_stats = prompt_stats(
        las30b, lambda p: f"{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
    )
    las_stats.sort(key=lambda s: s["mean"], reverse=True)
    print(f"  {'Prompt':<35} {'Category':<14} {'Mean':>7} {'Std':>7}")
    print(f"  {'-'*35} {'-'*14} {'-'*7} {'-'*7}")
    for s in las_stats:
        print(f"  {s['name']:<35} {s['category']:<14} {s['mean']:>7.4f} {s['std']:>7.4f}")
    las_means = [s["mean"] for s in las_stats]
    print(f"\n  Spread: {max(las_means) - min(las_means):.4f}")

    # --- 30b spaghetti 1x ---
    print(f"\n--- 30B AWQ Spaghetti 1x ---")
    spag_stats = prompt_stats(
        spag30b, lambda p: f"{p['id']:02d}_{_slug(p['name'])}_1x_s{{seed}}"
    )
    spag_means = [s["mean"] for s in spag_stats]
    print(f"  Spread: {max(spag_means) - min(spag_means):.4f} "
          f"(range: {min(spag_means):.4f}\u2013{max(spag_means):.4f})")

    # --- 8b lasagna ---
    print(f"\n--- 8B q4_K_M Lasagna (sorted by mean) ---")
    las8b_stats = prompt_stats(
        d8b, lambda p: f"lasagna_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
    )
    las8b_stats.sort(key=lambda s: s["mean"], reverse=True)
    print(f"  {'Prompt':<35} {'Category':<14} {'Mean':>7} {'Std':>7}")
    print(f"  {'-'*35} {'-'*14} {'-'*7} {'-'*7}")
    for s in las8b_stats:
        print(f"  {s['name']:<35} {s['category']:<14} {s['mean']:>7.4f} {s['std']:>7.4f}")
    las8b_means = [s["mean"] for s in las8b_stats]
    print(f"\n  Spread: {max(las8b_means) - min(las8b_means):.4f}")

    # --- 8b spaghetti ---
    print(f"\n--- 8B q4_K_M Spaghetti ---")
    spag8b_stats = prompt_stats(
        d8b, lambda p: f"spaghetti_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
    )
    spag8b_means = [s["mean"] for s in spag8b_stats]
    print(f"  Spread: {max(spag8b_means) - min(spag8b_means):.4f} "
          f"(range: {min(spag8b_means):.4f}\u2013{max(spag8b_means):.4f})")

    # --- Cross-model correlation ---
    print(f"\n--- Cross-Model Ranking (Lasagna) ---")
    print(f"  Spearman rho: {rho:.4f}")

    # --- Fixed-point attractor analysis (30b lasagna) ---
    print(f"\n--- Fixed-Point Attractors (30B Lasagna) ---")
    fp_count = 0
    total_runs = 0
    for p in PROMPTS:
        slug = _slug(p["name"])
        for seed in SEEDS:
            key = f"{p['id']:02d}_{slug}_s{seed}"
            if key not in las30b:
                continue
            total_runs += 1
            iters = las30b[key]
            sims = [r["cosine_similarity"] for r in iters
                    if r.get("cosine_similarity") is not None]
            if len(sims) >= 10:
                tail = sims[-10:]
                if max(tail) - min(tail) < 0.005:
                    fp_count += 1
    print(f"  Runs locked into fixed point (last 10 iters spread < 0.005): "
          f"{fp_count}/{total_runs} ({fp_count / total_runs * 100:.1f}%)")

    # --- Category means ---
    print(f"\n--- Category Means ---")
    print(f"  {'Category':<14} {'30B spag':>9} {'30B las':>9} {'8B spag':>9} {'8B las':>9}")
    for cat in CATEGORY_ORDER:
        vals = cat_means[cat]
        print(f"  {cat:<14} {vals[0]:>9.4f} {vals[1]:>9.4f} {vals[2]:>9.4f} {vals[3]:>9.4f}")

    # --- Save summary JSON ---
    summary = {
        "30b_lasagna": {s["name"]: {"mean": s["mean"], "std": s["std"], "category": s["category"]}
                        for s in las_stats},
        "30b_spaghetti_1x": {s["name"]: {"mean": s["mean"], "std": s["std"], "category": s["category"]}
                             for s in spag_stats},
        "8b_lasagna": {s["name"]: {"mean": s["mean"], "std": s["std"], "category": s["category"]}
                       for s in las8b_stats},
        "8b_spaghetti": {s["name"]: {"mean": s["mean"], "std": s["std"], "category": s["category"]}
                         for s in spag8b_stats},
        "cross_model_spearman_rho": round(rho, 4),
        "fixed_point_fraction": round(fp_count / total_runs, 4) if total_runs > 0 else 0,
        "category_means": cat_means,
        "spreads": {
            "30b_spaghetti_1x": round(max(spag_means) - min(spag_means), 4),
            "30b_lasagna": round(max(las_means) - min(las_means), 4),
            "8b_spaghetti": round(max(spag8b_means) - min(spag8b_means), 4),
            "8b_lasagna": round(max(las8b_means) - min(las8b_means), 4),
        },
    }
    stats_path = OUT_DIR / "summary_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary stats saved: {stats_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BLOG_IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    print(f"  30b spaghetti: {SPAG30B_DATA}")
    print(f"  30b lasagna:   {LAS30B_DATA}")
    print(f"  8b both:       {DATA_8B}")
    spag30b, las30b, d8b = load_data()
    print(f"  30b spaghetti keys: {len(spag30b)}")
    print(f"  30b lasagna keys:   {len(las30b)}")
    print(f"  8b keys:            {len(d8b)}")

    print("\nGenerating plots...")
    plot_hero_lasagna(las30b)
    plot_complexity_threshold(spag30b, las30b)
    rho = plot_cross_model_ranking(las30b, d8b)
    plot_seed_variance(d8b)
    cat_means = plot_category_heatmap(spag30b, las30b, d8b)

    print_summary(spag30b, las30b, d8b, rho, cat_means)

    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"Blog images in: {BLOG_IMG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
