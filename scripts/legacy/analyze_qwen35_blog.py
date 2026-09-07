#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Analyze Qwen3.5 drift experiments and generate comparison plots for Part 4 blog.

Loads new Qwen3.5 data + existing datasets and generates 5 comparison plots.

Data sources:
  - results/drift_experiment_qwen35/*/drift_data.json (new sysprompt data)
  - results/drift_experiment_temperature_qwen35/*/drift_data.json (new temp data)
  - results/drift_experiment_sysprompt/drift_data.json (existing 30B spaghetti)
  - results/drift_experiment_sysprompt_lasagna/drift_data.json (existing 30B lasagna)
  - results/drift_experiment_sysprompt_qwen3_8b/drift_data.json (existing 8B both)
  - results/drift_experiment_temperature/drift_data.json (existing 4-model temp)
  - results/drift_experiment_rq1_seeds/drift_data.json (existing 17-model baseline)

Plots:
  1. Baseline drift overlay — Qwen3.5 models on RQ1 17-model chart
  2. System prompt ranking comparison — slope chart across models
  3. Complexity threshold — spaghetti vs lasagna grouped bars
  4. Temperature comparison — Qwen3.5 vs existing models
  5. Category heatmap — extended with Qwen3.5 columns

Usage:
    uv run python scripts/analyze_qwen35_blog.py
"""

from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr

# --- Paths ---
RQ1_DATA = Path("results/drift_experiment_rq1_seeds/drift_data.json")
SPAG30B_DATA = Path("results/drift_experiment_sysprompt/drift_data.json")
LAS30B_DATA = Path("results/drift_experiment_sysprompt_lasagna/drift_data.json")
DATA_8B = Path("results/drift_experiment_sysprompt_qwen3_8b/drift_data.json")
TEMP_EXISTING = Path("results/drift_experiment_temperature/drift_data.json")
QWEN35_SYSPROMPT_DIR = Path("results/drift_experiment_qwen35")
QWEN35_TEMP_DIR = Path("results/drift_experiment_temperature_qwen35")

OUT_DIR = Path("results/qwen35_blog_analysis")
BLOG_IMG_DIR = Path("../j8web/images/llm-telephone-drift-qwen35")

ITERATIONS = 30
SEEDS = [42, 137, 256, 512, 1024]
TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]

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

# Distinct colors for models in the baseline overlay plot
MODEL_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#ffe119",
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_qwen35_data(base_dir: Path) -> dict[str, tuple[str, dict]]:
    """Find all Qwen3.5 data files. Returns {display_label: (safe_name, models_dict)}."""
    results = {}
    for data_file in sorted(base_dir.glob("*/drift_data.json")):
        data = load_json(data_file)
        model_name = data.get("model", data_file.parent.name)
        safe_name = data_file.parent.name
        results[model_name] = (safe_name, data.get("models", {}))
    return results


def get_finals_from_data(models: dict, key_pattern: str) -> list[float]:
    """Get final cosine similarity for each seed matching key_pattern ({seed} placeholder)."""
    finals = []
    for seed in SEEDS:
        key = key_pattern.format(seed=seed)
        if key not in models:
            continue
        iters = models[key]
        sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
        if sims:
            finals.append(sims[-1])
    return finals


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


def prompt_stats(models: dict, key_fn) -> list[dict]:
    """Compute per-prompt stats. key_fn(prompt) -> key_pattern with {seed}."""
    stats = []
    for p in PROMPTS:
        pattern = key_fn(p)
        finals = get_finals_from_data(models, pattern)
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


def _save_plot(fig: plt.Figure, name: str) -> None:
    """Save to results/ at 150 DPI and blog/ at 100 DPI."""
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  {name}: {path}")

    if BLOG_IMG_DIR.exists() or True:  # Always try
        BLOG_IMG_DIR.mkdir(parents=True, exist_ok=True)
        blog_path = BLOG_IMG_DIR / name
        fig.savefig(blog_path, dpi=100, bbox_inches="tight")
        print(f"  {name} (blog): {blog_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Baseline drift overlay — Qwen3.5 on RQ1 17-model chart
# ──────────────────────────────────────────────────────────────────────
def plot_baseline_overlay(rq1_models: dict, qwen35_sysprompt: dict[str, tuple[str, dict]]) -> None:
    """Overlay Qwen3.5 baseline (no system prompt) spaghetti curves on 17-model RQ1 chart."""
    fig, ax = plt.subplots(figsize=(16, 9))

    # --- RQ1 models (light gray background) ---
    rq1_labels = sorted(set(k.rsplit("_s", 1)[0] for k in rq1_models.keys()))

    for idx, label in enumerate(rq1_labels):
        curves = []
        for seed in SEEDS:
            key = f"{label}_s{seed}"
            if key not in rq1_models:
                continue
            iters = rq1_models[key]
            sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
            if sims:
                curves.append(sims)

        if not curves:
            continue

        min_len = min(len(c) for c in curves)
        aligned = np.array([c[:min_len] for c in curves])
        xs = np.arange(1, min_len + 1)
        mean_curve = aligned.mean(axis=0)
        final_mean = float(mean_curve[-1])

        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(xs, mean_curve, color=color, linewidth=1.0, alpha=0.4,
                label=f"{label} ({final_mean:.2f})")
        ax.fill_between(xs, aligned.min(axis=0), aligned.max(axis=0),
                         color=color, alpha=0.05)

    # --- Qwen3.5 models (bold, on top) ---
    qwen35_colors = {"35b": "#ff0000", "122b": "#0000ff"}  # Red for smaller, blue for larger
    for model_name, (safe_name, models) in qwen35_sysprompt.items():
        # Extract baseline spaghetti curves (prompt id 1, "Baseline - None")
        slug = _slug("Baseline - None")
        pattern = f"spaghetti_01_{slug}_s{{seed}}"
        curves = get_seed_curves(models, pattern)

        if not curves:
            print(f"  WARNING: No baseline spaghetti data for {model_name}")
            continue

        min_len = min(len(c) for c in curves)
        aligned = np.array([c[:min_len] for c in curves])
        xs = np.arange(1, min_len + 1)
        mean_curve = aligned.mean(axis=0)
        min_curve = aligned.min(axis=0)
        max_curve = aligned.max(axis=0)
        final_mean = float(mean_curve[-1])

        # Pick color based on model size hint
        color = "#ff0000"
        for size_hint, c in qwen35_colors.items():
            if size_hint in model_name.lower() or size_hint in safe_name:
                color = c
                break

        ax.plot(xs, mean_curve, color=color, linewidth=3.0, alpha=1.0,
                label=f"{model_name} ({final_mean:.2f})", zorder=10)
        ax.fill_between(xs, min_curve, max_curve, color=color, alpha=0.15, zorder=9)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax.set_title(
        "Qwen3.5 vs 17 Models — Baseline Spaghetti Drift\n"
        "Mean across 5 seeds, T=0.7, paraphrase (bold = Qwen3.5)",
        fontsize=13,
    )
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    # Sort legend by final mean descending
    handles, labels = ax.get_legend_handles_labels()
    def _sort_key(label):
        m = re.search(r"\(([\d.]+)\)$", label)
        return float(m.group(1)) if m else 0
    sorted_pairs = sorted(zip(handles, labels), key=lambda p: _sort_key(p[1]), reverse=True)
    if sorted_pairs:
        handles, labels = zip(*sorted_pairs)
    ax.legend(handles, labels, fontsize=6.5, loc="lower left", ncol=2)

    _save_plot(fig, "baseline_overlay.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 2: System prompt ranking comparison — slope chart
# ──────────────────────────────────────────────────────────────────────
def plot_ranking_comparison(
    las30b: dict,
    d8b: dict,
    qwen35_sysprompt: dict[str, tuple[str, dict]],
) -> dict[str, float]:
    """Slope chart comparing lasagna rankings across 30B, 8B, and Qwen3.5 models."""
    # Build stats for each model on lasagna
    model_stats = {}

    # 30B AWQ
    model_stats["30B AWQ"] = prompt_stats(
        las30b, lambda p: f"{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
    )
    # 8B q4_K_M
    model_stats["8B q4_K_M"] = prompt_stats(
        d8b, lambda p: f"lasagna_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
    )
    # Qwen3.5 models
    for model_name, (safe_name, models) in qwen35_sysprompt.items():
        short_label = model_name.split("/")[-1] if "/" in model_name else model_name
        stats = prompt_stats(
            models, lambda p: f"lasagna_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
        )
        # Only include if we have data
        if any(s["n_seeds"] > 0 for s in stats):
            model_stats[short_label] = stats

    if len(model_stats) < 2:
        print("  WARNING: Not enough models with lasagna data for ranking comparison")
        return {}

    # Compute rankings per model
    model_rankings = {}  # model -> {prompt_name: rank}
    model_means = {}     # model -> {prompt_name: mean}
    for model_label, stats in model_stats.items():
        ordered = sorted(stats, key=lambda s: s["mean"], reverse=True)
        model_rankings[model_label] = {s["name"]: i for i, s in enumerate(ordered)}
        model_means[model_label] = {s["name"]: s["mean"] for s in stats}

    # Pairwise Spearman correlations
    rho_results = {}
    model_labels = list(model_stats.keys())
    names = [p["name"] for p in PROMPTS]
    for i in range(len(model_labels)):
        for j in range(i + 1, len(model_labels)):
            m1, m2 = model_labels[i], model_labels[j]
            r1 = [model_rankings[m1].get(n, 19) for n in names]
            r2 = [model_rankings[m2].get(n, 19) for n in names]
            rho, pval = spearmanr(r1, r2)
            rho_results[f"{m1} vs {m2}"] = round(float(rho), 3)

    # Draw slope chart
    n_models = len(model_labels)
    n_prompts = len(PROMPTS)
    fig, ax = plt.subplots(figsize=(4 + 3 * n_models, 12))

    x_positions = np.linspace(0.15, 0.85, n_models)

    for p in PROMPTS:
        name = p["name"]
        color = CATEGORY_COLORS.get(p["category"], "#888888")

        points_x = []
        points_y = []
        for mi, model_label in enumerate(model_labels):
            rank = model_rankings[model_label].get(name, 19)
            points_x.append(x_positions[mi])
            points_y.append(rank)

        ax.plot(points_x, points_y, color=color, linewidth=1.5, alpha=0.6)
        for mi, (px, py) in enumerate(zip(points_x, points_y)):
            ax.plot(px, py, "o", color=color, markersize=6, zorder=5)

        # Label left and right
        mean_left = model_means[model_labels[0]].get(name, 0)
        mean_right = model_means[model_labels[-1]].get(name, 0)
        ax.text(x_positions[0] - 0.02, points_y[0], f"{name} ({mean_left:.3f})",
                ha="right", va="center", fontsize=6.5, color=color)
        ax.text(x_positions[-1] + 0.02, points_y[-1], f"{name} ({mean_right:.3f})",
                ha="left", va="center", fontsize=6.5, color=color)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(n_prompts - 0.5, -0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_prompts))
    ax.set_yticklabels([f"#{i+1}" for i in range(n_prompts)], fontsize=7)
    ax.set_ylabel("Rank (1 = best)", fontsize=11)

    rho_str = ", ".join(f"{k}: {v:.2f}" for k, v in rho_results.items())
    ax.set_title(
        f"System Prompt Rankings: Lasagna Recipe Across Models\n"
        f"Spearman rho: {rho_str}",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.15)

    cat_handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    ax.legend(handles=cat_handles, fontsize=8, loc="lower center", ncol=3)

    fig.tight_layout()
    _save_plot(fig, "ranking_comparison.png")

    return rho_results


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Complexity threshold — spaghetti vs lasagna grouped bars
# ──────────────────────────────────────────────────────────────────────
def plot_complexity_threshold(qwen35_sysprompt: dict[str, tuple[str, dict]]) -> None:
    """Grouped bar chart: spaghetti vs lasagna for each Qwen3.5 model, alongside summary of 30B/8B."""
    n_models = len(qwen35_sysprompt)
    if n_models == 0:
        print("  WARNING: No Qwen3.5 sysprompt data for complexity threshold plot")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(14 * n_models / 2, 8), squeeze=False)

    for ax_idx, (model_name, (safe_name, models)) in enumerate(qwen35_sysprompt.items()):
        ax = axes[0][ax_idx]
        short_label = model_name.split("/")[-1] if "/" in model_name else model_name

        pstats = []
        for p in PROMPTS:
            slug = _slug(p["name"])

            finals = {"spaghetti": [], "lasagna": []}
            for recipe in ("spaghetti", "lasagna"):
                base_key = f"{recipe}_{p['id']:02d}_{slug}"
                for seed in SEEDS:
                    key = f"{base_key}_s{seed}"
                    if key not in models:
                        continue
                    iters = models[key]
                    sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                    if sims:
                        finals[recipe].append(sims[-1])

            mean_spag = float(np.mean(finals["spaghetti"])) if finals["spaghetti"] else 0
            std_spag = float(np.std(finals["spaghetti"])) if finals["spaghetti"] else 0
            mean_las = float(np.mean(finals["lasagna"])) if finals["lasagna"] else 0
            std_las = float(np.std(finals["lasagna"])) if finals["lasagna"] else 0

            pstats.append((p["name"], p["category"], mean_spag, std_spag, mean_las, std_las))

        # Sort by lasagna mean descending
        pstats.sort(key=lambda x: x[4], reverse=True)

        names = [s[0] for s in pstats]
        categories = [s[1] for s in pstats]
        means_spag = [s[2] for s in pstats]
        stds_spag = [s[3] for s in pstats]
        means_las = [s[4] for s in pstats]
        stds_las = [s[5] for s in pstats]

        x = np.arange(len(names))
        width = 0.35
        colors = [CATEGORY_COLORS.get(c, "#888888") for c in categories]

        ax.bar(x - width / 2, means_spag, width, yerr=stds_spag, capsize=2,
               color=colors, edgecolor="white", linewidth=0.5, label="Spaghetti (3 steps)")
        ax.bar(x + width / 2, means_las, width, yerr=stds_las, capsize=2,
               color=colors, edgecolor="white", linewidth=0.5, hatch="//", alpha=0.85,
               label="Lasagna (21 steps)")

        ax.set_xlabel("System Prompt", fontsize=10)
        ax.set_ylabel("Mean Final Cosine Similarity", fontsize=10)
        ax.set_title(f"Complexity Threshold — {short_label}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Spaghetti (3 steps) vs Lasagna (21 steps) — Qwen3.5 Models\n"
        "Mean Final Similarity +/- std across 5 seeds",
        fontsize=13,
    )
    fig.tight_layout()
    _save_plot(fig, "complexity_threshold.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 4: Temperature comparison — Qwen3.5 vs existing models
# ──────────────────────────────────────────────────────────────────────
def plot_temperature_comparison(
    existing_temp_models: dict,
    qwen35_temp: dict[str, tuple[str, dict]],
) -> None:
    """Line plot of final similarity vs temperature for all models."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Existing 4 models
    existing_labels = ["qwen3:30b-instruct", "gemma3:27b", "qwen3:8b", "llama3.1:8b"]
    existing_colors = ["#4363d8", "#3cb44b", "#f58231", "#e6194b"]

    for label, color in zip(existing_labels, existing_colors):
        temp_means = []
        temp_stds = []
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            finals = []
            for seed in seeds_for_temp:
                key = f"{label}_t{temp:.1f}_s{seed}"
                if key not in existing_temp_models:
                    continue
                iters = existing_temp_models[key]
                sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                if sims:
                    finals.append(sims[-1])
            temp_means.append(float(np.mean(finals)) if finals else np.nan)
            temp_stds.append(float(np.std(finals)) if finals else 0)

        ax.errorbar(TEMPERATURES, temp_means, yerr=temp_stds, color=color,
                     linewidth=2, marker="o", markersize=6, capsize=4,
                     label=label, linestyle="--", alpha=0.7)

    # Qwen3.5 models (bold)
    qwen35_line_colors = ["#ff0000", "#0000ff", "#ff00ff", "#00cccc"]
    for (model_name, (safe_name, models)), color in zip(qwen35_temp.items(), qwen35_line_colors):
        short_label = model_name.split("/")[-1] if "/" in model_name else model_name
        temp_means = []
        temp_stds = []
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            finals = []
            for seed in seeds_for_temp:
                key = f"{model_name}_t{temp:.1f}_s{seed}"
                if key not in models:
                    continue
                iters = models[key]
                sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                if sims:
                    finals.append(sims[-1])
            temp_means.append(float(np.mean(finals)) if finals else np.nan)
            temp_stds.append(float(np.std(finals)) if finals else 0)

        ax.errorbar(TEMPERATURES, temp_means, yerr=temp_stds, color=color,
                     linewidth=3, marker="s", markersize=8, capsize=5,
                     label=short_label, linestyle="-", alpha=1.0, zorder=10)

    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel("Mean Final Cosine Similarity", fontsize=12)
    ax.set_title(
        "Temperature vs Drift Stability — Qwen3.5 (bold) vs Existing Models\n"
        "Spaghetti recipe, 30 iterations, mean +/- std across seeds",
        fontsize=13,
    )
    ax.set_xticks(TEMPERATURES)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    _save_plot(fig, "temperature_comparison.png")


# ──────────────────────────────────────────────────────────────────────
# Plot 5: Category heatmap — extended with Qwen3.5 columns
# ──────────────────────────────────────────────────────────────────────
def plot_category_heatmap(
    spag30b: dict,
    las30b: dict,
    d8b: dict,
    qwen35_sysprompt: dict[str, tuple[str, dict]],
) -> dict:
    """Heatmap of category means across all models and conditions."""
    # Build condition list: 30B spag, 30B las, 8B spag, 8B las, then Qwen3.5 models
    conditions = ["30B\nSpaghetti", "30B\nLasagna", "8B\nSpaghetti", "8B\nLasagna"]

    # Add Qwen3.5 conditions
    qwen35_labels = []
    for model_name in qwen35_sysprompt:
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        # Truncate long names
        if len(short) > 15:
            short = short[:12] + "..."
        conditions.append(f"{short}\nSpaghetti")
        conditions.append(f"{short}\nLasagna")
        qwen35_labels.append(model_name)

    cat_means = {}
    for cat in CATEGORY_ORDER:
        prompts_in_cat = [p for p in PROMPTS if p["category"] == cat]
        means_per_cond = []

        # 30b spaghetti 1x
        vals = []
        for p in prompts_in_cat:
            f = get_finals_from_data(spag30b, f"{p['id']:02d}_{_slug(p['name'])}_1x_s{{seed}}")
            if f:
                vals.append(float(np.mean(f)))
        means_per_cond.append(float(np.mean(vals)) if vals else 0)

        # 30b lasagna
        vals = []
        for p in prompts_in_cat:
            f = get_finals_from_data(las30b, f"{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")
            if f:
                vals.append(float(np.mean(f)))
        means_per_cond.append(float(np.mean(vals)) if vals else 0)

        # 8b spaghetti
        vals = []
        for p in prompts_in_cat:
            f = get_finals_from_data(d8b, f"spaghetti_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")
            if f:
                vals.append(float(np.mean(f)))
        means_per_cond.append(float(np.mean(vals)) if vals else 0)

        # 8b lasagna
        vals = []
        for p in prompts_in_cat:
            f = get_finals_from_data(d8b, f"lasagna_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")
            if f:
                vals.append(float(np.mean(f)))
        means_per_cond.append(float(np.mean(vals)) if vals else 0)

        # Qwen3.5 models
        for model_name in qwen35_labels:
            _, models = qwen35_sysprompt[model_name]
            for recipe in ("spaghetti", "lasagna"):
                vals = []
                for p in prompts_in_cat:
                    f = get_finals_from_data(
                        models,
                        f"{recipe}_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
                    )
                    if f:
                        vals.append(float(np.mean(f)))
                means_per_cond.append(float(np.mean(vals)) if vals else 0)

        cat_means[cat] = means_per_cond

    grid = np.array([cat_means[c] for c in CATEGORY_ORDER])

    fig, ax = plt.subplots(figsize=(3 + 1.5 * len(conditions), 6))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=0.70, vmax=0.98)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels(CATEGORY_ORDER, fontsize=11)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Prompt Category", fontsize=12)
    ax.set_title(
        "Mean Final Similarity by Category and Condition\n"
        "Extended with Qwen3.5 Models",
        fontsize=13,
    )

    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(conditions)):
            val = grid[i, j]
            if val == 0:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")
            else:
                text_color = "white" if val < 0.83 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Final Cosine Similarity")
    fig.tight_layout()
    _save_plot(fig, "category_heatmap.png")

    return cat_means


# ──────────────────────────────────────────────────────────────────────
# Console summary + JSON export
# ──────────────────────────────────────────────────────────────────────
def print_summary(
    qwen35_sysprompt: dict[str, tuple[str, dict]],
    qwen35_temp: dict[str, tuple[str, dict]],
    rho_results: dict,
) -> None:
    print(f"\n{'='*80}")
    print("QWEN3.5 EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    all_model_summaries = {}

    for model_name, (safe_name, models) in qwen35_sysprompt.items():
        short_label = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"\n--- {short_label} ---")

        for recipe in ("spaghetti", "lasagna"):
            recipe_label = "Spaghetti (3-step)" if recipe == "spaghetti" else "Lasagna (21-step)"
            print(f"\n  {recipe_label} (sorted by mean):")
            stats = prompt_stats(
                models,
                lambda p, r=recipe: f"{r}_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"
            )
            stats.sort(key=lambda s: s["mean"], reverse=True)

            print(f"  {'Prompt':<35} {'Category':<14} {'Mean':>7} {'Std':>7}")
            print(f"  {'-'*35} {'-'*14} {'-'*7} {'-'*7}")
            for s in stats:
                print(f"  {s['name']:<35} {s['category']:<14} {s['mean']:>7.4f} {s['std']:>7.4f}")

            means = [s["mean"] for s in stats if s["n_seeds"] > 0]
            if means:
                print(f"  Spread: {max(means) - min(means):.4f} "
                      f"(range: {min(means):.4f}-{max(means):.4f})")

            all_model_summaries[f"{short_label}_{recipe}"] = {
                s["name"]: {"mean": s["mean"], "std": s["std"], "category": s["category"]}
                for s in stats
            }

    # Temperature summary
    for model_name, (safe_name, models) in qwen35_temp.items():
        short_label = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"\n--- {short_label} Temperature ---")
        print(f"  {'Temperature':<15} {'Mean':>10} {'Std':>10} {'N':>5}")
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            finals = []
            for seed in seeds_for_temp:
                key = f"{model_name}_t{temp:.1f}_s{seed}"
                if key not in models:
                    continue
                iters = models[key]
                sims = [r["cosine_similarity"] for r in iters if r.get("cosine_similarity") is not None]
                if sims:
                    finals.append(sims[-1])
            if finals:
                print(f"  T={temp:<11.1f} {np.mean(finals):>10.4f} {np.std(finals):>10.4f} {len(finals):>5d}")
            else:
                print(f"  T={temp:<11.1f} {'N/A':>10} {'N/A':>10} {'0':>5}")

    # Cross-model correlations
    if rho_results:
        print(f"\n--- Cross-Model Ranking Correlations (Lasagna) ---")
        for pair, rho in rho_results.items():
            print(f"  {pair}: Spearman rho = {rho:.3f}")

    # Save summary JSON
    summary = {
        "model_summaries": all_model_summaries,
        "cross_model_correlations": rho_results,
    }
    stats_path = OUT_DIR / "summary_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary stats saved: {stats_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading datasets...")

    # Qwen3.5 new data
    qwen35_sysprompt = find_qwen35_data(QWEN35_SYSPROMPT_DIR)
    qwen35_temp = find_qwen35_data(QWEN35_TEMP_DIR)
    print(f"  Qwen3.5 sysprompt models: {list(qwen35_sysprompt.keys()) or 'none found'}")
    print(f"  Qwen3.5 temp models: {list(qwen35_temp.keys()) or 'none found'}")

    for model_name, (safe_name, models) in qwen35_sysprompt.items():
        print(f"    {model_name}: {len(models)} runs")
    for model_name, (safe_name, models) in qwen35_temp.items():
        print(f"    {model_name}: {len(models)} runs")

    # Existing data
    rq1_models = {}
    if RQ1_DATA.exists():
        rq1_models = load_json(RQ1_DATA).get("models", {})
        print(f"  RQ1 seeds: {len(rq1_models)} runs")
    else:
        print(f"  RQ1 seeds: NOT FOUND")

    spag30b = {}
    if SPAG30B_DATA.exists():
        spag30b = load_json(SPAG30B_DATA).get("models", {})
        print(f"  30B spaghetti: {len(spag30b)} runs")

    las30b = {}
    if LAS30B_DATA.exists():
        las30b = load_json(LAS30B_DATA).get("models", {})
        print(f"  30B lasagna: {len(las30b)} runs")

    d8b = {}
    if DATA_8B.exists():
        d8b = load_json(DATA_8B).get("models", {})
        print(f"  8B both: {len(d8b)} runs")

    existing_temp_models = {}
    if TEMP_EXISTING.exists():
        existing_temp_models = load_json(TEMP_EXISTING).get("models", {})
        print(f"  Existing temp: {len(existing_temp_models)} runs")

    if not qwen35_sysprompt and not qwen35_temp:
        print("\nERROR: No Qwen3.5 data found. Run experiments first.")
        return

    # --- Generate plots ---
    print("\nGenerating plots...")

    if qwen35_sysprompt and rq1_models:
        plot_baseline_overlay(rq1_models, qwen35_sysprompt)
    else:
        print("  Skipping baseline overlay (missing RQ1 or Qwen3.5 sysprompt data)")

    rho_results = {}
    if qwen35_sysprompt and las30b and d8b:
        rho_results = plot_ranking_comparison(las30b, d8b, qwen35_sysprompt)
    else:
        print("  Skipping ranking comparison (missing data)")

    if qwen35_sysprompt:
        plot_complexity_threshold(qwen35_sysprompt)
    else:
        print("  Skipping complexity threshold (no Qwen3.5 sysprompt data)")

    if qwen35_temp or existing_temp_models:
        plot_temperature_comparison(existing_temp_models, qwen35_temp)
    else:
        print("  Skipping temperature comparison (no temperature data)")

    if qwen35_sysprompt and spag30b and las30b and d8b:
        plot_category_heatmap(spag30b, las30b, d8b, qwen35_sysprompt)
    else:
        print("  Skipping category heatmap (missing data)")

    # --- Summary ---
    print_summary(qwen35_sysprompt, qwen35_temp, rho_results)

    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"Blog images in: {BLOG_IMG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
