#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Unified analysis script for arXiv paper figures, tables, and statistics.

Loads all 9 experiment datasets and generates:
  - 13 figures (10 main + 3 appendix) at 300 DPI
  - 2 LaTeX tables (booktabs)
  - all_stats.json summary

Usage:
    uv run python scripts/analyze_paper.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import kruskal, wilcoxon, spearmanr, pearsonr, rankdata, norm

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS AND METADATA
# ═══════════════════════════════════════════════════════════════════════════════

# --- Paths ---
RQ1_DATA = Path("results/drift_experiment_rq1_seeds/drift_data.json")
RQ2_Q8_DATA = Path("results/drift_experiment_rq2_seeds/drift_data_q8.json")
RQ2_FP16_DATA = Path("results/drift_experiment_rq2_seeds/drift_data_fp16.json")
TEMP_DATA = Path("results/drift_experiment_temperature/drift_data.json")
LASAGNA_DATA = Path("results/drift_experiment_rq1_lasagna/drift_data.json")
SPAG30B_DATA = Path("results/drift_experiment_sysprompt/drift_data.json")
LAS30B_DATA = Path("results/drift_experiment_sysprompt_lasagna/drift_data.json")
DATA_8B = Path("results/drift_experiment_sysprompt_qwen3_8b/drift_data.json")
EMBED_STATS = Path("results/embedding_validation/correlation_stats.json")
EMBED_RAW = Path("results/embedding_validation/raw_similarities.json")

OUT_DIR = Path("results/paper_figures")
TABLE_DIR = Path("paper/tables")

ITERATIONS = 30
SEEDS = [42, 137, 256, 512, 1024]
BOOTSTRAP_N = 10000
RNG = np.random.RandomState(42)

# --- RQ1 models (17) ---
RQ1_MODELS = [
    "gemma3:27b", "qwen3-next:80b", "llama3.1:8b", "gpt-oss:120b",
    "nemotron-3-nano:30b", "qwen3-coder-next", "glm-4.7-flash",
    "devstral-2:123b", "qwen3-vl:30b-gguf", "medgemma-27b",
    "qwen3-coder:30b", "ministral-3:14b", "qwen3:30b-thinking",
    "glm-4.5-air", "qwen3:30b-instruct", "devstral-small-2:24b",
    "qwen3-vl:30b-awq",
]

RQ1_DISPLAY = {
    "gemma3:27b": "Gemma3 27B",
    "qwen3-next:80b": "Qwen3-Next 80B",
    "llama3.1:8b": "Llama 3.1 8B",
    "gpt-oss:120b": "GPT-OSS 120B",
    "nemotron-3-nano:30b": "Nemotron-3 Nano 30B",
    "qwen3-coder-next": "Qwen3-Coder-Next",
    "glm-4.7-flash": "GLM 4.7 Flash",
    "devstral-2:123b": "Devstral-2 123B",
    "qwen3-vl:30b-gguf": "Qwen3-VL 30B GGUF",
    "medgemma-27b": "MedGemma 27B",
    "qwen3-coder:30b": "Qwen3-Coder 30B",
    "ministral-3:14b": "Ministral-3 14B",
    "qwen3:30b-thinking": "Qwen3 30B Thinking",
    "glm-4.5-air": "GLM 4.5 Air",
    "qwen3:30b-instruct": "Qwen3 30B Instruct",
    "devstral-small-2:24b": "Devstral-Small-2 24B",
    "qwen3-vl:30b-awq": "Qwen3-VL 30B AWQ",
}

# --- RQ2 models (10) ---
RQ2_MODELS = [
    "qwen3:0.6b-fp16", "qwen3:0.6b-q4_K_M", "qwen3:1.7b-q8_0",
    "qwen3:4b-instruct-2507-q4_K_M", "qwen3:4b-instruct-2507-fp16",
    "qwen3:8b-q4_K_M", "qwen3:30b-a3b-instruct-2507-q8_0",
    "qwen3:30b-a3b-instruct-2507-q4_K_M", "qwen3:30b-a3b-thinking-2507-q4_K_M",
    "qwen3-next:80b-q4_K_M",
]

RQ2_META = {
    "qwen3:0.6b-fp16":                   ("0.6B fp16",       "0.6B",      "fp16"),
    "qwen3:0.6b-q4_K_M":                 ("0.6B q4\\_K\\_M", "0.6B",      "q4_K_M"),
    "qwen3:1.7b-q8_0":                   ("1.7B q8\\_0",     "1.7B",      "q8_0"),
    "qwen3:4b-instruct-2507-q4_K_M":     ("4B Inst q4\\_K\\_M", "4B Inst","q4_K_M"),
    "qwen3:4b-instruct-2507-fp16":        ("4B Inst fp16",    "4B Inst",   "fp16"),
    "qwen3:8b-q4_K_M":                   ("8B q4\\_K\\_M",   "8B",        "q4_K_M"),
    "qwen3:30b-a3b-instruct-2507-q8_0":  ("30B Inst q8\\_0", "30B Inst",  "q8_0"),
    "qwen3:30b-a3b-instruct-2507-q4_K_M":("30B Inst q4\\_K\\_M","30B Inst","q4_K_M"),
    "qwen3:30b-a3b-thinking-2507-q4_K_M":("30B Think q4\\_K\\_M","30B Think","q4_K_M"),
    "qwen3-next:80b-q4_K_M":             ("80B Next q4\\_K\\_M","80B Next","q4_K_M"),
}

# Display names without LaTeX escaping (for figures)
RQ2_DISPLAY = {
    "qwen3:0.6b-fp16":                   "0.6B fp16",
    "qwen3:0.6b-q4_K_M":                 "0.6B q4_K_M",
    "qwen3:1.7b-q8_0":                   "1.7B q8_0",
    "qwen3:4b-instruct-2507-q4_K_M":     "4B Inst q4_K_M",
    "qwen3:4b-instruct-2507-fp16":        "4B Inst fp16",
    "qwen3:8b-q4_K_M":                   "8B q4_K_M",
    "qwen3:30b-a3b-instruct-2507-q8_0":  "30B Inst q8_0",
    "qwen3:30b-a3b-instruct-2507-q4_K_M":"30B Inst q4_K_M",
    "qwen3:30b-a3b-thinking-2507-q4_K_M":"30B Think q4_K_M",
    "qwen3-next:80b-q4_K_M":             "80B Next q4_K_M",
}

RQ2_SIZE_ORDER = ["0.6B", "1.7B", "4B Inst", "8B", "30B Inst", "30B Think", "80B Next"]
RQ2_QUANT_ORDER = ["fp16", "q8_0", "q4_K_M"]

# --- Temperature models (4) ---
TEMP_MODELS = ["qwen3:30b-instruct", "gemma3:27b", "qwen3:8b", "llama3.1:8b"]
TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]

# --- System prompts (20) ---
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

# --- Tiers ---
TIER_THRESHOLDS = [("Stable", 0.80), ("Moderate", 0.60), ("Degraded", 0.40), ("Catastrophic", 0.0)]
TIER_COLORS = {"Stable": "#2196f3", "Moderate": "#ff9800", "Degraded": "#9c27b0", "Catastrophic": "#f44336"}

# --- Paper style ---
PAPER_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# RQ2 overlay: 6 selected models
RQ2_OVERLAY_MODELS = [
    "qwen3:0.6b-fp16",
    "qwen3:0.6b-q4_K_M",
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen3:30b-a3b-instruct-2507-q8_0",
    "qwen3:30b-a3b-thinking-2507-q4_K_M",
    "qwen3-next:80b-q4_K_M",
]

# RQ1 small multiples: 6 tier-spanning models
RQ1_MULTIPLES_MODELS = [
    "qwen3:30b-instruct", "devstral-small-2:24b", "qwen3-next:80b",
    "medgemma-27b", "llama3.1:8b", "glm-4.5-air",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extract_sims(iterations: list[dict]) -> list[float]:
    """Extract cosine_similarity values, skipping None/error entries."""
    return [r["cosine_similarity"] for r in iterations
            if r.get("status") == "success" and r.get("cosine_similarity") is not None]


def get_seed_curves(models: dict, base_key: str, seeds: list[int] = SEEDS) -> list[np.ndarray]:
    """Get per-seed similarity curves. base_key uses {seed} placeholder."""
    curves = []
    for seed in seeds:
        key = base_key.format(seed=seed)
        if key not in models:
            continue
        sims = extract_sims(models[key])
        if sims:
            curves.append(np.array(sims))
    return curves


def get_finals(models: dict, base_key: str, seeds: list[int] = SEEDS) -> list[float]:
    """Get final cosine similarity for each seed."""
    finals = []
    for curve in get_seed_curves(models, base_key, seeds):
        finals.append(float(curve[-1]))
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
            "name": p["name"], "id": p["id"], "category": p["category"],
            "mean": mean, "std": std, "n_seeds": len(finals), "finals": finals,
        })
    return stats


def bootstrap_ci(values: list[float], stat_fn=np.mean, n_boot: int = BOOTSTRAP_N,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high)."""
    arr = np.array(values)
    point = float(stat_fn(arr))
    if len(arr) < 2:
        return point, point, point
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(arr, size=len(arr), replace=True)
        boot_stats[i] = stat_fn(sample)
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, ci_low, ci_high


def bootstrap_ci_curves(curves: list[np.ndarray], n_boot: int = BOOTSTRAP_N,
                        alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean_curve, ci_low_curve, ci_high_curve) aligned to min length."""
    min_len = min(len(c) for c in curves)
    aligned = np.array([c[:min_len] for c in curves])
    mean_curve = aligned.mean(axis=0)
    n = len(curves)
    if n < 2:
        return mean_curve, mean_curve, mean_curve
    ci_low = np.empty(min_len)
    ci_high = np.empty(min_len)
    for t in range(min_len):
        col = aligned[:, t]
        boot = np.empty(n_boot)
        for i in range(n_boot):
            boot[i] = RNG.choice(col, size=n, replace=True).mean()
        ci_low[t] = np.percentile(boot, 100 * alpha / 2)
        ci_high[t] = np.percentile(boot, 100 * (1 - alpha / 2))
    return mean_curve, ci_low, ci_high


def assign_tier(mean_final: float) -> str:
    for tier_name, threshold in TIER_THRESHOLDS:
        if mean_final >= threshold:
            return tier_name
    return "Catastrophic"


def detect_fixed_point(sims: list[float], tail: int = 10, threshold: float = 0.005) -> bool:
    if len(sims) < tail:
        return False
    return (max(sims[-tail:]) - min(sims[-tail:])) < threshold


def cohens_d(group1: list[float], group2: list[float]) -> float:
    a, b = np.array(group1), np.array(group2)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def dunn_posthoc(groups: dict[str, list[float]]) -> dict[tuple[str, str], float]:
    """Dunn's test with Bonferroni correction. Returns {(g1,g2): p_adj}."""
    all_vals = []
    group_labels = []
    for name, vals in groups.items():
        all_vals.extend(vals)
        group_labels.extend([name] * len(vals))
    ranks = rankdata(all_vals)

    group_ranks = {}
    idx = 0
    for name, vals in groups.items():
        group_ranks[name] = ranks[idx:idx + len(vals)]
        idx += len(vals)

    N = len(all_vals)
    names = list(groups.keys())
    n_comparisons = len(names) * (len(names) - 1) // 2
    results = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n_i = len(groups[names[i]])
            n_j = len(groups[names[j]])
            mean_i = group_ranks[names[i]].mean()
            mean_j = group_ranks[names[j]].mean()
            se = np.sqrt((N * (N + 1) / 12) * (1.0 / n_i + 1.0 / n_j))
            if se == 0:
                results[(names[i], names[j])] = 1.0
                continue
            z = abs(mean_i - mean_j) / se
            p = 2 * (1 - norm.cdf(z))
            p_adj = min(p * n_comparisons, 1.0)
            results[(names[i], names[j])] = p_adj

    return results


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    text = text.replace("_", r"\_")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    return text


def save_figure(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def numpy_to_python(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STATISTICAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rq1_stats(models: dict) -> dict:
    """Compute RQ1 per-model stats, tiers, and tier-level tests."""
    per_model = {}
    tier_groups: dict[str, list[float]] = {"Stable": [], "Moderate": [], "Degraded": [], "Catastrophic": []}

    for label in RQ1_MODELS:
        finals = get_finals(models, f"{label}_s{{seed}}")
        if not finals:
            continue
        # n_full: chains that actually reached iteration 30. n: chains that
        # produced at least one scored iteration (the value the mean is over).
        n_full = 0
        for seed in SEEDS:
            key = f"{label}_s{seed}"
            if key in models and len(extract_sims(models[key])) >= ITERATIONS:
                n_full += 1
        point, ci_lo, ci_hi = bootstrap_ci(finals)
        tier = assign_tier(point)
        per_model[label] = {
            "display": RQ1_DISPLAY.get(label, label),
            "mean": point, "ci_low": ci_lo, "ci_high": ci_hi,
            "std": float(np.std(finals)), "n": len(finals),
            "n_full": n_full,
            "tier": tier, "finals": finals,
        }
        tier_groups[tier].extend(finals)

    # Kruskal-Wallis across non-empty tiers
    non_empty = {k: v for k, v in tier_groups.items() if len(v) >= 2}
    kw_result: dict = {"H": None, "p": None}
    if len(non_empty) >= 2:
        try:
            H, p = kruskal(*non_empty.values())
            kw_result = {"H": float(H), "p": float(p)}
        except Exception:
            pass

    # Dunn's post-hoc
    dunn_result = {}
    if len(non_empty) >= 2:
        dunn_raw = dunn_posthoc(non_empty)
        dunn_result = {f"{k[0]}_vs_{k[1]}": v for k, v in dunn_raw.items()}

    # Cohen's d for key contrasts
    effect_sizes = {}
    if "Stable" in non_empty and "Catastrophic" in non_empty:
        effect_sizes["stable_vs_catastrophic"] = cohens_d(
            tier_groups["Stable"], tier_groups["Catastrophic"])
    if "Stable" in non_empty and "Moderate" in non_empty:
        effect_sizes["stable_vs_moderate"] = cohens_d(
            tier_groups["Stable"], tier_groups["Moderate"])

    return {
        "per_model": per_model,
        "tier_groups": {k: v for k, v in tier_groups.items()},
        "kruskal_wallis": kw_result,
        "dunn_posthoc": dunn_result,
        "effect_sizes": effect_sizes,
    }


def compute_rq2_stats(q8_models: dict, fp16_models: dict) -> dict:
    """Compute RQ2 per-model paired stats and overall Wilcoxon."""
    per_model = {}
    model_deltas_means = []

    for label in RQ2_MODELS:
        q8_finals = get_finals(q8_models, f"{label}_s{{seed}}")
        fp16_finals = get_finals(fp16_models, f"{label}_s{{seed}}")

        if not q8_finals or not fp16_finals:
            continue

        n_pairs = min(len(q8_finals), len(fp16_finals))
        deltas = [fp16_finals[i] - q8_finals[i] for i in range(n_pairs)]

        q8_mean, q8_ci_lo, q8_ci_hi = bootstrap_ci(q8_finals)
        fp16_mean, fp16_ci_lo, fp16_ci_hi = bootstrap_ci(fp16_finals)
        delta_mean, delta_ci_lo, delta_ci_hi = bootstrap_ci(deltas)

        # Wilcoxon per model (n=5, low power)
        wil_p = None
        if n_pairs >= 5:
            try:
                _, wil_p = wilcoxon(deltas)
                wil_p = float(wil_p)
            except Exception:
                pass

        d = cohens_d(fp16_finals, q8_finals)

        per_model[label] = {
            "display": RQ2_DISPLAY.get(label, label),
            "display_latex": RQ2_META[label][0],
            "size_group": RQ2_META[label][1],
            "weight_quant": RQ2_META[label][2],
            "q8_mean": q8_mean, "q8_ci": (q8_ci_lo, q8_ci_hi),
            "fp16_mean": fp16_mean, "fp16_ci": (fp16_ci_lo, fp16_ci_hi),
            "delta_mean": delta_mean, "delta_ci": (delta_ci_lo, delta_ci_hi),
            "wilcoxon_p": wil_p, "cohens_d": d,
            "deltas": deltas, "n_pairs": n_pairs,
        }
        model_deltas_means.append(delta_mean)

    # Overall Wilcoxon on model-level mean deltas
    overall_wilcoxon: dict = {"p": None, "n": len(model_deltas_means)}
    if len(model_deltas_means) >= 5:
        try:
            _, p = wilcoxon(model_deltas_means)
            overall_wilcoxon["p"] = float(p)
        except Exception:
            pass

    return {"per_model": per_model, "overall_wilcoxon": overall_wilcoxon}


def compute_rq3_stats(spag30b: dict, las30b: dict, d8b: dict) -> dict:
    """Compute RQ3 stats: per-condition prompt stats, Spearman, fixed-point fraction."""
    stats_30b_spag = prompt_stats(
        spag30b, lambda p: f"{p['id']:02d}_{_slug(p['name'])}_1x_s{{seed}}")
    stats_30b_las = prompt_stats(
        las30b, lambda p: f"{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")
    stats_8b_spag = prompt_stats(
        d8b, lambda p: f"spaghetti_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")
    stats_8b_las = prompt_stats(
        d8b, lambda p: f"lasagna_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}")

    # Spearman: 30B vs 8B lasagna
    means_30b = [s["mean"] for s in stats_30b_las]
    means_8b = [s["mean"] for s in stats_8b_las]
    rho, rho_p = spearmanr(means_30b, means_8b)

    # Bootstrap CI on Spearman
    rho_boots = []
    indices = np.arange(len(means_30b))
    for _ in range(BOOTSTRAP_N):
        idx = RNG.choice(indices, size=len(indices), replace=True)
        r, _ = spearmanr([means_30b[i] for i in idx], [means_8b[i] for i in idx])
        rho_boots.append(r)
    rho_ci = (float(np.percentile(rho_boots, 2.5)), float(np.percentile(rho_boots, 97.5)))

    # Fixed-point fraction (30B lasagna)
    fp_count, total_runs = 0, 0
    for p in PROMPTS:
        slug = _slug(p["name"])
        for seed in SEEDS:
            key = f"{p['id']:02d}_{slug}_s{seed}"
            if key not in las30b:
                continue
            total_runs += 1
            sims = extract_sims(las30b[key])
            if detect_fixed_point(sims):
                fp_count += 1
    fp_fraction = fp_count / total_runs if total_runs > 0 else 0

    # Per-condition spreads
    spreads = {}
    for name, stats_list in [("30b_spag", stats_30b_spag), ("30b_las", stats_30b_las),
                              ("8b_spag", stats_8b_spag), ("8b_las", stats_8b_las)]:
        means = [s["mean"] for s in stats_list]
        spreads[name] = max(means) - min(means) if means else 0

    # Category means for heatmap
    cat_means = {}
    for cat in CATEGORY_ORDER:
        row = []
        for cond_stats in [stats_30b_spag, stats_30b_las, stats_8b_spag, stats_8b_las]:
            vals = [s["mean"] for s in cond_stats if s["category"] == cat]
            row.append(float(np.mean(vals)) if vals else 0)
        cat_means[cat] = row

    return {
        "30b_spag": stats_30b_spag, "30b_las": stats_30b_las,
        "8b_spag": stats_8b_spag, "8b_las": stats_8b_las,
        "spearman": {"rho": float(rho), "p": float(rho_p), "ci": rho_ci},
        "fixed_point_fraction": fp_fraction,
        "fixed_point_count": fp_count, "fixed_point_total": total_runs,
        "spreads": spreads, "category_means": cat_means,
    }


def compute_temperature_stats(models: dict) -> dict:
    """Compute temperature x model stats."""
    per_model = {}
    for label in TEMP_MODELS:
        temp_stats = {}
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            key_pattern = f"{label}_t{temp:.1f}_s{{seed}}"
            finals = get_finals(models, key_pattern, seeds_for_temp)
            if not finals:
                continue
            if len(finals) >= 2:
                point, ci_lo, ci_hi = bootstrap_ci(finals)
            else:
                point = finals[0]
                ci_lo, ci_hi = point, point
            temp_stats[temp] = {
                "mean": point, "ci_low": ci_lo, "ci_high": ci_hi,
                "n": len(finals), "finals": finals,
            }
        per_model[label] = temp_stats
    return {"per_model": per_model}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _tier_legend_label(tier: str, stats: dict) -> str:
    """Legend label for a tier, flagging bands that no model occupies."""
    occupied = any(info["tier"] == tier for info in stats["per_model"].values())
    return tier if occupied else f"{tier} (no models)"


def plot_fig1_rq1_lines(models: dict, stats: dict) -> None:
    """Fig 1: 17-model line plot with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_models = sorted(stats["per_model"].items(), key=lambda x: x[1]["mean"], reverse=True)

    for label, info in sorted_models:
        curves = get_seed_curves(models, f"{label}_s{{seed}}")
        if not curves:
            continue
        mean_curve, ci_lo, ci_hi = bootstrap_ci_curves(curves)
        xs = np.arange(1, len(mean_curve) + 1)
        color = TIER_COLORS[info["tier"]]
        display = info["display"]
        ax.plot(xs, mean_curve, color=color, linewidth=1.3,
                label=f"{display} ({info['mean']:.3f})")
        ax.fill_between(xs, ci_lo, ci_hi, color=color, alpha=0.08)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cosine Similarity to Original")
    ax.set_title("Semantic Drift Across 17 Models (5 seeds, 95% bootstrap CI)")
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    tier_handles = [mpatches.Patch(color=TIER_COLORS[t], label=_tier_legend_label(t, stats))
                    for t in ["Stable", "Moderate", "Degraded", "Catastrophic"]]
    tier_legend = ax.legend(handles=tier_handles, fontsize=8, loc="upper right", title="Tier")
    ax.add_artist(tier_legend)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6, loc="lower left", ncol=2,
              title="Model (final mean)")

    save_figure(fig, "fig1_rq1_lines.png")


def plot_fig2_rq1_boxplot(models: dict, stats: dict) -> None:
    """Fig 2: Box plot of final similarities by model."""
    sorted_models = sorted(stats["per_model"].items(), key=lambda x: x[1]["mean"], reverse=True)

    labels_list = []
    data = []
    colors = []
    for label, info in sorted_models:
        labels_list.append(info["display"])
        data.append(info["finals"])
        colors.append(TIER_COLORS[info["tier"]])

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, finals in enumerate(data):
        jitter = RNG.normal(0, 0.08, len(finals))
        ax.scatter(np.full(len(finals), i + 1) + jitter, finals,
                   color="black", s=12, alpha=0.6, zorder=5)

    ax.set_xticks(range(1, len(labels_list) + 1))
    ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Final Cosine Similarity (iteration 30)")
    ax.set_title("Distribution of Final Similarity Across Seeds (17 Models)")
    ax.set_ylim(0, 1.05)

    tier_handles = [mpatches.Patch(color=TIER_COLORS[t], alpha=0.7,
                                   label=_tier_legend_label(t, stats))
                    for t in ["Stable", "Moderate", "Degraded", "Catastrophic"]]
    ax.legend(handles=tier_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    save_figure(fig, "fig2_rq1_boxplot.png")


def plot_fig3_rq2_delta_bars(stats: dict) -> None:
    """Fig 3: KV cache delta horizontal bars with CIs and Wilcoxon p."""
    per_model = stats["per_model"]
    sorted_models = sorted(per_model.items(), key=lambda x: x[1]["delta_mean"], reverse=True)

    names = [info["display"] for _, info in sorted_models]
    deltas = [info["delta_mean"] for _, info in sorted_models]
    ci_lo = [info["delta_ci"][0] for _, info in sorted_models]
    ci_hi = [info["delta_ci"][1] for _, info in sorted_models]
    errors_lo = [d - lo for d, lo in zip(deltas, ci_lo)]
    errors_hi = [hi - d for d, hi in zip(deltas, ci_hi)]
    colors = ["#4caf50" if d >= 0 else "#f44336" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(names))
    ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.5,
            xerr=[errors_lo, errors_hi], capsize=3, error_kw={"linewidth": 1})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"$\Delta$ Final Similarity (fp16 KV $-$ q8_0 KV)")
    ax.set_title("KV Cache Precision Impact on Final Similarity (10 Qwen3 Models)")

    for i, (label, info) in enumerate(sorted_models):
        d = info["delta_mean"]
        sign = "+" if d >= 0 else ""
        p_str = f"p={info['wilcoxon_p']:.3f}" if info["wilcoxon_p"] is not None else "p=N/A"
        x_pos = max(d, ci_hi[i], 0) + 0.01
        ax.text(x_pos, i, f"{sign}{d:.3f} ({p_str})", va="center", ha="left",
                fontsize=7.5, color="black")

    fig.tight_layout()
    save_figure(fig, "fig3_rq2_delta_bars.png")


def plot_fig4_rq2_heatmap(stats: dict) -> None:
    """Fig 4: Size x weight quant heatmap of deltas."""
    per_model = stats["per_model"]
    grid = np.full((len(RQ2_SIZE_ORDER), len(RQ2_QUANT_ORDER)), np.nan)

    for label, info in per_model.items():
        size = info["size_group"]
        quant = info["weight_quant"]
        if size in RQ2_SIZE_ORDER and quant in RQ2_QUANT_ORDER:
            row = RQ2_SIZE_ORDER.index(size)
            col = RQ2_QUANT_ORDER.index(quant)
            grid[row, col] = info["delta_mean"]

    max_abs = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.01)
    norm_map = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="RdYlGn", norm=norm_map, aspect="auto")

    ax.set_xticks(range(len(RQ2_QUANT_ORDER)))
    ax.set_xticklabels(["fp16", "q8\\_0", "q4\\_K\\_M"], fontsize=10)
    ax.set_yticks(range(len(RQ2_SIZE_ORDER)))
    ax.set_yticklabels(RQ2_SIZE_ORDER, fontsize=10)
    ax.set_xlabel("Weight Quantization")
    ax.set_ylabel("Model Size")
    ax.set_title("KV Cache Delta (fp16 - q8_0) by Size and Quantization")

    for i in range(len(RQ2_SIZE_ORDER)):
        for j in range(len(RQ2_QUANT_ORDER)):
            val = grid[i, j]
            if not np.isnan(val):
                sign = "+" if val >= 0 else ""
                text_color = "black" if abs(val) < 0.3 else "white"
                ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)
            else:
                ax.text(j, i, "\u2014", ha="center", va="center", fontsize=10, color="#cccccc")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta (fp16 KV - q8_0 KV)")
    fig.tight_layout()
    save_figure(fig, "fig4_rq2_heatmap.png")


def plot_fig5_complexity(rq3_stats: dict, spag30b: dict, las30b: dict) -> None:
    """Fig 5: Complexity threshold bars -- spaghetti vs lasagna (30B)."""
    spag_stats = rq3_stats["30b_spag"]
    las_stats = rq3_stats["30b_las"]

    las_order = sorted(range(len(las_stats)), key=lambda i: las_stats[i]["mean"], reverse=True)

    names = [las_stats[i]["name"] for i in las_order]
    cats = [las_stats[i]["category"] for i in las_order]
    las_means = [las_stats[i]["mean"] for i in las_order]
    las_stds = [las_stats[i]["std"] for i in las_order]

    spag_by_name = {s["name"]: s for s in spag_stats}
    spag_means = [spag_by_name[n]["mean"] for n in names]
    spag_stds = [spag_by_name[n]["std"] for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [CATEGORY_COLORS.get(c, "#888888") for c in cats]
    ax.bar(x - width / 2, spag_means, width, yerr=spag_stds, capsize=2,
           color=colors, edgecolor="white", linewidth=0.5, label="Spaghetti (3 steps)")
    ax.bar(x + width / 2, las_means, width, yerr=las_stds, capsize=2,
           color=colors, edgecolor="white", linewidth=0.5, hatch="//", alpha=0.85,
           label="Lasagna (21 steps)")

    ax.set_xlabel("System Prompt (sorted by lasagna mean)")
    ax.set_ylabel("Mean Final Cosine Similarity (5 seeds)")
    ax.set_title("Complexity Threshold: Spaghetti vs Lasagna (30B AWQ)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7)
    ax.set_ylim(0.75, 1.0)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    save_figure(fig, "fig5_complexity_threshold.png")


def plot_fig6_hero_lasagna(rq3_stats: dict, las30b: dict) -> None:
    """Fig 6: 30B lasagna hero line plot, 20 prompts, category-colored."""
    fig, ax = plt.subplots(figsize=(14, 8))

    plot_items = []
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

    plot_items.sort(key=lambda x: x[0], reverse=True)

    for final_mean, name, category, xs, mean_curve, min_curve, max_curve in plot_items:
        color = CATEGORY_COLORS.get(category, "#888888")
        ax.plot(xs, mean_curve, color=color, linewidth=1.5,
                label=f"{name} ({final_mean:.3f})")
        ax.fill_between(xs, min_curve, max_curve, color=color, alpha=0.12)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cosine Similarity to Original")
    ax.set_title("System Prompt Ablation \u2014 30B AWQ, Lasagna Recipe (20 prompts, 5 seeds)")
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    cat_handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    cat_legend = ax.legend(handles=cat_handles, fontsize=8, loc="upper right", title="Category")
    ax.add_artist(cat_legend)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6, loc="lower left", ncol=2,
              title="System Prompt (final mean)")

    fig.tight_layout()
    save_figure(fig, "fig6_hero_lasagna.png")


def plot_fig7_cross_model(rq3_stats: dict, las30b: dict, d8b: dict) -> None:
    """Fig 7: 30B vs 8B slope chart with Spearman annotation."""
    stats_30b = rq3_stats["30b_las"]
    stats_8b = rq3_stats["8b_las"]
    spearman_stats = rq3_stats["spearman"]

    order_30b = sorted(stats_30b, key=lambda s: s["mean"], reverse=True)
    order_8b = sorted(stats_8b, key=lambda s: s["mean"], reverse=True)

    rank_30b = {s["name"]: i for i, s in enumerate(order_30b)}
    rank_8b = {s["name"]: i for i, s in enumerate(order_8b)}

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
    ax.set_ylabel("Rank (1 = best)")
    rho = spearman_stats["rho"]
    p_val = spearman_stats["p"]
    ci = spearman_stats["ci"]
    ax.set_title(
        f"System Prompt Rankings: 30B vs 8B on Lasagna\n"
        f"Spearman \u03c1 = {rho:.3f} (p = {p_val:.3f}), "
        f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
    )

    cat_handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    ax.legend(handles=cat_handles, fontsize=8, loc="lower center", ncol=3)

    fig.tight_layout()
    save_figure(fig, "fig7_cross_model.png")


def plot_fig8_category_heatmap(rq3_stats: dict) -> None:
    """Fig 8: 6 categories x 4 conditions heatmap."""
    cat_means = rq3_stats["category_means"]
    conditions = ["30B\nSpaghetti", "30B\nLasagna", "8B\nSpaghetti", "8B\nLasagna"]
    grid = np.array([cat_means[c] for c in CATEGORY_ORDER])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto", vmin=0.75, vmax=0.95)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels(CATEGORY_ORDER, fontsize=10)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Prompt Category")
    ax.set_title("Mean Final Similarity by Category and Condition")

    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(conditions)):
            val = grid[i, j]
            text_color = "white" if val < 0.83 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Final Cosine Similarity")
    fig.tight_layout()
    save_figure(fig, "fig8_category_heatmap.png")


def plot_fig9_temperature(temp_stats: dict) -> None:
    """Fig 9: Temperature x model interaction plot."""
    model_colors = {
        "qwen3:30b-instruct": "#2196f3",
        "gemma3:27b": "#4caf50",
        "qwen3:8b": "#ff9800",
        "llama3.1:8b": "#f44336",
    }
    model_display = {
        "qwen3:30b-instruct": "Qwen3 30B Instruct",
        "gemma3:27b": "Gemma3 27B",
        "qwen3:8b": "Qwen3 8B",
        "llama3.1:8b": "Llama 3.1 8B",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for label in TEMP_MODELS:
        if label not in temp_stats["per_model"]:
            continue
        ts = temp_stats["per_model"][label]
        temps = sorted(ts.keys())
        means = [ts[t]["mean"] for t in temps]
        ci_lo = [ts[t]["ci_low"] for t in temps]
        ci_hi = [ts[t]["ci_high"] for t in temps]

        errors_lo = [m - lo for m, lo in zip(means, ci_lo)]
        errors_hi = [hi - m for m, hi in zip(means, ci_hi)]

        color = model_colors.get(label, "#888888")
        display = model_display.get(label, label)
        ax.errorbar(temps, means, yerr=[errors_lo, errors_hi],
                    color=color, linewidth=2, marker="o", markersize=6,
                    capsize=4, label=display)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean Final Cosine Similarity")
    ax.set_title("Temperature Sensitivity Across 4 Models (5 seeds per temp, T=0: 1 seed)")
    ax.set_xticks(TEMPERATURES)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    save_figure(fig, "fig9_temperature.png")


def plot_fig10_embedding() -> None:
    """Fig 10: Embedding validation scatter (Qwen vs bge-m3)."""
    if not EMBED_RAW.exists() or not EMBED_STATS.exists():
        print("  SKIP fig10 (embedding validation data not found)")
        return

    raw = json.loads(EMBED_RAW.read_text())
    stats = json.loads(EMBED_STATS.read_text())

    qwen_sims = raw["qwen_sims"]
    bge_sims = raw["bge_sims"]
    pearson_r_val = stats["pearson_r"]
    spearman_rho_val = stats["spearman_rho"]
    n_total = len(qwen_sims)

    if n_total > 5000:
        idx = RNG.choice(n_total, 5000, replace=False)
        plot_qwen = [qwen_sims[i] for i in idx]
        plot_bge = [bge_sims[i] for i in idx]
        label = f"N={n_total} (5000 shown)"
    else:
        plot_qwen = qwen_sims
        plot_bge = bge_sims
        label = f"N={n_total}"

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(plot_qwen, plot_bge, alpha=0.12, s=6, color="#4488ff", edgecolors="none")

    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1, alpha=0.5, label="y = x")

    qw = np.array(plot_qwen)
    bg = np.array(plot_bge)
    m, b = np.polyfit(qw, bg, 1)
    fit_x = np.linspace(qw.min(), qw.max(), 100)
    ax.plot(fit_x, m * fit_x + b, "-", color="#f44336", linewidth=1.5,
            label=f"Linear fit (r = {pearson_r_val:.3f})")

    ax.set_xlabel("Qwen3-Embedding-0.6B Cosine Similarity")
    ax.set_ylabel("bge-m3 Cosine Similarity")
    ax.set_title(
        f"Embedding Model Validation\n"
        f"Pearson r = {pearson_r_val:.3f}, Spearman \u03c1 = {spearman_rho_val:.3f} ({label})"
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    save_figure(fig, "fig10_embedding_scatter.png")


def plot_figA1_rq2_overlay(q8_models: dict, fp16_models: dict, rq2_stats: dict) -> None:
    """Fig A1: Selected RQ2 models with CI bands, solid=fp16, dashed=q8."""
    overlay_colors = ["#4fc3f7", "#81c784", "#ce93d8", "#ffb74d", "#fff176", "#ff8a65"]

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, label in enumerate(RQ2_OVERLAY_MODELS):
        if label not in rq2_stats["per_model"]:
            continue
        color = overlay_colors[i % len(overlay_colors)]
        info = rq2_stats["per_model"][label]
        display = info["display"]
        delta = info["delta_mean"]
        sign = "+" if delta >= 0 else ""

        # q8 curves (dashed)
        q8_curves = get_seed_curves(q8_models, f"{label}_s{{seed}}")
        if q8_curves:
            mean_q8, ci_lo_q8, ci_hi_q8 = bootstrap_ci_curves(q8_curves)
            xs = np.arange(1, len(mean_q8) + 1)
            ax.plot(xs, mean_q8, linestyle="--", linewidth=1.5, color=color, alpha=0.7)
            ax.fill_between(xs, ci_lo_q8, ci_hi_q8, color=color, alpha=0.05)

        # fp16 curves (solid)
        fp16_curves = get_seed_curves(fp16_models, f"{label}_s{{seed}}")
        if fp16_curves:
            mean_fp, ci_lo_fp, ci_hi_fp = bootstrap_ci_curves(fp16_curves)
            xs = np.arange(1, len(mean_fp) + 1)
            ax.plot(xs, mean_fp, linestyle="-", linewidth=2, color=color,
                    label=f"{display} ({sign}{delta:.3f})")
            ax.fill_between(xs, ci_lo_fp, ci_hi_fp, color=color, alpha=0.08)

    style_handles = [
        Line2D([0], [0], color="gray", linewidth=2, linestyle="-", label="fp16 KV cache"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", alpha=0.7, label="q8_0 KV cache"),
    ]
    main_legend = ax.legend(fontsize=9, loc="lower left", title="Model (delta final sim)")
    ax.add_artist(main_legend)
    ax.legend(handles=style_handles, fontsize=9, loc="upper right")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cosine Similarity to Original")
    ax.set_title("KV Cache Precision: Selected Models with 95% CI Bands")
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    fig.tight_layout()
    save_figure(fig, "figA1_rq2_overlay.png")


def plot_figA2_seed_variance(d8b: dict) -> None:
    """Fig A2: 8B spaghetti seed variance -- 6 most variable prompts."""
    prompt_spreads = []
    for p in PROMPTS:
        slug = _slug(p["name"])
        pattern = f"spaghetti_{p['id']:02d}_{slug}_s{{seed}}"
        curves = get_seed_curves(d8b, pattern)
        finals = [float(c[-1]) for c in curves if len(c) > 0]
        if finals:
            spread = max(finals) - min(finals)
            prompt_spreads.append((spread, p, curves))

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

        finals = [float(c[-1]) for c in curves if len(c) > 0]
        ax.set_title(f"{p['name']}\nspread={spread:.3f} [{min(finals):.2f}\u2013{max(finals):.2f}]",
                     fontsize=9, fontweight="bold")
        ax.set_xlim(0.5, ITERATIONS + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 10, 20, 30])
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower left")

    fig.suptitle(
        "Seed Variance on 8B q4_K_M \u2014 Spaghetti Recipe\n"
        "6 Most Variable System Prompts (5 seeds each)",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, "figA2_seed_variance.png")


def plot_figA3_rq1_multiples(models: dict, stats: dict) -> None:
    """Fig A3: 2x3 grid of per-seed lines for 6 RQ1 models."""
    seed_colors = ["#2196f3", "#4caf50", "#ff9800", "#f44336", "#9c27b0"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)

    for idx, label in enumerate(RQ1_MULTIPLES_MODELS):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        if label not in stats["per_model"]:
            ax.set_visible(False)
            continue

        info = stats["per_model"][label]
        curves = get_seed_curves(models, f"{label}_s{{seed}}")
        tier_color = TIER_COLORS[info["tier"]]

        for i, curve in enumerate(curves):
            xs = np.arange(1, len(curve) + 1)
            ax.plot(xs, curve, color=seed_colors[i % len(seed_colors)],
                    linewidth=1.2, alpha=0.7, marker="o", markersize=1.5,
                    label=f"seed {SEEDS[i]}")

        ax.set_title(f"{info['display']} ({info['tier']}, \u03bc={info['mean']:.3f})",
                     fontsize=9, fontweight="bold", color=tier_color)
        ax.set_xlim(0.5, ITERATIONS + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 10, 20, 30])
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower left")

    fig.suptitle("Per-Seed Trajectories for 6 Representative Models (RQ1)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "figA3_rq1_small_multiples.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LATEX TABLE EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_table1_rq1(stats: dict) -> None:
    """Export Table 1: RQ1 model summary."""
    per_model = stats["per_model"]
    sorted_models = sorted(per_model.items(), key=lambda x: x[1]["mean"], reverse=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-model drift statistics (RQ1). Mean final cosine similarity")
    lines.append(r"  with 95\% bootstrap CI, sorted by mean descending. Five chains were")
    lines.append(r"  launched per model; $n$ is the number that returned at least one scored")
    lines.append(r"  iteration and is the number the mean and CI are computed over, while")
    lines.append(r"  $n_{30}$ is the number that ran the full 30 iterations. Rows with")
    lines.append(r"  $n<5$ or $n_{30}<n$ are marked $\dagger$: the final similarity for a")
    lines.append(r"  truncated chain is its last scored iteration, not iteration 30.}")
    lines.append(r"\label{tab:rq1}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccl}")
    lines.append(r"\toprule")
    lines.append(r"Model & Mean & 95\% CI & $n$ & $n_{30}$ & Tier \\")
    lines.append(r"\midrule")

    prev_tier = None
    for label, info in sorted_models:
        tier = info["tier"]
        if prev_tier is not None and tier != prev_tier:
            lines.append(r"\midrule")
        prev_tier = tier

        display = escape_latex(info["display"])
        n = info["n"]
        n_full = info.get("n_full", n)
        dagger = r"$^\dagger$" if (n < 5 or n_full < n) else ""
        mean = f"{info['mean']:.3f}{dagger}"
        ci = f"[{info['ci_low']:.3f}, {info['ci_high']:.3f}]"
        lines.append(f"{display} & {mean} & {ci} & {n} & {n_full} & {tier} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = TABLE_DIR / "table1_rq1.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"  table1_rq1.tex")


def export_table2_rq2(stats: dict) -> None:
    """Export Table 2: RQ2 KV cache summary."""
    per_model = stats["per_model"]
    sorted_models = sorted(per_model.items(), key=lambda x: x[1]["delta_mean"], reverse=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{KV cache precision effects (RQ2). Paired delta = fp16 KV $-$ q8\_0 KV")
    lines.append(r"  on final cosine similarity, 5 seed-matched pairs per model, with the")
    lines.append(r"  bootstrap 95\% CI of the delta ($B=10{,}000$). With $n=5$ pairs the")
    lines.append(r"  two-sided Wilcoxon signed-rank test cannot return a $p$ below 0.125,")
    lines.append(r"  so the $p$ column is reported for completeness only and no per-model")
    lines.append(r"  test here is capable of reaching $p<0.05$; the delta CIs are the")
    lines.append(r"  informative quantity.}")
    lines.append(r"\label{tab:rq2}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccll}")
    lines.append(r"\toprule")
    lines.append(r"Model & Size & Quant & q8 Mean & fp16 Mean & $\Delta$ & 95\% CI of $\Delta$ & $p$ \\")
    lines.append(r"\midrule")

    for label, info in sorted_models:
        display = info["display_latex"]
        size = escape_latex(info["size_group"])
        quant = escape_latex(info["weight_quant"])
        q8 = f"{info['q8_mean']:.3f}"
        fp16 = f"{info['fp16_mean']:.3f}"
        sign = "+" if info["delta_mean"] >= 0 else ""
        delta = f"{sign}{info['delta_mean']:.3f}"

        p_str = (f"{info['wilcoxon_p']:.3f}"
                 if info["wilcoxon_p"] is not None else "N/A")
        dci = info.get("delta_ci")
        ci_str = f"[{dci[0]:.3f}, {dci[1]:.3f}]" if dci else "---"

        lines.append(f"{display} & {size} & {quant} & {q8} & {fp16} & {delta} & "
                     f"{ci_str} & {p_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = TABLE_DIR / "table2_rq2.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"  table2_rq2.tex")


def export_table3_inventory(configs: dict[str, dict]) -> None:
    """Export Appendix B: model inventory, built from the data files themselves.

    `configs` maps experiment label -> the `models_config` block recorded in
    that experiment's drift_data.json. Parameter counts and quantization are
    parsed out of the served model tag; nothing is filled in by hand, and a
    field the tag does not state is printed as "---".
    """
    EXP_ORDER = ["RQ1", "RQ2", "RQ3", "Temp"]

    entries: dict[str, dict] = {}
    for exp_label, cfg in configs.items():
        for _key, info in (cfg or {}).items():
            tag = info.get("model", _key)
            backend = {"ollama": "Ollama", "vllm": "vLLM",
                       "openai": "vLLM"}.get(
                str(info.get("service_type", "")).lower(),
                str(info.get("service_type", "---")))
            e = entries.setdefault(tag, {"backend": backend, "exps": set()})
            e["exps"].add(exp_label)

    size_re = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z])", re.I)
    quant_pats = [
        ("q4_k_m", "Q4\\_K\\_M"), ("q8_0", "Q8\\_0"), ("q6_k", "Q6\\_K"),
        ("fp16", "fp16"), ("bf16", "BF16"), ("awq", "AWQ 4-bit"),
        ("mxfp4", "MXFP4"), ("qat", "QAT"),
    ]

    rows = []
    for tag, e in entries.items():
        m = size_re.search(tag)
        params = f"{m.group(1)}B" if m else "---"
        quant = "---"
        low = tag.lower()
        for needle, disp in quant_pats:
            if needle in low:
                quant = disp
                break
        exps = ", ".join(x for x in EXP_ORDER if x in e["exps"])
        rows.append((tag, params, quant, e["backend"], exps))

    def sort_key(r):
        try:
            return (0, float(r[1].rstrip("B")), r[0])
        except ValueError:
            return (1, 0.0, r[0])

    rows.sort(key=sort_key)

    L = []
    A = L.append
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\caption{Complete model inventory, generated from the \texttt{models\_config}")
    A(r"  block recorded in each experiment's raw data file. Parameter counts and")
    A(r"  quantization are parsed from the served model tag; ``---'' means the tag")
    A(r"  does not state that field. ``Temp'' is the temperature ablation of")
    A(r"  Section~\ref{sec:temperature}.}")
    A(r"\label{tab:inventory}")
    A(r"\scriptsize")
    A(r"\resizebox{\textwidth}{!}{%")
    A(r"\begin{tabular}{lllll}")
    A(r"\toprule")
    A(r"Served model tag & Params & Quantization & Backend & Experiments \\")
    A(r"\midrule")
    for tag, params, quant, backend, exps in rows:
        A(f"\\texttt{{{escape_latex(tag)}}} & {params} & {quant} & {backend} & {exps} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}}")
    A(r"\end{table}")

    path = TABLE_DIR / "table3_inventory.tex"
    path.write_text("\n".join(L) + "\n")
    print("  table3_inventory.tex")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY + MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def save_all_stats(rq1: dict, rq2: dict, rq3: dict, temp: dict) -> None:
    """Save all statistics to JSON."""
    all_stats = {
        "rq1": {
            "per_model": {k: {kk: vv for kk, vv in v.items() if kk != "finals"}
                          for k, v in rq1["per_model"].items()},
            "kruskal_wallis": rq1["kruskal_wallis"],
            "dunn_posthoc": rq1["dunn_posthoc"],
            "effect_sizes": rq1["effect_sizes"],
        },
        "rq2": {
            "per_model": {k: {kk: vv for kk, vv in v.items() if kk not in ("deltas",)}
                          for k, v in rq2["per_model"].items()},
            "overall_wilcoxon": rq2["overall_wilcoxon"],
        },
        "rq3": {
            "spearman": rq3["spearman"],
            "fixed_point_fraction": rq3["fixed_point_fraction"],
            "fixed_point_count": rq3["fixed_point_count"],
            "fixed_point_total": rq3["fixed_point_total"],
            "spreads": rq3["spreads"],
            "category_means": rq3["category_means"],
        },
        "temperature": {
            "per_model": {
                label: {str(t): {k: v for k, v in ts.items() if k != "finals"}
                        for t, ts in temps.items()}
                for label, temps in temp["per_model"].items()
            },
        },
    }

    if EMBED_STATS.exists():
        all_stats["embedding"] = json.loads(EMBED_STATS.read_text())

    path = OUT_DIR / "all_stats.json"
    path.write_text(json.dumps(numpy_to_python(all_stats), indent=2))
    print(f"  all_stats.json")


def print_console_summary(rq1: dict, rq2: dict, rq3: dict, temp: dict) -> None:
    """Print formatted summary for all RQs."""
    print(f"\n{'='*80}")
    print("PAPER STATISTICS SUMMARY")
    print(f"{'='*80}")

    # RQ1
    print(f"\n--- RQ1: Model Architecture ({len(rq1['per_model'])} models) ---")
    sorted_models = sorted(rq1["per_model"].items(), key=lambda x: x[1]["mean"], reverse=True)
    print(f"  {'Model':<28} {'Mean':>7} {'95% CI':>17} {'Tier':<13}")
    print(f"  {'-'*28} {'-'*7} {'-'*17} {'-'*13}")
    for label, info in sorted_models:
        print(f"  {info['display']:<28} {info['mean']:>7.3f} "
              f"[{info['ci_low']:.3f}, {info['ci_high']:.3f}] {info['tier']:<13}")

    kw = rq1["kruskal_wallis"]
    if kw["H"] is not None:
        print(f"\n  Kruskal-Wallis H = {kw['H']:.2f}, p = {kw['p']:.2e}")
    for key, val in rq1.get("dunn_posthoc", {}).items():
        print(f"  Dunn's {key}: p_adj = {val:.4f}")
    for key, val in rq1["effect_sizes"].items():
        print(f"  Cohen's d ({key}): {val:.2f}")

    # RQ2
    print(f"\n--- RQ2: KV Cache Precision ({len(rq2['per_model'])} models) ---")
    sorted_rq2 = sorted(rq2["per_model"].items(), key=lambda x: x[1]["delta_mean"], reverse=True)
    print(f"  {'Model':<20} {'q8':>7} {'fp16':>7} {'Delta':>7} {'p':>8}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for label, info in sorted_rq2:
        sign = "+" if info["delta_mean"] >= 0 else ""
        p_str = f"{info['wilcoxon_p']:.3f}" if info["wilcoxon_p"] is not None else "N/A"
        print(f"  {info['display']:<20} {info['q8_mean']:>7.3f} {info['fp16_mean']:>7.3f} "
              f"{sign}{info['delta_mean']:>6.3f} {p_str:>8}")

    ow = rq2["overall_wilcoxon"]
    if ow["p"] is not None:
        print(f"\n  Overall Wilcoxon: p = {ow['p']:.4f} (n = {ow['n']})")

    # RQ3
    print(f"\n--- RQ3: System Prompts ---")
    sp = rq3["spearman"]
    print(f"  Spearman rho (30B vs 8B lasagna): {sp['rho']:.3f} "
          f"(p = {sp['p']:.3f}), 95% CI [{sp['ci'][0]:.3f}, {sp['ci'][1]:.3f}]")
    print(f"  Fixed-point fraction (30B lasagna): "
          f"{rq3['fixed_point_count']}/{rq3['fixed_point_total']} "
          f"({rq3['fixed_point_fraction']*100:.1f}%)")
    for cond, spread in rq3["spreads"].items():
        print(f"  Spread ({cond}): {spread:.4f}")

    # Temperature
    print(f"\n--- Temperature Ablation ---")
    for label, temps in temp["per_model"].items():
        row = f"  {label:<25}"
        for t in TEMPERATURES:
            if t in temps:
                row += f"  T={t}: {temps[t]['mean']:.3f}"
        print(row)

    # Embedding
    if EMBED_STATS.exists():
        embed = json.loads(EMBED_STATS.read_text())
        print(f"\n--- Embedding Validation ---")
        print(f"  Pearson r = {embed['pearson_r']:.4f}, Spearman rho = {embed['spearman_rho']:.4f}")
        print(f"  N pairs = {embed['n_pairs']}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PAPER_STYLE)

    # --- Load all datasets ---
    print("Loading datasets...")

    rq1_raw = json.loads(RQ1_DATA.read_text())
    rq1_models = rq1_raw["models"]
    print(f"  RQ1: {len(rq1_models)} runs")

    rq2_q8_raw = json.loads(RQ2_Q8_DATA.read_text())
    rq2_q8_models = rq2_q8_raw["models"]
    print(f"  RQ2 q8: {len(rq2_q8_models)} runs")

    rq2_fp16_raw = json.loads(RQ2_FP16_DATA.read_text())
    rq2_fp16_models = rq2_fp16_raw["models"]
    print(f"  RQ2 fp16: {len(rq2_fp16_models)} runs")

    temp_raw = json.loads(TEMP_DATA.read_text())
    temp_models = temp_raw["models"]
    print(f"  Temperature: {len(temp_models)} runs")

    lasagna_raw = json.loads(LASAGNA_DATA.read_text())
    lasagna_models = lasagna_raw["models"]
    print(f"  Lasagna: {len(lasagna_models)} runs")

    spag30b_raw = json.loads(SPAG30B_DATA.read_text())
    spag30b_models = spag30b_raw["models"]
    print(f"  Sysprompt 30B spaghetti: {len(spag30b_models)} runs")

    las30b_raw = json.loads(LAS30B_DATA.read_text())
    las30b_models = las30b_raw["models"]
    print(f"  Sysprompt 30B lasagna: {len(las30b_models)} runs")

    d8b_raw = json.loads(DATA_8B.read_text())
    d8b_models = d8b_raw["models"]
    print(f"  Sysprompt 8B: {len(d8b_models)} runs")

    print(f"  Embedding validation: {EMBED_STATS.exists()}")

    # --- Compute stats ---
    print("\nComputing statistics...")
    rq1_stats = compute_rq1_stats(rq1_models)
    print(f"  RQ1: {len(rq1_stats['per_model'])} models, "
          f"H={rq1_stats['kruskal_wallis'].get('H', 'N/A')}")
    rq2_stats = compute_rq2_stats(rq2_q8_models, rq2_fp16_models)
    print(f"  RQ2: {len(rq2_stats['per_model'])} models")
    rq3_stats = compute_rq3_stats(spag30b_models, las30b_models, d8b_models)
    print(f"  RQ3: rho={rq3_stats['spearman']['rho']:.3f}")
    temp_stats = compute_temperature_stats(temp_models)
    print(f"  Temperature: {len(temp_stats['per_model'])} models")

    # --- Generate figures ---
    print("\nGenerating 13 figures...")
    plot_fig1_rq1_lines(rq1_models, rq1_stats)
    plot_fig2_rq1_boxplot(rq1_models, rq1_stats)
    plot_fig3_rq2_delta_bars(rq2_stats)
    plot_fig4_rq2_heatmap(rq2_stats)
    plot_fig5_complexity(rq3_stats, spag30b_models, las30b_models)
    plot_fig6_hero_lasagna(rq3_stats, las30b_models)
    plot_fig7_cross_model(rq3_stats, las30b_models, d8b_models)
    plot_fig8_category_heatmap(rq3_stats)
    plot_fig9_temperature(temp_stats)
    plot_fig10_embedding()
    plot_figA1_rq2_overlay(rq2_q8_models, rq2_fp16_models, rq2_stats)
    plot_figA2_seed_variance(d8b_models)
    plot_figA3_rq1_multiples(rq1_models, rq1_stats)

    inventory_configs = {
        "RQ1": rq1_raw.get("models_config", {}),
        "RQ2": {**rq2_q8_raw.get("models_config", {}),
                **rq2_fp16_raw.get("models_config", {})},
        # The three system-prompt files each record a single served model at
        # the top level rather than a models_config block.
        "RQ3": {raw.get("model", f"rq3_{i}"): {
                    "model": raw.get("model", "---"),
                    "service_type": raw.get("service_type", "---")}
                for i, raw in enumerate([spag30b_raw, las30b_raw, d8b_raw])
                if raw.get("model")},
        "Temp": temp_raw.get("models_config", {}),
    }

    # --- Export tables ---
    print("\nExporting LaTeX tables...")
    export_table1_rq1(rq1_stats)
    export_table2_rq2(rq2_stats)
    export_table3_inventory(inventory_configs)

    # --- Save stats ---
    print("\nSaving statistics...")
    save_all_stats(rq1_stats, rq2_stats, rq3_stats, temp_stats)

    # --- Console summary ---
    print_console_summary(rq1_stats, rq2_stats, rq3_stats, temp_stats)

    print(f"\n{'='*80}")
    print(f"All outputs in: {OUT_DIR}")
    print(f"Tables in: {TABLE_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
