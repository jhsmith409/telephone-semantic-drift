#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Unified analysis script for Qwen3.5 paper (Paper 2).

Loads 6 experiment datasets and generates:
  - 14 figures (11 main + 3 appendix) at 300 DPI
  - 4 LaTeX tables (booktabs), mirrored into paper2/tables/ and paper/tables/
  - all_stats.json summary (all-chain AND clean-chain arms + runaway rates)
  - summary_for_text.md

Usage:
    uv run python scripts/analyze_qwen35_paper.py
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
from scipy.stats import kruskal, wilcoxon, spearmanr, rankdata, norm

# =============================================================================
# 1. CONSTANTS AND METADATA
# =============================================================================

# --- Paths ---
EXP1_DATA = Path("results/qwen35_size_quant/drift_data.json")
EXP2_DATA = Path("results/qwen35_temperature/drift_data.json")
EXP3A_DATA = Path("results/qwen35_vllm/drift_data.json")
EXP3B_DATA = Path("results/qwen35_vllm_temperature/drift_data.json")
EXP3C_DATA = Path("results/qwen35_vllm_sysprompt/drift_data.json")
EXP4_DATA = Path("results/qwen35_sysprompt/drift_data.json")
PAPER1_RQ1_DATA = Path("results/drift_experiment_rq1_seeds/drift_data.json")
PAPER1_QWEN3_DATA = Path("results/drift_experiment_qwen3/drift_data.json")
NVFP4_EXP1_DATA = Path("results/qwen35_nvfp4/exp1_drift_data.json")
NVFP4_EXP2_DATA = Path("results/qwen35_nvfp4/exp2_drift_data.json")
NVFP4_SYSPROMPT_DATA = Path("results/qwen35_nvfp4/sysprompt_drift_data.json")
GPTQ_EXP1_DATA = Path("results/qwen35_gptq/exp1_drift_data.json")
GPTQ_EXP2_DATA = Path("results/qwen35_gptq/exp2_drift_data.json")
GPTQ_SYSPROMPT_DATA = Path("results/qwen35_gptq/sysprompt_drift_data.json")
AWQ9B_EXP1_DATA = Path("results/qwen35_9b_awq/exp1_drift_data.json")
AWQ9B_EXP2_DATA = Path("results/qwen35_9b_awq/exp2_drift_data.json")
AWQ9B_SYSPROMPT_DATA = Path("results/qwen35_9b_awq/sysprompt_drift_data.json")
NEMO_EXP1_DATA = Path("results/qwen35_nemo/exp1_drift_data.json")
NEMO_EXP2_DATA = Path("results/qwen35_nemo/exp2_drift_data.json")
NEMO_SYSPROMPT_DATA = Path("results/qwen35_nemo/sysprompt_drift_data.json")

OUT_DIR = Path("results/qwen35_paper_figures")
TABLE_DIR = Path("paper2/tables")
# Tables are mirrored into both paper trees under identical filenames.
TABLE_DIRS = [Path("paper2/tables"), Path("paper/tables")]

ITERATIONS = 30
# A chain counts as a completed condition only if it produced ITERATIONS
# successful, scored iterations. Partial chains are excluded from every
# statistic (never padded, never silently averaged in) and the surviving
# count is carried through as "n".
MIN_COMPLETE_ITERS = 30
SEEDS = [42, 137, 256, 512, 1024]
BOOTSTRAP_N = 10000
RNG = np.random.RandomState(42)

# --- Exp 1: 18 Size x Quant models ---
EXP1_MODELS = [
    "qwen3.5:0.8b-q8_0", "qwen3.5:0.8b-bf16",
    "qwen3.5:2b-q4_K_M", "qwen3.5:2b-q8_0", "qwen3.5:2b-bf16",
    "qwen3.5:4b-q4_K_M", "qwen3.5:4b-q8_0", "qwen3.5:4b-bf16",
    "qwen3.5:9b-q4_K_M", "qwen3.5:9b-q8_0", "qwen3.5:9b-bf16",
    "qwen3.5:27b-q4_K_M", "qwen3.5:27b-q8_0", "qwen3.5:27b-bf16",
    "qwen3.5:35b-q4_K_M", "qwen3.5:35b-q8_0", "qwen3.5:35b-bf16",
    "qwen3.5:122b-q4_K_M",
]

EXP1_DISPLAY = {}
EXP1_META = {}  # label -> (display_latex, size_group, quant)
for label in EXP1_MODELS:
    # Parse "qwen3.5:Xb-quant" or "qwen3.5:Xb" (default q4)
    m = re.match(r"qwen3\.5:(\S+?)(?:-(q4_K_M|q8_0|bf16))?$", label)
    if m:
        size_raw = m.group(1)
        quant = m.group(2) or "q4_K_M"
        # Clean size: "0.8b" -> "0.8B", "35b" -> "35B MoE"
        size_num = size_raw.rstrip("b").upper()
        size_display = f"{size_num}B"
        if size_raw in ("35b",):
            size_display = "35B MoE"
        quant_display = quant.replace("_", r"\_") if "K" in quant else quant
        display = f"{size_display} {quant}"
        display_latex = f"{size_display} {quant_display}"
        EXP1_DISPLAY[label] = display
        EXP1_META[label] = (display_latex, size_display, quant)

SIZE_ORDER = ["0.8B", "2B", "4B", "9B", "27B", "35B MoE", "122B"]
QUANT_ORDER = ["bf16", "q8_0", "q4_K_M"]

# --- Temperature models ---
TEMP_OLLAMA_MODELS = {
    "qwen3.5:2b": "2B (Q4)",
    "qwen3.5:9b": "9B (Q4)",
    "qwen3.5:27b": "27B (Q4)",
    "qwen3.5:35b": "35B MoE (Q4)",
}
TEMP_VLLM_MODEL = "qwen3.5:35b-awq"
TEMP_VLLM_DISPLAY = "35B MoE (AWQ)"
TEMP_NVFP4_MODEL = "qwen3.5:35b-nvfp4"
TEMP_NVFP4_DISPLAY = "35B MoE (NVFP4)"
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

# Sysprompt models: Ollama 9B, 27B + vLLM 35B AWQ
SYSPROMPT_MODELS = {
    "9b": ("qwen3.5:9b", "9B (Q4)"),
    "27b": ("qwen3.5:27b", "27B (Q4)"),
}

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

# --- Cross-generation matched pairs (Qwen3 -> Qwen3.5) ---
CROSS_GEN_PAIRS = [
    # (qwen3_label_in_rq1, qwen3_label_in_qwen3_quant, qwen3.5_label, display)
    # Use pilot data (no seeds) for Qwen3 quant models
    ("qwen3:0.6b-q4_K_M",                "qwen3.5:0.8b-q8_0",    "0.6B vs 0.8B"),     # closest small
    ("qwen3:1.7b-q4_K_M",                "qwen3.5:2b-q4_K_M",    "1.7B vs 2B"),
    ("qwen3:4b-instruct-2507-q4_K_M",    "qwen3.5:4b-q4_K_M",    "4B vs 4B"),
    ("qwen3:8b-q4_K_M",                  "qwen3.5:9b-q4_K_M",    "8B vs 9B"),
    ("qwen3:30b-a3b-instruct-2507-q4_K_M", "qwen3.5:35b-q4_K_M", "30B MoE vs 35B MoE"),
    ("qwen3:32b-q4_K_M",                 "qwen3.5:27b-q4_K_M",   "32B dense vs 27B dense"),
]


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extract_sims(iterations: list[dict]) -> list[float]:
    """Extract cosine_similarity values, skipping None/error entries."""
    return [r["cosine_similarity"] for r in iterations
            if r.get("status") == "success" and r.get("cosine_similarity") is not None]


def get_seed_curves(models: dict, base_key: str, seeds: list[int] = SEEDS,
                    min_iters: int = MIN_COMPLETE_ITERS) -> list[np.ndarray]:
    """Get per-seed similarity curves for COMPLETE chains only.

    base_key uses a {seed} placeholder. A chain is complete when it has at
    least `min_iters` successful, scored iterations; incomplete chains are
    dropped so that every downstream statistic reports its true n.
    """
    curves = []
    for seed in seeds:
        key = base_key.format(seed=seed)
        if key not in models:
            continue
        sims = extract_sims(models[key])
        if len(sims) >= min_iters:
            curves.append(np.array(sims[:min_iters]))
    return curves


def get_finals(models: dict, base_key: str, seeds: list[int] = SEEDS) -> list[float]:
    """Get final cosine similarity for each COMPLETE seed chain."""
    finals = []
    for curve in get_seed_curves(models, base_key, seeds):
        finals.append(float(curve[-1]))
    return finals


def get_finals_by_seed(models: dict, base_key: str,
                       seeds: list[int] = SEEDS) -> dict[int, float]:
    """Map seed -> final similarity, complete chains only (for paired tests)."""
    out = {}
    for seed in seeds:
        key = base_key.format(seed=seed)
        if key not in models:
            continue
        sims = extract_sims(models[key])
        if len(sims) >= MIN_COMPLETE_ITERS:
            out[seed] = float(sims[MIN_COMPLETE_ITERS - 1])
    return out


# -----------------------------------------------------------------------------
# Chain classification: clean / runaway / incomplete
# -----------------------------------------------------------------------------
# "Runaway thinking" iterations are successful API calls that returned an empty
# assistant message because the whole generation budget went to reasoning
# tokens. They are identifiable by output_message == "" with a non-trivial
# elapsed_seconds (a full max-token generation at that model's decode speed).
#
# The rerun harness had an early-stop rule: after two consecutive empty outputs
# it stopped calling the model and appended empty records with
# elapsed_seconds == 0 for the remaining iterations. Those are PADDING - they
# belong to a chain that already went runaway and are never counted as separate
# runaway events.
#
# The same rule is applied to every model, including 0.8B.

_CLS_CACHE: dict[int, dict] = {}


def _is_empty_output(rec: dict) -> bool:
    return not (rec.get("output_message") or "").strip()


def _is_padding(rec: dict) -> bool:
    """Empty record appended by the early-stop rule (no model call was made)."""
    return _is_empty_output(rec) and not (rec.get("elapsed_seconds") or 0)


def classify_chain(iterations: list[dict],
                   min_iters: int = MIN_COMPLETE_ITERS) -> dict:
    """Classify one chain as clean / runaway / incomplete.

    Returns a dict with:
      status        - "clean" | "runaway" | "incomplete"
      n_success     - number of successful scored iterations
      runaway_onset - 1-based iteration index of the first real empty output
                      (None for clean/incomplete chains)
      n_runaway_events - empty outputs that cost a real generation
      n_padding     - empty outputs with elapsed_seconds == 0
    """
    succ = [r for r in iterations
            if r.get("status") == "success" and r.get("cosine_similarity") is not None]
    last_ok = bool(iterations) and iterations[-1].get("status") == "success"
    info = {
        "n_success": len(succ),
        "runaway_onset": None,
        "n_runaway_events": 0,
        "n_padding": 0,
    }
    if len(succ) < min_iters or not last_ok:
        info["status"] = "incomplete"
        return info

    window = succ[:min_iters]
    real_empty = [i for i, r in enumerate(window)
                  if _is_empty_output(r) and not _is_padding(r)]
    padding = [i for i, r in enumerate(window) if _is_padding(r)]
    any_empty = [i for i, r in enumerate(window) if _is_empty_output(r)]

    info["n_runaway_events"] = len(real_empty)
    info["n_padding"] = len(padding)

    if not any_empty:
        info["status"] = "clean"
        return info

    info["status"] = "runaway"
    # Onset is the first genuine runaway generation. If a chain somehow only
    # carries padding, fall back to the first empty record of any kind.
    info["runaway_onset"] = (real_empty[0] if real_empty else any_empty[0]) + 1
    return info


def chain_classes(models: dict) -> dict[str, dict]:
    """Classify every chain in a loaded data dict (cached per dict object)."""
    cache_key = id(models)
    cached = _CLS_CACHE.get(cache_key)
    if cached is not None and len(cached) == len(models):
        return cached
    out = {key: classify_chain(iters) for key, iters in models.items()}
    _CLS_CACHE[cache_key] = out
    return out


def classify_all_files(sources: dict[str, dict]) -> dict:
    """Per-file clean / runaway / incomplete counts plus the chain-level detail."""
    per_file = {}
    for name, models in sources.items():
        cls = {key: classify_chain(iters) for key, iters in models.items()}
        counts = {"clean": 0, "runaway": 0, "incomplete": 0}
        for info in cls.values():
            counts[info["status"]] += 1
        complete = counts["clean"] + counts["runaway"]
        rate = counts["runaway"] / complete if complete else 0.0
        lo, hi = wilson_ci(counts["runaway"], complete)
        per_file[name] = {
            "clean": counts["clean"],
            "runaway": counts["runaway"],
            "incomplete": counts["incomplete"],
            "n_total": len(models),
            "runaway_rate": rate,
            "runaway_ci": [lo, hi],
            "runaway_chains": sorted(
                ({"key": k, "runaway_onset": v["runaway_onset"],
                  "n_runaway_events": v["n_runaway_events"],
                  "n_padding": v["n_padding"]}
                 for k, v in cls.items() if v["status"] == "runaway"),
                key=lambda d: d["key"]),
        }
    totals = {"clean": 0, "runaway": 0, "incomplete": 0}
    for v in per_file.values():
        for k in totals:
            totals[k] += v[k]
    complete = totals["clean"] + totals["runaway"]
    totals["runaway_rate"] = totals["runaway"] / complete if complete else 0.0
    totals["runaway_ci"] = list(wilson_ci(totals["runaway"], complete))
    return {"per_file": per_file, "totals": totals}


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, (centre - half) / denom)),
            float(min(1.0, (centre + half) / denom)))


def get_finals_split(models: dict, base_key: str,
                     seeds: list[int] = SEEDS) -> dict:
    """Final similarities for a condition, split by chain classification.

    Returns {"all": [...], "clean": [...], "n_clean": int, "n_runaway": int,
             "n_incomplete": int, "n_total": int, "onsets": [...]}.
    `all` keeps the previous behaviour (every complete chain, runaways
    included); `clean` keeps only chains with no empty outputs at all.
    """
    cls = chain_classes(models)
    out = {"all": [], "clean": [], "n_clean": 0, "n_runaway": 0,
           "n_incomplete": 0, "n_total": 0, "onsets": []}
    for seed in seeds:
        key = base_key.format(seed=seed)
        if key not in models:
            continue
        out["n_total"] += 1
        info = cls[key]
        if info["status"] == "incomplete":
            out["n_incomplete"] += 1
            continue
        sims = extract_sims(models[key])
        final = float(sims[MIN_COMPLETE_ITERS - 1])
        out["all"].append(final)
        if info["status"] == "clean":
            out["clean"].append(final)
            out["n_clean"] += 1
        else:
            out["n_runaway"] += 1
            if info["runaway_onset"] is not None:
                out["onsets"].append(info["runaway_onset"])
    return out


def _arm_stat(values: list[float]) -> dict:
    """mean / bootstrap CI / n for one arm (all or clean)."""
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if len(values) < 2:
        v = float(values[0])
        return {"mean": v, "ci_low": v, "ci_high": v, "n": 1}
    point, lo, hi = bootstrap_ci(values)
    return {"mean": point, "ci_low": lo, "ci_high": hi, "n": len(values)}


def condition_stats(models: dict, base_key: str,
                    seeds: list[int] = SEEDS) -> dict:
    """Full per-condition statistic block: all arm, clean arm, runaway rate.

    The top-level mean/ci_low/ci_high/n keys reproduce the previous
    (all-chain) behaviour so existing figures and writers keep working.
    """
    split = get_finals_split(models, base_key, seeds)
    all_stat = _arm_stat(split["all"])
    clean_stat = _arm_stat(split["clean"])
    n_complete = split["n_clean"] + split["n_runaway"]
    rate = split["n_runaway"] / n_complete if n_complete else 0.0
    lo, hi = wilson_ci(split["n_runaway"], n_complete)
    return {
        # back-compatible (all chains)
        "mean": all_stat["mean"], "ci_low": all_stat["ci_low"],
        "ci_high": all_stat["ci_high"], "n": all_stat["n"],
        "std": float(np.std(split["all"])) if split["all"] else 0.0,
        "finals": split["all"],
        # new
        "all": all_stat,
        "clean": clean_stat,
        "finals_clean": split["clean"],
        "n_clean": split["n_clean"],
        "n_runaway": split["n_runaway"],
        "n_incomplete": split["n_incomplete"],
        "n_total": split["n_total"],
        "runaway_rate": rate,
        "runaway_ci": [lo, hi],
        "runaway_onsets": split["onsets"],
        "mean_clean": clean_stat["mean"],
    }


def _fmt_rate(stat: dict) -> str:
    """'0.40 (2/5)' style runaway-rate cell for tables."""
    n = stat.get("n_clean", 0) + stat.get("n_runaway", 0)
    if not n:
        return "---"
    return f"{stat['runaway_rate']:.2f} ({stat['n_runaway']}/{n})"


def _fmt_mean(value) -> str:
    return "---" if value is None else f"{value:.3f}"


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
    """Dunn's test with Bonferroni correction."""
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
    text = text.replace("_", r"\_")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    return text


def write_table(name: str, body: str) -> None:
    """Write one LaTeX table into every configured table directory."""
    for d in TABLE_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(body)
    print(f"  {name} -> " + ", ".join(str(d) for d in TABLE_DIRS))


def save_figure(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def numpy_to_python(obj):
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


def prompt_stats(models: dict, key_fn) -> list[dict]:
    """Compute per-prompt stats. key_fn(prompt) -> key_pattern with {seed}."""
    stats = []
    for p in PROMPTS:
        pattern = key_fn(p)
        cond = condition_stats(models, pattern)
        entry = dict(cond)
        if entry["mean"] is None:
            entry["mean"] = 0.0
            entry["ci_low"] = entry["ci_high"] = 0.0
        entry.update({
            "name": p["name"], "id": p["id"], "category": p["category"],
            "n_seeds": cond["n"],
            "mean_clean": cond["clean"]["mean"],
        })
        stats.append(entry)
    return stats


# =============================================================================
# 3. STATISTICAL FUNCTIONS
# =============================================================================

def compute_rq1_stats(models: dict) -> dict:
    """Compute RQ1 per-model stats for spaghetti recipe."""
    per_model = {}
    tier_groups: dict[str, list[float]] = {"Stable": [], "Moderate": [], "Degraded": [], "Catastrophic": []}

    for label in EXP1_MODELS:
        cond = condition_stats(models, f"{label}_s{{seed}}")
        if not cond["finals"]:
            continue
        tier = assign_tier(cond["mean"])
        tier_clean = (assign_tier(cond["clean"]["mean"])
                      if cond["clean"]["mean"] is not None else tier)
        per_model[label] = {
            "display": EXP1_DISPLAY.get(label, label),
            "display_latex": EXP1_META.get(label, (label, "", ""))[0],
            "size_group": EXP1_META.get(label, ("", "", ""))[1],
            "quant": EXP1_META.get(label, ("", "", ""))[2],
            "tier": tier, "tier_clean": tier_clean,
            **cond,
        }
        tier_groups[tier].extend(cond["finals"])

    # Kruskal-Wallis across non-empty tiers
    non_empty = {k: v for k, v in tier_groups.items() if len(v) >= 2}
    kw_result: dict = {"H": None, "p": None}
    if len(non_empty) >= 2:
        try:
            H, p = kruskal(*non_empty.values())
            kw_result = {"H": float(H), "p": float(p)}
        except Exception:
            pass

    dunn_result = {}
    if len(non_empty) >= 2:
        dunn_raw = dunn_posthoc(non_empty)
        dunn_result = {f"{k[0]}_vs_{k[1]}": v for k, v in dunn_raw.items()}

    effect_sizes = {}
    if "Stable" in non_empty and "Catastrophic" in non_empty:
        effect_sizes["stable_vs_catastrophic"] = cohens_d(
            tier_groups["Stable"], tier_groups["Catastrophic"])
    if "Stable" in non_empty and "Moderate" in non_empty:
        effect_sizes["stable_vs_moderate"] = cohens_d(
            tier_groups["Stable"], tier_groups["Moderate"])

    return {
        "per_model": per_model,
        "tier_groups": tier_groups,
        "kruskal_wallis": kw_result,
        "dunn_posthoc": dunn_result,
        "effect_sizes": effect_sizes,
    }


def compute_rq1_lasagna_stats(models: dict) -> dict:
    """Compute RQ1 per-model stats for lasagna recipe."""
    per_model = {}
    for label in EXP1_MODELS:
        cond = condition_stats(models, f"lasagna_{label}_s{{seed}}")
        if not cond["finals"]:
            continue
        tier = assign_tier(cond["mean"])
        tier_clean = (assign_tier(cond["clean"]["mean"])
                      if cond["clean"]["mean"] is not None else tier)
        per_model[label] = {
            "display": EXP1_DISPLAY.get(label, label),
            "tier": tier, "tier_clean": tier_clean,
            **cond,
        }
    return {"per_model": per_model}


QUANT_METHODS = [
    # (key, display, source dict name, label prefix)
    ("gguf",  "GGUF Q4\\_K\\_M (Ollama)", "exp1",  "qwen3.5:35b-q4_K_M"),
    ("awq",   "AWQ 4-bit (vLLM)",         "exp3a", "qwen3.5:35b-awq"),
    ("nvfp4", "NVFP4 4-bit (sglang)",     "exp1",  "qwen3.5:35b-nvfp4"),
    ("gptq",  "GPTQ-Int4 (vLLM)",         "exp1",  "qwen3.5:35b-gptq"),
]

QUANT_COLORS = {"gguf": "#f44336", "awq": "#2196f3",
                "nvfp4": "#4caf50", "gptq": "#9c27b0"}


def compute_quant_comparison_stats(exp1_models: dict, exp3a_models: dict) -> dict:
    """Four-way 35B MoE quantization comparison: GGUF / AWQ / NVFP4 / GPTQ.

    Each non-GGUF method is also compared to GGUF pairwise by seed (only seeds
    where both chains completed contribute to the delta).
    """
    sources = {"exp1": exp1_models, "exp3a": exp3a_models}
    result = {}
    for recipe_prefix, recipe_name in [("", "spaghetti"), ("lasagna_", "lasagna")]:
        methods = {}
        by_seed = {}
        for key, display, src, label in QUANT_METHODS:
            seed_map = get_finals_by_seed(sources[src], f"{recipe_prefix}{label}_s{{seed}}")
            if not seed_map:
                continue
            cond = condition_stats(sources[src], f"{recipe_prefix}{label}_s{{seed}}")
            methods[key] = {
                "display": display, "label": label,
                "ci": (cond["ci_low"], cond["ci_high"]),
                **cond,
            }
            by_seed[key] = seed_map

        if "gguf" not in methods:
            result[recipe_name] = None
            continue

        deltas = {}
        for key in methods:
            if key == "gguf":
                continue
            shared = sorted(set(by_seed[key]) & set(by_seed["gguf"]))
            if not shared:
                continue
            d = [by_seed[key][s] - by_seed["gguf"][s] for s in shared]
            dm, dlo, dhi = bootstrap_ci(d)
            wil_p = None
            if len(d) >= 5:
                try:
                    _, wil_p = wilcoxon(d)
                    wil_p = float(wil_p)
                except Exception:
                    pass
            deltas[key] = {
                "delta_mean": dm, "delta_ci": (dlo, dhi), "n_pairs": len(d),
                "wilcoxon_p": wil_p,
                "cohens_d": cohens_d(methods[key]["finals"], methods["gguf"]["finals"]),
            }

        result[recipe_name] = {"methods": methods, "deltas": deltas}
    return result


# Extra temperature models held in the merged Exp 2 dict (same key format)
TEMP_EXTRA_MODELS = {
    "qwen3.5:35b-nvfp4": "35B MoE (NVFP4)",
    "qwen3.5:35b-gptq": "35B MoE (GPTQ-Int4)",
    "qwen3.5:9b-awq": "9B (AWQ)",
    "nemo:30b-nvfp4": "Nemotron 30B (NVFP4)",
}


def compute_temperature_stats(exp2_models: dict, exp3b_models: dict) -> dict:
    """Temperature stats across Ollama, vLLM, sglang models."""
    per_model = {}

    def collect(source: dict, model_key: str, display: str) -> None:
        temp_stats = {}
        for temp in TEMPERATURES:
            seeds_for_temp = [SEEDS[0]] if temp == 0.0 else SEEDS
            spag_stat = _temp_stat(condition_stats(
                source, f"{model_key}_t{temp:.1f}_s{{seed}}", seeds_for_temp))
            las_stat = _temp_stat(condition_stats(
                source, f"lasagna_{model_key}_t{temp:.1f}_s{{seed}}", seeds_for_temp))
            if spag_stat or las_stat:
                temp_stats[temp] = {"spaghetti": spag_stat, "lasagna": las_stat}
        if temp_stats:
            per_model[model_key] = {"display": display, "temps": temp_stats}

    for model_key, display in TEMP_OLLAMA_MODELS.items():
        collect(exp2_models, model_key, display)

    if exp3b_models:
        collect(exp3b_models, TEMP_VLLM_MODEL, TEMP_VLLM_DISPLAY)

    for model_key, display in TEMP_EXTRA_MODELS.items():
        collect(exp2_models, model_key, display)

    return {"per_model": per_model}


def _temp_stat(cond: dict) -> dict | None:
    """Keep the previous mean/ci/n/finals shape; carry the new fields along."""
    if not cond["finals"]:
        return None
    return cond


# Sysprompt conditions with flat keys ("{recipe}_{id}_{slug}_s{seed}"),
# one data file per model: short_key -> display
FLAT_SYSPROMPT_MODELS = {
    "35b_awq":   "35B AWQ",
    "35b_nvfp4": "35B NVFP4",
    "35b_gptq":  "35B GPTQ",
    "9b_awq":    "9B AWQ",
    "nemo_30b":  "Nemotron 30B NVFP4",
}

# Order used for the RQ4 table / cross-model figure
RQ4_MODEL_ORDER = ["9b", "27b", "35b_awq", "35b_nvfp4", "35b_gptq",
                   "9b_awq", "nemo_30b"]
RQ4_DISPLAY = {
    "9b": "9B (GGUF Q4)", "27b": "27B (GGUF Q4)", "35b_awq": "35B AWQ",
    "35b_nvfp4": "35B NVFP4", "35b_gptq": "35B GPTQ", "9b_awq": "9B AWQ",
    "nemo_30b": "Nemotron 30B (control)",
}


def compute_sysprompt_stats(exp4_models: dict, flat_sources: dict[str, dict]) -> dict:
    """System prompt stats for the Ollama models and every flat-key model file."""
    conditions = {}

    # Ollama: key = "{model_tag}_{recipe}_{id:02d}_{slug}_s{seed}"
    for short_key, (model_tag, display) in SYSPROMPT_MODELS.items():
        for recipe in ["spaghetti", "lasagna"]:
            cond_name = f"{short_key}_{recipe}"

            def make_key_fn(mt, rec):
                return lambda p: f"{mt}_{rec}_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"

            stats = prompt_stats(exp4_models, make_key_fn(model_tag, recipe))
            if any(s["n_seeds"] for s in stats):
                conditions[cond_name] = {"display": f"{display} {recipe}", "stats": stats}

    # Flat-key files (vLLM / sglang): key = "{recipe}_{id:02d}_{slug}_s{seed}"
    for short_key, source in flat_sources.items():
        if not source:
            continue
        display = FLAT_SYSPROMPT_MODELS.get(short_key, short_key)
        for recipe in ["spaghetti", "lasagna"]:
            cond_name = f"{short_key}_{recipe}"

            def make_key_fn_flat(rec):
                return lambda p: f"{rec}_{p['id']:02d}_{_slug(p['name'])}_s{{seed}}"

            stats = prompt_stats(source, make_key_fn_flat(recipe))
            if any(s["n_seeds"] for s in stats):
                conditions[cond_name] = {"display": f"{display} {recipe}", "stats": stats}

    # Spearman: pairwise on lasagna, all available models
    spearman_keys = [f"{m}_lasagna" for m in RQ4_MODEL_ORDER
                     if f"{m}_lasagna" in conditions]
    spearman_pairs = {}
    for i, a_key in enumerate(spearman_keys):
        for b_key in spearman_keys[i + 1:]:
            sa = conditions[a_key]["stats"]
            sb = conditions[b_key]["stats"]
            paired = [(x["mean"], y["mean"]) for x, y in zip(sa, sb)
                      if x["n_seeds"] > 0 and y["n_seeds"] > 0]
            if len(paired) < 5:
                continue
            means_a = [v[0] for v in paired]
            means_b = [v[1] for v in paired]
            rho, rho_p = spearmanr(means_a, means_b)
            rho_boots = []
            indices = np.arange(len(means_a))
            for _ in range(BOOTSTRAP_N):
                idx = RNG.choice(indices, size=len(indices), replace=True)
                r, _ = spearmanr([means_a[j] for j in idx], [means_b[j] for j in idx])
                if not np.isnan(r):
                    rho_boots.append(r)
            rho_ci = (float(np.percentile(rho_boots, 2.5)),
                      float(np.percentile(rho_boots, 97.5))) if rho_boots else (float("nan"),) * 2
            spearman_pairs[f"{a_key}_vs_{b_key}"] = {
                "rho": float(rho), "p": float(rho_p), "ci": rho_ci,
                "n_prompts": len(paired),
            }

    # Category means for heatmap
    cond_order = []
    for m in RQ4_MODEL_ORDER:
        for recipe in ["spaghetti", "lasagna"]:
            if f"{m}_{recipe}" in conditions:
                cond_order.append(f"{m}_{recipe}")
    cat_means = {}
    for cat in CATEGORY_ORDER:
        row = []
        for cname in cond_order:
            vals = [s["mean"] for s in conditions[cname]["stats"]
                    if s["category"] == cat and s["n_seeds"] > 0]
            row.append(float(np.mean(vals)) if vals else 0)
        cat_means[cat] = row

    # Spreads / best / worst per condition
    spreads = {}
    extremes = {}
    spreads_clean = {}
    extremes_clean = {}
    runaway_by_condition = {}
    for cname, cdata in conditions.items():
        valid = [s for s in cdata["stats"] if s["n_seeds"] > 0]
        if len(valid) >= 2:
            best = max(valid, key=lambda s: s["mean"])
            worst = min(valid, key=lambda s: s["mean"])
            spreads[cname] = best["mean"] - worst["mean"]
            extremes[cname] = {
                "best": {"name": best["name"], "mean": best["mean"], "n": best["n_seeds"]},
                "worst": {"name": worst["name"], "mean": worst["mean"], "n": worst["n_seeds"]},
                "n_prompts": len(valid),
                "mean_of_prompts": float(np.mean([s["mean"] for s in valid])),
                "min_n": min(s["n_seeds"] for s in valid),
            }
        else:
            spreads[cname] = 0

        # Same summary computed on clean chains only.
        valid_c = [s for s in cdata["stats"] if s["n_clean"] > 0]
        if len(valid_c) >= 2:
            bestc = max(valid_c, key=lambda s: s["clean"]["mean"])
            worstc = min(valid_c, key=lambda s: s["clean"]["mean"])
            spreads_clean[cname] = bestc["clean"]["mean"] - worstc["clean"]["mean"]
            extremes_clean[cname] = {
                "best": {"name": bestc["name"], "mean": bestc["clean"]["mean"],
                         "n": bestc["n_clean"]},
                "worst": {"name": worstc["name"], "mean": worstc["clean"]["mean"],
                          "n": worstc["n_clean"]},
                "n_prompts": len(valid_c),
                "mean_of_prompts": float(np.mean([s["clean"]["mean"] for s in valid_c])),
                "min_n": min(s["n_clean"] for s in valid_c),
            }
        else:
            spreads_clean[cname] = 0

        n_run = sum(s["n_runaway"] for s in cdata["stats"])
        n_cmp = sum(s["n_clean"] + s["n_runaway"] for s in cdata["stats"])
        lo, hi = wilson_ci(n_run, n_cmp)
        runaway_by_condition[cname] = {
            "n_runaway": n_run, "n_complete": n_cmp,
            "runaway_rate": (n_run / n_cmp) if n_cmp else 0.0,
            "runaway_ci": [lo, hi],
        }

    return {
        "conditions": conditions,
        "spearman_pairs": spearman_pairs,
        "category_means": cat_means,
        "cond_order": cond_order,
        "spreads": spreads,
        "extremes": extremes,
        "spreads_clean": spreads_clean,
        "extremes_clean": extremes_clean,
        "runaway_by_condition": runaway_by_condition,
    }


def compute_cross_gen_stats(exp1_models: dict, qwen3_pilot: dict) -> dict:
    """Cross-generation comparison: Qwen3 pilot (no seeds) vs Qwen3.5 (5 seeds)."""
    pairs = []
    for qwen3_label, qwen35_label, display in CROSS_GEN_PAIRS:
        # Qwen3 pilot: single run, no seeds
        qwen3_sims = extract_sims(qwen3_pilot.get(qwen3_label, []))
        qwen3_final = qwen3_sims[-1] if qwen3_sims else None

        # Qwen3.5: 5 seeds
        cond = condition_stats(exp1_models, f"{qwen35_label}_s{{seed}}")
        if not cond["finals"]:
            qwen35_mean = None
            qwen35_ci = None
        else:
            qwen35_mean = cond["mean"]
            qwen35_ci = (cond["ci_low"], cond["ci_high"])

        qwen3_cls = classify_chain(qwen3_pilot.get(qwen3_label, []))
        clean = cond["clean"]
        pairs.append({
            "display": display,
            "qwen3_label": qwen3_label,
            "qwen35_label": qwen35_label,
            "qwen3_final": qwen3_final,
            "qwen3_status": qwen3_cls["status"],
            "qwen35_mean": qwen35_mean,
            "qwen35_ci": qwen35_ci,
            "qwen35_all": cond["all"],
            "qwen35_clean": clean,
            "qwen35_clean_mean": clean["mean"],
            "qwen35_clean_ci": (clean["ci_low"], clean["ci_high"]),
            "n_clean": cond["n_clean"],
            "n_runaway": cond["n_runaway"],
            "n_total": cond["n_total"],
            "runaway_rate": cond["runaway_rate"],
            "runaway_ci": cond["runaway_ci"],
        })

    return {"pairs": pairs}


# =============================================================================
# 4. FIGURE FUNCTIONS
# =============================================================================

def plot_fig1_size_quant_lines(models: dict, stats: dict) -> None:
    """Fig 1: 18-model line plot (spaghetti) with 95% CI bands."""
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
    ax.set_title("Qwen3.5 Semantic Drift: 18 Models, Spaghetti Recipe (5 seeds, 95% CI)")
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    tier_handles = [mpatches.Patch(color=TIER_COLORS[t], label=t)
                    for t in ["Stable", "Moderate", "Degraded", "Catastrophic"]]
    tier_legend = ax.legend(handles=tier_handles, fontsize=8, loc="upper right", title="Tier")
    ax.add_artist(tier_legend)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6, loc="lower left", ncol=2,
              title="Model (final mean)")

    save_figure(fig, "fig1_size_quant_lines.png")


def plot_fig2_boxplot(spag_stats: dict, las_stats: dict) -> None:
    """Fig 2: Side-by-side boxplot for spaghetti and lasagna."""
    # Sort by spaghetti mean
    sorted_labels = sorted(
        [l for l in spag_stats["per_model"]],
        key=lambda l: spag_stats["per_model"][l]["mean"], reverse=True)

    n = len(sorted_labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, stats, title in [(ax1, spag_stats, "Spaghetti (3 steps)"),
                              (ax2, las_stats, "Lasagna (21 steps)")]:
        data = []
        colors = []
        labels_list = []
        for label in sorted_labels:
            info = stats["per_model"].get(label)
            if info and info["finals"]:
                data.append(info["finals"])
                colors.append(TIER_COLORS[info["tier"]])
                labels_list.append(EXP1_DISPLAY.get(label, label))
            else:
                data.append([])
                colors.append("#cccccc")
                labels_list.append(EXP1_DISPLAY.get(label, label))

        bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, finals in enumerate(data):
            if finals:
                jitter = RNG.normal(0, 0.08, len(finals))
                ax.scatter(np.full(len(finals), i + 1) + jitter, finals,
                           color="black", s=12, alpha=0.6, zorder=5)

        ax.set_xticks(range(1, len(labels_list) + 1))
        ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)

    ax1.set_ylabel("Final Cosine Similarity (iteration 30)")
    fig.suptitle("Qwen3.5 Final Similarity Distribution (5 seeds per model)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "fig2_size_quant_boxplot.png")


def plot_fig3_quant_heatmap(spag_stats: dict, las_stats: dict) -> None:
    """Fig 3: Size x Quant heatmap, side by side for both recipes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, stats, title in [(ax1, spag_stats, "Spaghetti"),
                              (ax2, las_stats, "Lasagna")]:
        grid = np.full((len(SIZE_ORDER), len(QUANT_ORDER)), np.nan)

        for label, info in stats["per_model"].items():
            meta = EXP1_META.get(label)
            if not meta:
                continue
            _, size, quant = meta
            if size in SIZE_ORDER and quant in QUANT_ORDER:
                row = SIZE_ORDER.index(size)
                col = QUANT_ORDER.index(quant)
                grid[row, col] = info["mean"]

        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(QUANT_ORDER)))
        ax.set_xticklabels(["BF16", "Q8_0", "Q4_K_M"], fontsize=10)
        ax.set_yticks(range(len(SIZE_ORDER)))
        ax.set_yticklabels(SIZE_ORDER, fontsize=10)
        ax.set_xlabel("Quantization")
        ax.set_ylabel("Model Size")
        ax.set_title(title)

        for i in range(len(SIZE_ORDER)):
            for j in range(len(QUANT_ORDER)):
                val = grid[i, j]
                if not np.isnan(val):
                    text_color = "black" if val > 0.4 else "white"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold", color=text_color)
                else:
                    ax.text(j, i, "\u2014", ha="center", va="center",
                            fontsize=10, color="#cccccc")

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Mean Final Cosine Similarity")
    fig.suptitle("Qwen3.5: Size x Quantization Heatmap", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "fig3_quant_heatmap.png")


def plot_fig4_quant_comparison(quant_stats: dict, exp1_models: dict,
                               exp3a_models: dict) -> None:
    """Fig 4: four-way GGUF / AWQ / NVFP4 / GPTQ comparison for the 35B MoE."""
    sources = {"exp1": exp1_models, "exp3a": exp3a_models}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, recipe in zip(axes, ["spaghetti", "lasagna"]):
        prefix = "" if recipe == "spaghetti" else "lasagna_"
        info = quant_stats.get(recipe) or {}
        methods = info.get("methods", {})

        for key, display, src, label in QUANT_METHODS:
            curves = get_seed_curves(sources[src], f"{prefix}{label}_s{{seed}}")
            if not curves:
                continue
            mean_c, lo_c, hi_c = bootstrap_ci_curves(curves)
            xs = np.arange(1, len(mean_c) + 1)
            color = QUANT_COLORS[key]
            n = methods.get(key, {}).get("n", len(curves))
            plain = display.replace("\\_", "_")
            ax.plot(xs, mean_c, color=color, linewidth=2,
                    label=f"{plain} (n={n})")
            ax.fill_between(xs, lo_c, hi_c, color=color, alpha=0.12)

        delta_lines = []
        for key in ("awq", "nvfp4", "gptq"):
            d = info.get("deltas", {}).get(key)
            if not d:
                continue
            sign = "+" if d["delta_mean"] >= 0 else ""
            name = {"awq": "AWQ", "nvfp4": "NVFP4", "gptq": "GPTQ"}[key]
            delta_lines.append(
                f"{name}$-$GGUF $\\Delta$ = {sign}{d['delta_mean']:.3f} "
                f"[{d['delta_ci'][0]:.3f}, {d['delta_ci'][1]:.3f}]")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine Similarity")
        title = recipe.capitalize()
        if delta_lines:
            title += "\n" + "\n".join(delta_lines)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0.5, ITERATIONS + 0.5)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Qwen3.5 35B MoE: GGUF vs AWQ vs NVFP4 vs GPTQ-Int4", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "fig4_gguf_vs_awq.png")


def plot_fig5_temperature(temp_stats: dict) -> None:
    """Fig 5: Temperature x model interaction, side by side for both recipes."""
    model_colors = {
        "qwen3.5:2b": "#f44336",
        "qwen3.5:9b": "#ff9800",
        "qwen3.5:27b": "#4caf50",
        "qwen3.5:35b": "#2196f3",
        TEMP_VLLM_MODEL: "#9c27b0",
        "qwen3.5:35b-nvfp4": "#009688",
        "qwen3.5:35b-gptq": "#795548",
        "qwen3.5:9b-awq": "#e91e63",
        "nemo:30b-nvfp4": "#607d8b",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, recipe, title in [(ax1, "spaghetti", "Spaghetti"),
                               (ax2, "lasagna", "Lasagna")]:
        for model_key, info in temp_stats["per_model"].items():
            ts = info["temps"]
            temps_with_data = sorted(t for t in ts if ts[t].get(recipe) is not None)
            if not temps_with_data:
                continue

            means = [ts[t][recipe]["mean"] for t in temps_with_data]
            ci_lo = [ts[t][recipe]["ci_low"] for t in temps_with_data]
            ci_hi = [ts[t][recipe]["ci_high"] for t in temps_with_data]
            errors_lo = [m - lo for m, lo in zip(means, ci_lo)]
            errors_hi = [hi - m for m, hi in zip(means, ci_hi)]

            color = model_colors.get(model_key, "#888888")
            ax.errorbar(temps_with_data, means, yerr=[errors_lo, errors_hi],
                        color=color, linewidth=2, marker="o", markersize=6,
                        capsize=4, label=info["display"])

        ax.set_xlabel("Temperature")
        ax.set_title(title)
        ax.set_xticks(TEMPERATURES)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower left")

    ax1.set_ylabel("Mean Final Cosine Similarity")
    fig.suptitle("Temperature Sensitivity: Qwen3.5 Models", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "fig5_temperature.png")


def plot_fig6_complexity(spag_stats: dict, las_stats: dict) -> None:
    """Fig 6: Spaghetti vs lasagna grouped bars for all 18 models."""
    # Sort by spaghetti mean
    sorted_labels = sorted(
        [l for l in spag_stats["per_model"]],
        key=lambda l: spag_stats["per_model"][l]["mean"], reverse=True)

    names = [EXP1_DISPLAY.get(l, l) for l in sorted_labels]
    spag_means = [spag_stats["per_model"][l]["mean"] for l in sorted_labels]
    las_means = [las_stats["per_model"].get(l, {}).get("mean", 0) for l in sorted_labels]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, spag_means, width, color="#2196f3",
           edgecolor="white", linewidth=0.5, label="Spaghetti (3 steps)")
    ax.bar(x + width / 2, las_means, width, color="#ff9800",
           edgecolor="white", linewidth=0.5, label="Lasagna (21 steps)")

    ax.set_xlabel("Model (sorted by spaghetti mean)")
    ax.set_ylabel("Mean Final Cosine Similarity (5 seeds)")
    ax.set_title("Complexity Threshold: Spaghetti vs Lasagna")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    save_figure(fig, "fig6_complexity_threshold.png")


def plot_fig7_sysprompt_hero(sysprompt_stats: dict, exp4_models: dict) -> None:
    """Fig 7: 27B lasagna hero line plot, 20 prompts, category-colored."""
    fig, ax = plt.subplots(figsize=(14, 8))

    plot_items = []
    for p in PROMPTS:
        slug = _slug(p["name"])
        pattern = f"qwen3.5:27b_lasagna_{p['id']:02d}_{slug}_s{{seed}}"
        curves = get_seed_curves(exp4_models, pattern)
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
    ax.set_title("System Prompt Ablation \u2014 Qwen3.5 27B, Lasagna Recipe (20 prompts, 5 seeds)")
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    cat_handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    cat_legend = ax.legend(handles=cat_handles, fontsize=8, loc="upper right", title="Category")
    ax.add_artist(cat_legend)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6, loc="lower left", ncol=2,
              title="System Prompt (final mean)")

    fig.tight_layout()
    save_figure(fig, "fig7_sysprompt_hero.png")


def plot_fig8_sysprompt_cross_model(sysprompt_stats: dict) -> None:
    """Fig 8: slope chart of prompt ranks across every model (lasagna)."""
    available = [f"{m}_lasagna" for m in RQ4_MODEL_ORDER
                 if f"{m}_lasagna" in sysprompt_stats["conditions"]]
    if len(available) < 2:
        print("  SKIP fig8 (insufficient sysprompt data)")
        return

    all_ranks = {}
    for cname in available:
        stats = [s for s in sysprompt_stats["conditions"][cname]["stats"]
                 if s["n_seeds"] > 0]
        sorted_stats = sorted(stats, key=lambda s: s["mean"], reverse=True)
        all_ranks[cname] = {s["name"]: i for i, s in enumerate(sorted_stats)}

    n = len(PROMPTS)
    fig, ax = plt.subplots(figsize=(15, 11))

    n_cols = len(available)
    x_positions = np.linspace(0.18, 0.82, n_cols)

    for p in PROMPTS:
        name = p["name"]
        color = CATEGORY_COLORS.get(p["category"], "#888888")
        points_x, points_y = [], []
        for x_pos, cname in zip(x_positions, available):
            if name in all_ranks[cname]:
                points_x.append(x_pos)
                points_y.append(all_ranks[cname][name])

        if len(points_x) >= 2:
            ax.plot(points_x, points_y, color=color, linewidth=1.3, alpha=0.55)
        for px, py in zip(points_x, points_y):
            ax.plot(px, py, "o", color=color, markersize=5, zorder=5)

        if points_x:
            stats_left = sysprompt_stats["conditions"][available[0]]["stats"]
            mean_left = next((s["mean"] for s in stats_left if s["name"] == name), 0)
            ax.text(points_x[0] - 0.02, points_y[0], f"{name} ({mean_left:.3f})",
                    ha="right", va="center", fontsize=6.5, color=color)
            if len(points_x) > 1:
                stats_right = sysprompt_stats["conditions"][available[-1]]["stats"]
                mean_right = next((s["mean"] for s in stats_right if s["name"] == name), 0)
                ax.text(points_x[-1] + 0.02, points_y[-1], f"({mean_right:.3f})",
                        ha="left", va="center", fontsize=6.5, color=color)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([RQ4_DISPLAY.get(c.replace("_lasagna", ""), c)
                        for c in available], fontsize=9, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{i+1}" for i in range(n)], fontsize=8)
    ax.set_ylabel("Rank (1 = best)")

    # Annotate Spearman rho for adjacent column pairs
    ann = []
    for a, b in zip(available, available[1:]):
        st = sysprompt_stats["spearman_pairs"].get(f"{a}_vs_{b}")
        if st:
            ann.append(f"{RQ4_DISPLAY.get(a.replace('_lasagna',''),a)}"
                       f"\u2013{RQ4_DISPLAY.get(b.replace('_lasagna',''),b)}: "
                       f"\u03c1={st['rho']:.2f}")
    title = "System Prompt Rankings Across Models (Lasagna Recipe)"
    if ann:
        title += "\nAdjacent-pair Spearman: " + "  |  ".join(ann)
    ax.set_title(title, fontsize=10)

    cat_handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    ax.legend(handles=cat_handles, fontsize=8, loc="lower center", ncol=6)

    fig.tight_layout()
    save_figure(fig, "fig8_sysprompt_cross_model.png")


def plot_fig9_category_heatmap(sysprompt_stats: dict) -> None:
    """Fig 9: prompt categories x conditions heatmap."""
    cat_means = sysprompt_stats["category_means"]
    cond_order = sysprompt_stats["cond_order"]
    cond_labels = []
    for c in cond_order:
        model, recipe = c.rsplit("_", 1)
        cond_labels.append(RQ4_DISPLAY.get(model, model).replace(" (control)", "")
                           + "\n" + ("Spag" if recipe == "spaghetti" else "Las"))

    grid = np.array([cat_means[c] for c in CATEGORY_ORDER])
    grid_masked = np.ma.masked_where(grid == 0, grid)

    fig, ax = plt.subplots(figsize=(1.4 * len(cond_labels) + 3, 5))
    vmin = float(grid_masked.min()) if grid_masked.count() > 0 else 0.6
    vmax = float(grid_masked.max()) if grid_masked.count() > 0 else 1.0
    im = ax.imshow(grid_masked, cmap="YlGnBu", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, fontsize=7)
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels(CATEGORY_ORDER, fontsize=9)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Prompt Category")
    ax.set_title("Mean Final Similarity by Category and Condition (Qwen3.5)")

    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(cond_labels)):
            val = grid[i, j]
            if val > 0:
                text_color = "white" if val < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=text_color)
            else:
                ax.text(j, i, "\u2014", ha="center", va="center", fontsize=8, color="#cccccc")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Final Cosine Similarity")
    fig.tight_layout()
    save_figure(fig, "fig9_category_heatmap.png")


def plot_fig10_cross_gen(cross_gen: dict) -> None:
    """Fig 10: Qwen3 vs Qwen3.5 paired bar chart."""
    pairs = cross_gen["pairs"]
    valid_pairs = [p for p in pairs if p["qwen3_final"] is not None and p["qwen35_mean"] is not None]

    if not valid_pairs:
        print("  SKIP fig10 (no valid cross-gen pairs)")
        return

    n = len(valid_pairs)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    qwen3_vals = [p["qwen3_final"] for p in valid_pairs]
    qwen35_vals = [p["qwen35_mean"] for p in valid_pairs]
    qwen35_errs = []
    for p in valid_pairs:
        if p["qwen35_ci"]:
            qwen35_errs.append([p["qwen35_mean"] - p["qwen35_ci"][0],
                                p["qwen35_ci"][1] - p["qwen35_mean"]])
        else:
            qwen35_errs.append([0, 0])
    qwen35_errs = list(zip(*qwen35_errs))

    ax.bar(x - width / 2, qwen3_vals, width, color="#ff9800", edgecolor="white",
           linewidth=0.5, label="Qwen3 (pilot, Q4_K_M)")
    ax.bar(x + width / 2, qwen35_vals, width, yerr=qwen35_errs, capsize=3,
           color="#2196f3", edgecolor="white", linewidth=0.5,
           label="Qwen3.5 (5 seeds, matched quant)")

    ax.set_xlabel("Size Pair")
    ax.set_ylabel("Final Cosine Similarity")
    ax.set_title("Cross-Generation Comparison: Qwen3 vs Qwen3.5")
    ax.set_xticks(x)
    ax.set_xticklabels([p["display"] for p in valid_pairs], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")

    # Annotate deltas
    for i, p in enumerate(valid_pairs):
        delta = p["qwen35_mean"] - p["qwen3_final"]
        sign = "+" if delta >= 0 else ""
        color = "#4caf50" if delta >= 0 else "#f44336"
        y_max = max(p["qwen3_final"], p["qwen35_mean"])
        ax.text(i, y_max + 0.03, f"{sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=color)

    fig.tight_layout()
    save_figure(fig, "fig10_cross_generation.png")


def plot_figA1_lasagna_lines(models: dict, las_stats: dict) -> None:
    """Fig A1: Same as Fig 1 but lasagna recipe."""
    fig, ax = plt.subplots(figsize=(12, 7))

    sorted_models = sorted(las_stats["per_model"].items(), key=lambda x: x[1]["mean"], reverse=True)

    for label, info in sorted_models:
        curves = get_seed_curves(models, f"lasagna_{label}_s{{seed}}")
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
    ax.set_title("Qwen3.5 Semantic Drift: 18 Models, Lasagna Recipe (5 seeds, 95% CI)")
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    tier_handles = [mpatches.Patch(color=TIER_COLORS[t], label=t)
                    for t in ["Stable", "Moderate", "Degraded", "Catastrophic"]]
    tier_legend = ax.legend(handles=tier_handles, fontsize=8, loc="upper right", title="Tier")
    ax.add_artist(tier_legend)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6, loc="lower left", ncol=2,
              title="Model (final mean)")

    save_figure(fig, "figA1_lasagna_lines.png")


def plot_figA2_seed_variance(models: dict) -> None:
    """Fig A2: 6 most variable models on spaghetti, individual seed trajectories."""
    seed_colors = ["#2196f3", "#4caf50", "#ff9800", "#f44336", "#9c27b0"]

    model_spreads = []
    for label in EXP1_MODELS:
        curves = get_seed_curves(models, f"{label}_s{{seed}}")
        finals = [float(c[-1]) for c in curves if len(c) > 0]
        if finals:
            spread = max(finals) - min(finals)
            model_spreads.append((spread, label, curves))

    model_spreads.sort(key=lambda x: x[0], reverse=True)
    show = model_spreads[:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)

    for idx, (spread, label, curves) in enumerate(show):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        for i, curve in enumerate(curves):
            xs = np.arange(1, len(curve) + 1)
            ax.plot(xs, curve, color=seed_colors[i % len(seed_colors)],
                    linewidth=1.2, alpha=0.7, marker="o", markersize=1.5,
                    label=f"seed {SEEDS[i]}")

        finals = [float(c[-1]) for c in curves if len(c) > 0]
        display = EXP1_DISPLAY.get(label, label)
        ax.set_title(f"{display}\nspread={spread:.3f} [{min(finals):.2f}\u2013{max(finals):.2f}]",
                     fontsize=9, fontweight="bold")
        ax.set_xlim(0.5, ITERATIONS + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 10, 20, 30])
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower left")

    fig.suptitle("Seed Variance: 6 Most Variable Qwen3.5 Models (Spaghetti)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "figA2_seed_variance.png")


def plot_figA3_temperature_multiples(temp_stats: dict) -> None:
    """Fig A3: Per-model small multiples showing all 5 temperature curves."""
    temp_colors = {0.0: "#2196f3", 0.3: "#4caf50", 0.5: "#ff9800", 0.7: "#f44336", 1.0: "#9c27b0"}

    models_with_data = [(k, v) for k, v in temp_stats["per_model"].items()
                        if len(v["temps"]) >= 3]
    n = len(models_with_data)
    if n == 0:
        print("  SKIP figA3 (no temperature data)")
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (model_key, info) in enumerate(models_with_data):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        for temp in TEMPERATURES:
            if temp not in info["temps"]:
                continue
            for recipe, linestyle in [("spaghetti", "-"), ("lasagna", "--")]:
                stat = info["temps"][temp].get(recipe)
                if not stat:
                    continue
                color = temp_colors.get(temp, "#888888")
                ax.plot([], [], color=color, linestyle=linestyle)  # for legend
                # Just plot the mean as a single point (we don't have per-iter curves here)
                ax.scatter([temp], [stat["mean"]], color=color, s=40, zorder=5,
                           marker="o" if recipe == "spaghetti" else "s")

        # Instead, plot as temp vs mean scatter
        for recipe, marker, ls in [("spaghetti", "o", "-"), ("lasagna", "s", "--")]:
            temps_r = []
            means_r = []
            for t in TEMPERATURES:
                if t in info["temps"] and info["temps"][t].get(recipe):
                    temps_r.append(t)
                    means_r.append(info["temps"][t][recipe]["mean"])
            if temps_r:
                ax.plot(temps_r, means_r, color="#333333", linestyle=ls, marker=marker,
                        markersize=5, linewidth=1.2, label=recipe.capitalize())

        ax.set_title(info["display"], fontsize=10, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(TEMPERATURES)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower left")

    # Hide unused axes
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Temperature Sensitivity per Model (Qwen3.5)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "figA3_temperature_multiples.png")


# =============================================================================
# 5. LATEX TABLE EXPORT
# =============================================================================

def export_table1_size_quant(spag_stats: dict, las_stats: dict) -> None:
    """Export Table 1: Size x Quant summary, clean-chain means as the primary
    number, with per-cell completed-chain counts and runaway-thinking rates.
    All-chain means are retained in all_stats.json only."""
    def sort_key(item):
        info = item[1]
        cm = info["clean"]["mean"]
        return cm if cm is not None else -1.0

    sorted_models = sorted(spag_stats["per_model"].items(),
                           key=sort_key, reverse=True)

    L = []
    A = L.append
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\caption{Qwen3.5 drift by size and quantization. Mean final cosine")
    A(r"  similarity at iteration 30 over \emph{clean} chains only (chains with")
    A(r"  no empty, runaway-thinking generation), with 95\% bootstrap CI,")
    A(r"  sorted by clean spaghetti mean descending. $n_c$ is the number of")
    A(r"  clean 30-iteration chains behind each mean; ``RA'' is the")
    A(r"  runaway-thinking rate, i.e.\ the fraction of completed chains that")
    A(r"  contained at least one empty generation (runaway chains are excluded")
    A(r"  from the means). All-chain means are reported in the supplementary")
    A(r"  \texttt{all\_stats.json}.}")
    A(r"\label{tab:size_quant}")
    A(r"\footnotesize")
    A(r"\resizebox{\textwidth}{!}{%")
    A(r"\begin{tabular}{llcccccccl}")
    A(r"\toprule")
    A(r"Size & Quant & \multicolumn{4}{c}{Spaghetti} & \multicolumn{3}{c}{Lasagna} & Tier \\")
    A(r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}")
    A(r" & & Mean & 95\% CI & $n_c$ & RA & Mean & $n_c$ & RA & \\")
    A(r"\midrule")

    prev_tier = None
    for label, info in sorted_models:
        tier = info.get("tier_clean", info["tier"])
        if prev_tier is not None and tier != prev_tier:
            A(r"\midrule")
        prev_tier = tier

        meta = EXP1_META.get(label, (label, "", ""))
        size = escape_latex(meta[1])
        quant = escape_latex(meta[2])
        c = info["clean"]
        mean_s = _fmt_mean(c["mean"])
        ci_s = ("---" if c["mean"] is None
                else f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
        ra_s = escape_latex(_fmt_rate(info))

        las_info = las_stats["per_model"].get(label)
        if las_info:
            mean_l = _fmt_mean(las_info["clean"]["mean"])
            n_l = str(las_info["n_clean"])
            ra_l = escape_latex(_fmt_rate(las_info))
        else:
            mean_l, n_l, ra_l = "---", "0", "---"

        A(f"{size} & {quant} & {mean_s} & {ci_s} & {info['n_clean']} & {ra_s} & "
          f"{mean_l} & {n_l} & {ra_l} & {tier} \\\\")

    A(r"\bottomrule")
    A(r"\end{tabular}}")
    A(r"\end{table}")

    write_table("table1_size_quant.tex", "\n".join(L) + "\n")


def export_table2_sysprompt(sysprompt_stats: dict) -> None:
    """Export Table 2: per-prompt lasagna clean-chain means for every model."""
    available = [f"{m}_lasagna" for m in RQ4_MODEL_ORDER
                 if f"{m}_lasagna" in sysprompt_stats["conditions"]]
    if not available:
        print("  SKIP table2 (no sysprompt data)")
        return

    headers = {c: RQ4_DISPLAY.get(c.replace("_lasagna", ""), c).replace(" (control)", "")
               for c in available}

    def entry_for(cname, name):
        st = sysprompt_stats["conditions"][cname]["stats"]
        return next((s for s in st if s["name"] == name), None)

    prompt_overall = {}
    for p in PROMPTS:
        vals = []
        for cname in available:
            e = entry_for(cname, p["name"])
            if e and e["n_clean"] > 0:
                vals.append(e["clean"]["mean"])
        prompt_overall[p["name"]] = float(np.mean(vals)) if vals else 0.0

    sorted_prompts = sorted(PROMPTS, key=lambda p: prompt_overall[p["name"]], reverse=True)
    n_cols = len(available)

    L = []
    A = L.append
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\caption{System prompt effects on the lasagna recipe across all tested")
    A(r"  Qwen3.5 configurations plus the out-of-family Nemotron control. Mean")
    A(r"  final cosine similarity over the \emph{clean} chains for that cell")
    A(r"  (chains containing no empty, runaway-thinking generation; up to 5 per")
    A(r"  cell). A superscript gives $n_c$ when fewer than 5 clean chains")
    A(r"  survive; ``---'' marks a cell with no clean chain at all. The last two")
    A(r"  rows give the smallest per-cell $n_c$ and the configuration-wide")
    A(r"  runaway-thinking rate. Sorted by the mean across configurations.}")
    A(r"\label{tab:sysprompt}")
    A(r"\scriptsize")
    A(r"\resizebox{\textwidth}{!}{%")
    A(r"\begin{tabular}{ll" + "c" * n_cols + "c}")
    A(r"\toprule")
    header = "Prompt & Category"
    for cname in available:
        header += f" & {escape_latex(headers[cname])}"
    header += r" & Mean \\"
    A(header)
    A(r"\midrule")

    for p in sorted_prompts:
        row = f"{escape_latex(p['name'])} & {p['category']}"
        for cname in available:
            e = entry_for(cname, p["name"])
            if e and e["n_clean"] > 0:
                sup = f"$^{{{e['n_clean']}}}$" if e["n_clean"] < 5 else ""
                row += f" & {e['clean']['mean']:.3f}{sup}"
            else:
                row += " & ---"
        ov = prompt_overall[p["name"]]
        row += f" & {ov:.3f}" if ov > 0 else " & ---"
        row += r" \\"
        A(row)

    A(r"\midrule")
    row = r"\multicolumn{2}{l}{Minimum $n_c$}"
    for cname in available:
        ns = [e["n_clean"] for e in sysprompt_stats["conditions"][cname]["stats"]]
        row += f" & {min(ns) if ns else 0}"
    row += r" & \\"
    A(row)
    row = r"\multicolumn{2}{l}{Runaway rate}"
    for cname in available:
        ra = sysprompt_stats["runaway_by_condition"].get(cname)
        row += (" & ---" if not ra or not ra["n_complete"]
                else f" & {ra['runaway_rate']:.2f}")
    row += r" & \\"
    A(row)

    A(r"\bottomrule")
    A(r"\end{tabular}}")
    A(r"\end{table}")

    write_table("table2_sysprompt.tex", "\n".join(L) + "\n")


def export_table3_rq4(sysprompt_stats: dict) -> None:
    """Export Table 3: RQ4 per-model spread / best / worst and Spearman matrix."""
    available = [m for m in RQ4_MODEL_ORDER
                 if f"{m}_lasagna" in sysprompt_stats["conditions"]]
    if not available:
        print("  SKIP table3 (no sysprompt data)")
        return

    ext = sysprompt_stats["extremes_clean"] or sysprompt_stats["extremes"]
    spr = (sysprompt_stats["spreads_clean"] if sysprompt_stats["extremes_clean"]
           else sysprompt_stats["spreads"])
    ra_by_cond = sysprompt_stats["runaway_by_condition"]

    L = []
    A = L.append
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\caption{RQ4 summary (lasagna recipe), computed over \emph{clean}")
    A(r"  chains only. Spread is the difference between the best- and")
    A(r"  worst-performing system prompt for that configuration;")
    A(r"  $n_{\min}$ is the smallest number of clean chains behind any")
    A(r"  prompt cell; ``RA'' is the configuration-wide runaway-thinking rate")
    A(r"  over all 20 prompts $\times$ 5 seeds.}")
    A(r"\label{tab:rq4_spread}")
    A(r"\small")
    A(r"\resizebox{\textwidth}{!}{%")
    A(r"\begin{tabular}{lccllcc}")
    A(r"\toprule")
    A(r"Configuration & Mean & Spread & Best prompt & Worst prompt & $n_{\min}$ & RA \\")
    A(r"\midrule")
    for m in available:
        c = f"{m}_lasagna"
        e = ext.get(c)
        if not e:
            continue
        ra = ra_by_cond.get(c)
        ra_s = escape_latex(
            "---" if not ra or not ra["n_complete"]
            else f"{ra['runaway_rate']:.2f} ({ra['n_runaway']}/{ra['n_complete']})")
        A(f"{escape_latex(RQ4_DISPLAY.get(m, m))} & {e['mean_of_prompts']:.3f} & "
          f"{spr.get(c, 0):.3f} & "
          f"{escape_latex(e['best']['name'])} ({e['best']['mean']:.3f}) & "
          f"{escape_latex(e['worst']['name'])} ({e['worst']['mean']:.3f}) & "
          f"{e['min_n']} & {ra_s} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}}")
    A(r"")
    A(r"\vspace{1em}")
    A(r"")
    A(r"\caption*{Pairwise Spearman $\rho$ between per-prompt mean rankings")
    A(r"  (20 prompts, lasagna), with 95\% bootstrap CIs ($B=10{,}000$).}")
    A(r"\small")
    A(r"\begin{tabular}{llcc}")
    A(r"\toprule")
    A(r"Configuration A & Configuration B & $\rho$ & 95\% CI \\")
    A(r"\midrule")
    for key, st in sysprompt_stats["spearman_pairs"].items():
        a, b = key.split("_vs_")
        an = RQ4_DISPLAY.get(a.replace("_lasagna", ""), a)
        bn = RQ4_DISPLAY.get(b.replace("_lasagna", ""), b)
        A(f"{escape_latex(an)} & {escape_latex(bn)} & {st['rho']:.3f} & "
          f"[{st['ci'][0]:.3f}, {st['ci'][1]:.3f}] \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")

    write_table("table3_rq4.tex", "\n".join(L) + "\n")


RUNAWAY_SYSPROMPT_SETS = [
    ("35b_nvfp4", "35B MoE NVFP4"),
    ("27b", "27B GGUF Q4\\_K\\_M"),
]


def collect_runaway_rq1(spag_stats: dict, las_stats: dict) -> list[dict]:
    """Per-model runaway rates for the RQ1 (size x quant) set, both recipes."""
    rows = []
    for label in EXP1_MODELS:
        sp = spag_stats["per_model"].get(label)
        la = las_stats["per_model"].get(label)
        if not sp and not la:
            continue
        meta = EXP1_META.get(label, (label, "", ""))
        rows.append({
            "label": label, "display": EXP1_DISPLAY.get(label, label),
            "size": meta[1], "quant": meta[2],
            "spaghetti": sp, "lasagna": la,
        })
    return rows


def collect_runaway_sysprompt(sysprompt_stats: dict) -> dict:
    """Per-prompt runaway rates for the sysprompt sets named above."""
    out = {}
    for short_key, display in RUNAWAY_SYSPROMPT_SETS:
        block = {}
        for recipe in ("spaghetti", "lasagna"):
            cname = f"{short_key}_{recipe}"
            cond = sysprompt_stats["conditions"].get(cname)
            if not cond:
                continue
            block[recipe] = {s["name"]: s for s in cond["stats"]}
        if block:
            out[short_key] = {"display": display, "recipes": block}
    return out


def export_table4_runaway(spag_stats: dict, las_stats: dict,
                          sysprompt_stats: dict) -> None:
    """Export Table 4: runaway-thinking rates by model/quant and by prompt."""
    rq1_rows = collect_runaway_rq1(spag_stats, las_stats)
    sysp = collect_runaway_sysprompt(sysprompt_stats)

    L = []
    A = L.append
    A(r"\begin{table}[p]")
    A(r"\centering")
    A(r"\caption{Runaway-thinking rates. A chain is \emph{runaway} when at")
    A(r"  least one of its 30 iterations returned an empty assistant message")
    A(r"  after a full-length generation, i.e.\ the entire token budget was")
    A(r"  consumed by reasoning; such chains never recover, because the same")
    A(r"  input is re-fed at the next iteration. Rate = runaway chains /")
    A(r"  completed chains, with a Wilson 95\% interval. Top block: RQ1 size")
    A(r"  $\times$ quantization set; per-prompt rates are in Table~\ref{tab:runaway_prompts}.}")
    A(r"\label{tab:runaway}")
    A(r"\footnotesize")
    A(r"\begin{tabular}{llcccc}")
    A(r"\toprule")
    A(r"Size & Quant & \multicolumn{2}{c}{Spaghetti} & \multicolumn{2}{c}{Lasagna} \\")
    A(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    A(r" & & Rate & 95\% CI & Rate & 95\% CI \\")
    A(r"\midrule")
    for r in rq1_rows:
        cells = []
        for recipe in ("spaghetti", "lasagna"):
            st = r[recipe]
            if not st or not (st["n_clean"] + st["n_runaway"]):
                cells += ["---", "---"]
            else:
                cells.append(escape_latex(_fmt_rate(st)))
                cells.append(f"[{st['runaway_ci'][0]:.2f}, {st['runaway_ci'][1]:.2f}]")
        A(f"{escape_latex(r['size'])} & {escape_latex(r['quant'])} & "
          + " & ".join(cells) + r" \\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"")

    A(r"\end{table}")
    if sysp:
        A(r"")
        A(r"\begin{table}[p]")
        A(r"\centering")
        A(r"\caption{Runaway-thinking rate per system prompt (up to 5 chains per")
        A(r"  cell; rate = runaway chains / completed chains).}")
        A(r"\label{tab:runaway_prompts}")
        A(r"\footnotesize")
        A(r"\resizebox{\textwidth}{!}{%")
        n_cols = 2 * len(sysp)
        A(r"\begin{tabular}{l" + "c" * n_cols + "}")
        A(r"\toprule")
        head = "System prompt"
        rule = []
        col = 2
        for short_key, block in sysp.items():
            head += r" & \multicolumn{2}{c}{" + block["display"] + "}"
            rule.append(rf"\cmidrule(lr){{{col}-{col+1}}}")
            col += 2
        A(head + r" \\")
        A("".join(rule))
        A("Prompt" + " & Spag. & Las." * len(sysp) + r" \\")
        A(r"\midrule")
        for p in PROMPTS:
            row = escape_latex(p["name"])
            for short_key, block in sysp.items():
                for recipe in ("spaghetti", "lasagna"):
                    st = block["recipes"].get(recipe, {}).get(p["name"])
                    if not st or not (st["n_clean"] + st["n_runaway"]):
                        row += " & ---"
                    else:
                        row += f" & {escape_latex(_fmt_rate(st))}"
            A(row + r" \\")
        A(r"\bottomrule")
        A(r"\end{tabular}}")

    A(r"\end{table}")
    write_table("table4_runaway.tex", "\n".join(L) + "\n")


def plot_fig11_runaway_rates(spag_stats: dict, las_stats: dict,
                             sysprompt_stats: dict) -> None:
    """Fig 11: runaway rates by size/quant (both recipes) and by system prompt."""
    rq1_rows = collect_runaway_rq1(spag_stats, las_stats)
    if not rq1_rows:
        print("  SKIP fig11 (no RQ1 data)")
        return

    quant_colors = {"bf16": "#2196f3", "q8_0": "#ff9800", "q4_K_M": "#f44336"}
    sizes = [s for s in SIZE_ORDER
             if any(r["size"] == s for r in rq1_rows)]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.42, wspace=0.18)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1], sharey=ax_s)
    ax_p = fig.add_subplot(gs[1, :])

    width = 0.26
    for ax, recipe, title in [(ax_s, "spaghetti", "Spaghetti (3 steps)"),
                              (ax_l, "lasagna", "Lasagna (21 steps)")]:
        x = np.arange(len(sizes))
        for qi, quant in enumerate(QUANT_ORDER):
            vals, errs_lo, errs_hi, present = [], [], [], []
            for si, size in enumerate(sizes):
                row = next((r for r in rq1_rows
                            if r["size"] == size and r["quant"] == quant), None)
                st = row[recipe] if row else None
                if not st or not (st["n_clean"] + st["n_runaway"]):
                    vals.append(np.nan)
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
                    present.append(None)
                    continue
                rate = st["runaway_rate"]
                vals.append(rate)
                errs_lo.append(max(0.0, rate - st["runaway_ci"][0]))
                errs_hi.append(max(0.0, st["runaway_ci"][1] - rate))
                present.append(st)
            offs = (qi - 1) * width
            ax.bar(x + offs, vals, width, color=quant_colors[quant],
                   edgecolor="white", linewidth=0.5,
                   yerr=[errs_lo, errs_hi], capsize=2, ecolor="#555555",
                   label=quant if ax is ax_s else None)
            for xi, (v, st) in enumerate(zip(vals, present)):
                if st is None or np.isnan(v) or v <= 0:
                    continue
                n = st["n_clean"] + st["n_runaway"]
                ax.text(x[xi] + offs, st["runaway_ci"][1] + 0.02,
                        f"{st['n_runaway']}/{n}",
                        ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Model size")
    ax_s.set_ylabel("Runaway-thinking rate")
    ax_s.legend(fontsize=8, title="Quantization", loc="upper center",
                ncol=3)

    # --- Panel C: NVFP4 lasagna runaway rate by system prompt ---
    cond = sysprompt_stats["conditions"].get("35b_nvfp4_lasagna")
    if cond:
        items = []
        for st in cond["stats"]:
            n = st["n_clean"] + st["n_runaway"]
            if n:
                items.append((st["runaway_rate"], st, n))
        items.sort(key=lambda t: t[0], reverse=True)
        x = np.arange(len(items))
        colors = [CATEGORY_COLORS.get(st["category"], "#888888") for _, st, _ in items]
        rates = [r for r, _, _ in items]
        lo = [max(0.0, r - st["runaway_ci"][0]) for r, st, _ in items]
        hi = [max(0.0, st["runaway_ci"][1] - r) for r, st, _ in items]
        ax_p.bar(x, rates, 0.68, color=colors, edgecolor="white", linewidth=0.5,
                 yerr=[lo, hi], capsize=2, ecolor="#555555")
        for xi, (r, st, n) in enumerate(items):
            if r <= 0:
                continue
            ax_p.text(xi, st["runaway_ci"][1] + 0.02, f"{st['n_runaway']}/{n}",
                      ha="center", va="bottom", fontsize=6)
        ax_p.set_xticks(x)
        ax_p.set_xticklabels([st["name"] for _, st, _ in items],
                             rotation=45, ha="right", fontsize=7)
        ax_p.set_ylim(0, 1.18)
        ax_p.set_ylabel("Runaway-thinking rate")
        ax_p.set_title("35B MoE NVFP4, lasagna: runaway rate by system prompt",
                       fontsize=11)
        handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c)
                   for c in CATEGORY_ORDER]
        ax_p.legend(handles=handles, fontsize=7, ncol=6, loc="upper right")
    else:
        ax_p.set_visible(False)

    fig.suptitle("Runaway Thinking: empty-generation chain rates", fontsize=13)
    save_figure(fig, "fig11_runaway_rates.png")


# =============================================================================
# 6. SUMMARY + MAIN
# =============================================================================

_DROP_KEYS = ("finals", "finals_clean")


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _DROP_KEYS}


def save_all_stats(rq1: dict, rq1_las: dict, quant: dict, temp: dict,
                   sysprompt: dict, cross_gen: dict, completeness: dict,
                   classification: dict) -> None:
    all_stats = {
        "rq1_spaghetti": {
            "per_model": {k: _strip(v) for k, v in rq1["per_model"].items()},
            "kruskal_wallis": rq1["kruskal_wallis"],
            "dunn_posthoc": rq1["dunn_posthoc"],
            "effect_sizes": rq1["effect_sizes"],
        },
        "rq1_lasagna": {
            "per_model": {k: _strip(v) for k, v in rq1_las["per_model"].items()},
        },
        "rq2_quantization": {
            recipe: (None if not info else {
                "methods": {k: _strip(v) for k, v in info["methods"].items()},
                "deltas": info["deltas"],
            })
            for recipe, info in quant.items()
        },
        "temperature": {
            "per_model": {
                label: {
                    "display": info["display"],
                    "temps": {
                        str(t): {
                            recipe: _strip(stat)
                            for recipe, stat in ts.items() if stat
                        }
                        for t, ts in info["temps"].items()
                    }
                }
                for label, info in temp["per_model"].items()
            },
        },
        "sysprompt": {
            "spearman_pairs": sysprompt["spearman_pairs"],
            "spreads": sysprompt["spreads"],
            "extremes": sysprompt["extremes"],
            "category_means": sysprompt["category_means"],
            "cond_order": sysprompt["cond_order"],
            "per_condition": {
                cname: [_strip(st) for st in cdata["stats"]]
                for cname, cdata in sysprompt["conditions"].items()
            },
            "spreads_clean": sysprompt["spreads_clean"],
            "extremes_clean": sysprompt["extremes_clean"],
            "runaway_by_condition": sysprompt["runaway_by_condition"],
        },
        "cross_generation": cross_gen,
        "completeness": completeness,
        "chain_classification": classification,
    }

    path = OUT_DIR / "all_stats.json"
    path.write_text(json.dumps(numpy_to_python(all_stats), indent=2))
    print("  all_stats.json")


def compute_completeness(sources: dict[str, dict]) -> dict:
    """Per-experiment completed-chain counts and inference totals."""
    per_exp = {}
    for name, models in sources.items():
        attempted = len(models)
        complete = 0
        partial = []
        for key, iters in models.items():
            n_ok = len(extract_sims(iters))
            if n_ok >= MIN_COMPLETE_ITERS:
                complete += 1
            else:
                partial.append({"key": key, "successful_iterations": n_ok})
        per_exp[name] = {
            "keys_present": attempted,
            "complete_chains": complete,
            "incomplete_chains": sorted(partial, key=lambda d: d["key"]),
            "inferences": complete * ITERATIONS,
        }
    total_complete = sum(v["complete_chains"] for v in per_exp.values())
    return {
        "per_experiment": per_exp,
        "total_complete_chains": total_complete,
        "total_inferences": total_complete * ITERATIONS,
        "min_complete_iters": MIN_COMPLETE_ITERS,
    }


def _md_rate(st: dict | None) -> str:
    if not st:
        return "--"
    n = st.get("n_clean", 0) + st.get("n_runaway", 0)
    if not n:
        return "--"
    lo, hi = st["runaway_ci"]
    return f"{st['runaway_rate']:.2f} ({st['n_runaway']}/{n}) [{lo:.2f}, {hi:.2f}]"


def _md_clean(st: dict | None) -> str:
    if not st or st["clean"]["mean"] is None:
        return "--"
    return f"{st['clean']['mean']:.3f}"


def write_summary_for_text(rq1: dict, rq1_las: dict, quant: dict, temp: dict,
                           sysprompt: dict, cross_gen: dict,
                           completeness: dict, classification: dict) -> None:
    """Emit results/qwen35_paper_figures/summary_for_text.md."""
    out = []
    W = out.append
    W("# Paper 2 numbers summary (auto-generated by scripts/analyze_qwen35_paper.py)")
    W("")
    W("All means are final (iteration 30) cosine similarity to the original text.")
    W(f"A chain counts only if it produced {MIN_COMPLETE_ITERS} successful scored")
    W("iterations; `n` is the number of such chains (5 attempted per cell).")
    W("")

    # --- Completeness ---
    W("## 0. Data completeness")
    W("")
    W("| Experiment | keys present | complete chains | inferences |")
    W("|---|---|---|---|")
    for name, v in completeness["per_experiment"].items():
        W(f"| {name} | {v['keys_present']} | {v['complete_chains']} | {v['inferences']} |")
    W(f"| **TOTAL** | | **{completeness['total_complete_chains']}** | "
      f"**{completeness['total_inferences']}** |")
    W("")
    for name, v in completeness["per_experiment"].items():
        if v["incomplete_chains"]:
            W(f"Incomplete in {name} ({len(v['incomplete_chains'])}): " +
              ", ".join(f"`{d['key']}` ({d['successful_iterations']} it)"
                        for d in v["incomplete_chains"]))
            W("")

    # --- Runaway thinking ---
    W("## 0b. Runaway thinking")
    W("")
    W("A chain is **runaway** when at least one of its 30 iterations was a")
    W("successful call that returned an empty assistant message after a")
    W("full-length generation (the whole token budget went to reasoning).")
    W("Once a chain goes runaway it never recovers: the same input is re-fed at")
    W("the next iteration. Empty records with `elapsed_seconds == 0` are")
    W("**padding** appended by the rerun harness after two consecutive empties;")
    W("they mark an already-runaway chain and are never counted as separate")
    W("events. **Clean** chains have no empty iteration at all; **incomplete**")
    W(f"chains have fewer than {MIN_COMPLETE_ITERS} successful scored iterations")
    W("(or do not end on a success). The same rule is applied to every model,")
    W("0.8B included.")
    W("")
    W("| Data file | clean | runaway | incomplete | runaway rate | 95% CI (Wilson) |")
    W("|---|---|---|---|---|---|")
    for name, v in classification["per_file"].items():
        W(f"| {name} | {v['clean']} | {v['runaway']} | {v['incomplete']} | "
          f"{v['runaway_rate']:.3f} | "
          f"[{v['runaway_ci'][0]:.3f}, {v['runaway_ci'][1]:.3f}] |")
    t = classification["totals"]
    W(f"| **TOTAL** | **{t['clean']}** | **{t['runaway']}** | "
      f"**{t['incomplete']}** | **{t['runaway_rate']:.3f}** | "
      f"[{t['runaway_ci'][0]:.3f}, {t['runaway_ci'][1]:.3f}] |")
    W("")
    onsets = [c["runaway_onset"] for v in classification["per_file"].values()
              for c in v["runaway_chains"] if c["runaway_onset"] is not None]
    if onsets:
        W(f"Runaway onset (first empty generation), n={len(onsets)} chains: "
          f"median iteration {int(np.median(onsets))}, "
          f"mean {np.mean(onsets):.1f}, range {min(onsets)}-{max(onsets)}.")
        W("")

    # --- RQ1 ---
    W("## 1. RQ1 - size x quantization")
    W("")
    W("Clean = clean chains only (primary); All = every completed chain")
    W("including runaways (previous behaviour).")
    W("")
    W("| Model | Spag clean | Spag all | n_clean | Spag runaway rate | "
      "Las clean | Las all | n_clean | Las runaway rate | Tier |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for label, info in sorted(
            rq1["per_model"].items(),
            key=lambda x: (x[1]["clean"]["mean"] if x[1]["clean"]["mean"] is not None
                           else -1.0), reverse=True):
        li = rq1_las["per_model"].get(label)
        las_all = f"{li['mean']:.3f}" if li else "--"
        las_nc = str(li["n_clean"]) if li else "0"
        W(f"| {info['display']} | {_md_clean(info)} | {info['mean']:.3f} | "
          f"{info['n_clean']} | {_md_rate(info)} | "
          f"{_md_clean(li)} | {las_all} | {las_nc} | {_md_rate(li)} | "
          f"{info.get('tier_clean', info['tier'])} |")
    W("")
    tiers = {}
    for info in rq1["per_model"].values():
        tiers.setdefault(info["tier"], []).append(info["mean"])
    W("Tier occupancy (spaghetti, by model mean):")
    for t, _thr in TIER_THRESHOLDS:
        v = tiers.get(t, [])
        if v:
            W(f"- {t}: {len(v)} models, model means {min(v):.3f}--{max(v):.3f}")
        else:
            W(f"- {t}: 0 models (band empty by observation)")
    kw = rq1["kruskal_wallis"]
    if kw["H"] is not None:
        W("")
        W(f"Kruskal-Wallis across occupied tiers: H = {kw['H']:.2f}, p = {kw['p']:.3e}")
    for k, v in rq1["effect_sizes"].items():
        W(f"Cohen's d ({k}): {v:.2f}")
    W("")

    # --- RQ2 ---
    W("## 2. RQ2 - four-way quantization comparison (35B MoE)")
    W("")
    for recipe in ["spaghetti", "lasagna"]:
        info = quant.get(recipe)
        W(f"### {recipe}")
        W("")
        if not info:
            W("No data.")
            W("")
            continue
        W("| Method | Clean mean | Clean 95% CI | n_clean | All mean | n_all | Runaway rate |")
        W("|---|---|---|---|---|---|---|")
        for key, _d, _s, _l in QUANT_METHODS:
            m = info["methods"].get(key)
            if not m:
                continue
            c = m["clean"]
            cci = ("--" if c["mean"] is None
                   else f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
            W(f"| {m['display'].replace(chr(92)+'_','_')} | {_md_clean(m)} | "
              f"{cci} | {m['n_clean']} | {m['mean']:.3f} | {m['n']} | "
              f"{_md_rate(m)} |")
        W("")
        W("Paired deltas vs GGUF Q4_K_M (same seed in both arms):")
        W("")
        W("| Method | delta | 95% CI | n pairs | Wilcoxon p | Cohen's d |")
        W("|---|---|---|---|---|---|")
        for key, d in info["deltas"].items():
            pv = f"{d['wilcoxon_p']:.3f}" if d["wilcoxon_p"] is not None else "n/a"
            W(f"| {key.upper()} | {d['delta_mean']:+.3f} | "
              f"[{d['delta_ci'][0]:.3f}, {d['delta_ci'][1]:.3f}] | {d['n_pairs']} | "
              f"{pv} | {d['cohens_d']:.2f} |")
        W("")

    # --- RQ3 ---
    W("## 3. RQ3 - temperature")
    W("")
    W("Cells are: clean mean / all mean (n_clean, runaway rate).")
    W("")
    W("| Model | Recipe | " + " | ".join(f"T={t}" for t in TEMPERATURES) + " |")
    W("|---|---|" + "---|" * len(TEMPERATURES))
    for label, info in temp["per_model"].items():
        for recipe in ["spaghetti", "lasagna"]:
            cells = []
            any_data = False
            for t in TEMPERATURES:
                st = info["temps"].get(t, {}).get(recipe)
                if st:
                    any_data = True
                    n_cmp = st["n_clean"] + st["n_runaway"]
                    ra = (f"{st['runaway_rate']:.2f}" if n_cmp else "--")
                    cells.append(
                        f"{_md_clean(st)} / {st['mean']:.3f} "
                        f"(nc={st['n_clean']}, RA={ra})")
                else:
                    cells.append("--")
            if any_data:
                W(f"| {info['display']} | {recipe} | " + " | ".join(cells) + " |")
    W("")

    # --- RQ4 ---
    W("## 4. RQ4 - system prompts (lasagna unless noted)")
    W("")
    W("Clean-chain summary first, all-chain mean alongside.")
    W("")
    W("| Configuration | Recipe | clean mean of 20 prompt means | all mean | "
      "spread (clean) | best (clean) | worst (clean) | min n_clean | runaway rate |")
    W("|---|---|---|---|---|---|---|---|---|")
    for cname in sysprompt["cond_order"]:
        e = sysprompt["extremes_clean"].get(cname)
        e_all = sysprompt["extremes"].get(cname)
        if not e and not e_all:
            continue
        model, recipe = cname.rsplit("_", 1)
        ra = sysprompt["runaway_by_condition"].get(cname, {})
        ra_s = (f"{ra['runaway_rate']:.3f} ({ra['n_runaway']}/{ra['n_complete']})"
                if ra.get("n_complete") else "--")
        all_mean = f"{e_all['mean_of_prompts']:.3f}" if e_all else "--"
        if e:
            W(f"| {RQ4_DISPLAY.get(model, model)} | {recipe} | "
              f"{e['mean_of_prompts']:.3f} | {all_mean} | "
              f"{sysprompt['spreads_clean'].get(cname, 0):.3f} | "
              f"{e['best']['name']} ({e['best']['mean']:.3f}) | "
              f"{e['worst']['name']} ({e['worst']['mean']:.3f}) | {e['min_n']} | {ra_s} |")
        else:
            W(f"| {RQ4_DISPLAY.get(model, model)} | {recipe} | -- | "
              f"{all_mean} | -- | -- | -- | 0 | {ra_s} |")
    W("")
    W("Highest per-prompt runaway rates (all sysprompt conditions):")
    W("")
    W("| Configuration | Recipe | Prompt | runaway rate | 95% CI |")
    W("|---|---|---|---|---|")
    rows = []
    for cname, cdata in sysprompt["conditions"].items():
        model, recipe = cname.rsplit("_", 1)
        for st in cdata["stats"]:
            n_cmp = st["n_clean"] + st["n_runaway"]
            if n_cmp and st["n_runaway"]:
                rows.append((st["runaway_rate"], RQ4_DISPLAY.get(model, model),
                             recipe, st))
    rows.sort(key=lambda r: r[0], reverse=True)
    for rate, disp, recipe, st in rows[:15]:
        W(f"| {disp} | {recipe} | {st['name']} | "
          f"{rate:.2f} ({st['n_runaway']}/{st['n_clean']+st['n_runaway']}) | "
          f"[{st['runaway_ci'][0]:.2f}, {st['runaway_ci'][1]:.2f}] |")
    W("")
    W("Pairwise Spearman rho on the 20 per-prompt means (lasagna):")
    W("")
    W("| A | B | rho | 95% CI | p | prompts |")
    W("|---|---|---|---|---|---|")
    for key, st in sysprompt["spearman_pairs"].items():
        a, b = key.split("_vs_")
        W(f"| {RQ4_DISPLAY.get(a.replace('_lasagna',''),a)} | "
          f"{RQ4_DISPLAY.get(b.replace('_lasagna',''),b)} | {st['rho']:.3f} | "
          f"[{st['ci'][0]:.3f}, {st['ci'][1]:.3f}] | {st['p']:.4f} | {st['n_prompts']} |")
    W("")
    W("Category means (rows = category, columns = " +
      ", ".join(sysprompt["cond_order"]) + "):")
    W("")
    W("| Category | " + " | ".join(sysprompt["cond_order"]) + " |")
    W("|---|" + "---|" * len(sysprompt["cond_order"]))
    for cat in CATEGORY_ORDER:
        row = sysprompt["category_means"][cat]
        W(f"| {cat} | " + " | ".join(f"{v:.3f}" if v else "--" for v in row) + " |")
    W("")

    # --- Cross-generation ---
    W("## 5. Cross-generation (Qwen3 pilot, single run vs Qwen3.5)")
    W("")
    W("| Pair | Qwen3 (1 run) | Qwen3.5 clean mean | clean 95% CI | n_clean | "
      "Qwen3.5 all mean | runaway rate | delta (clean - Qwen3) |")
    W("|---|---|---|---|---|---|---|---|")
    for pr in cross_gen["pairs"]:
        if pr["qwen3_final"] is None or pr["qwen35_mean"] is None:
            continue
        c = pr["qwen35_clean"]
        cm = c["mean"]
        cci = "--" if cm is None else f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
        delta = "--" if cm is None else f"{cm - pr['qwen3_final']:+.3f}"
        n_cmp = pr["n_clean"] + pr["n_runaway"]
        ra = (f"{pr['runaway_rate']:.2f} ({pr['n_runaway']}/{n_cmp})"
              if n_cmp else "--")
        W(f"| {pr['display']} | {pr['qwen3_final']:.3f} | "
          f"{'--' if cm is None else f'{cm:.3f}'} | {cci} | {pr['n_clean']} | "
          f"{pr['qwen35_mean']:.3f} | {ra} | {delta} |")
    W("")

    # --- Embedding validation ---
    W("## 6. Embedding validation (from results/embedding_validation_qwen35)")
    W("")
    val_path = Path("results/embedding_validation_qwen35/correlation_stats.json")
    if val_path.exists():
        try:
            v = json.loads(val_path.read_text())
            W("```json")
            W(json.dumps(v, indent=2)[:4000])
            W("```")
        except Exception as exc:  # pragma: no cover
            W(f"Could not parse {val_path}: {exc}")
    else:
        W(f"`{val_path}` not present; values carried from the paper appendix: "
          "20,072 of 22,167 pairs valid (90.6%), Pearson r = 0.825, "
          "Spearman rho = 0.815, 2,095 pairs excluded for exceeding bge-m3's "
          "8,192-token context.")
    W("")

    path = OUT_DIR / "summary_for_text.md"
    path.write_text("\n".join(out) + "\n")
    print(f"  summary_for_text.md")


def print_console_summary(rq1: dict, rq1_las: dict, quant: dict, temp: dict,
                          sysprompt: dict, cross_gen: dict, completeness: dict) -> None:
    print(f"\n{'='*80}")
    print("QWEN3.5 PAPER STATISTICS SUMMARY")
    print(f"{'='*80}")

    print(f"\n--- RQ1: Size x Quant, Spaghetti ({len(rq1['per_model'])} models) ---")
    for label, info in sorted(rq1["per_model"].items(),
                              key=lambda x: x[1]["mean"], reverse=True):
        cm = info["clean"]["mean"]
        cm_s = "  --  " if cm is None else f"{cm:>6.3f}"
        print(f"  {info['display']:<25} clean={cm_s} (n={info['n_clean']}) "
              f"all={info['mean']:.3f} (n={info['n']}) "
              f"RA={info['runaway_rate']:.2f} {info.get('tier_clean', info['tier'])}")
    kw = rq1["kruskal_wallis"]
    if kw["H"] is not None:
        print(f"  Kruskal-Wallis H = {kw['H']:.2f}, p = {kw['p']:.2e}")

    print("\n--- RQ2: four-way quantization ---")
    for recipe, info in quant.items():
        if not info:
            continue
        parts = [
            f"{k}: clean="
            + ("--" if v["clean"]["mean"] is None else f"{v['clean']['mean']:.3f}")
            + f"(n={v['n_clean']}) all={v['mean']:.3f}(n={v['n']}) "
              f"RA={v['runaway_rate']:.2f}"
            for k, v in info["methods"].items()]
        print(f"  {recipe}: " + ", ".join(parts))
        for k, d in info["deltas"].items():
            print(f"    {k}-gguf delta={d['delta_mean']:+.3f} "
                  f"[{d['delta_ci'][0]:.3f}, {d['delta_ci'][1]:.3f}] n={d['n_pairs']}")

    print("\n--- RQ4 spreads ---")
    for cname in sysprompt["cond_order"]:
        print(f"  {cname:<24} spread={sysprompt['spreads'].get(cname, 0):.3f}")

    print("\n--- Completeness ---")
    for name, v in completeness["per_experiment"].items():
        print(f"  {name:<18} {v['complete_chains']}/{v['keys_present']} complete "
              f"({v['inferences']} inferences)")
    print(f"  TOTAL complete chains: {completeness['total_complete_chains']}")
    print(f"  TOTAL inferences: {completeness['total_inferences']}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PAPER_STYLE)

    # --- Load datasets ---
    print("Loading datasets...")

    def load_json(path: Path, name: str) -> dict:
        if not path.exists():
            print(f"  {name}: NOT FOUND ({path})")
            return {}
        raw = json.loads(path.read_text())
        models = raw.get("models", {})
        print(f"  {name}: {len(models)} runs")
        return models

    exp1_models = load_json(EXP1_DATA, "Exp 1 (size x quant)")
    exp2_models = load_json(EXP2_DATA, "Exp 2 (temperature)")
    exp3a_models = load_json(EXP3A_DATA, "Exp 3a (vLLM drift)")
    exp3b_models = load_json(EXP3B_DATA, "Exp 3b (vLLM temp)")
    exp3c_models = load_json(EXP3C_DATA, "Exp 3c (vLLM sysprompt)")
    exp4_models = load_json(EXP4_DATA, "Exp 4 (sysprompt)")
    qwen3_pilot = load_json(PAPER1_QWEN3_DATA, "Paper 1 Qwen3 pilot")

    # Backends that write their own files. Exp 1 / Exp 2 keys carry the model
    # label, so they merge into the shared dicts without collision; sysprompt
    # keys do NOT carry the model label ("{recipe}_{id}_{slug}_s{seed}") and
    # must stay in separate dicts.
    extra_backends = [
        ("NVFP4",     NVFP4_EXP1_DATA,  NVFP4_EXP2_DATA,  NVFP4_SYSPROMPT_DATA,  "35b_nvfp4"),
        ("GPTQ-Int4", GPTQ_EXP1_DATA,   GPTQ_EXP2_DATA,   GPTQ_SYSPROMPT_DATA,   "35b_gptq"),
        ("9B AWQ",    AWQ9B_EXP1_DATA,  AWQ9B_EXP2_DATA,  AWQ9B_SYSPROMPT_DATA,  "9b_awq"),
        ("Nemotron",  NEMO_EXP1_DATA,   NEMO_EXP2_DATA,   NEMO_SYSPROMPT_DATA,   "nemo_30b"),
    ]
    flat_sysprompt_sources: dict[str, dict] = {"35b_awq": exp3c_models}
    # Snapshot the Ollama/vLLM dicts BEFORE the extra backends are merged in,
    # so each backend is counted against its own file exactly once.
    completeness_sources = {
        "exp1_ollama_size_quant": dict(exp1_models),
        "exp2_ollama_temperature": dict(exp2_models),
        "exp3a_vllm_awq_drift": dict(exp3a_models),
        "exp3b_vllm_awq_temperature": dict(exp3b_models),
        "exp3c_vllm_awq_sysprompt": dict(exp3c_models),
        "exp4_ollama_sysprompt": dict(exp4_models),
    }
    for name, e1, e2, sysp, short_key in extra_backends:
        d1 = load_json(e1, f"{name} Exp 1")
        d2 = load_json(e2, f"{name} Exp 2")
        ds = load_json(sysp, f"{name} sysprompt")
        if d1:
            exp1_models.update(d1)
            completeness_sources[f"exp1_{short_key}"] = d1
        if d2:
            exp2_models.update(d2)
            completeness_sources[f"exp2_{short_key}"] = d2
        if ds:
            flat_sysprompt_sources[short_key] = ds
            completeness_sources[f"sysprompt_{short_key}"] = ds

    # --- Completeness accounting (before any statistic is computed) ---
    completeness = compute_completeness(completeness_sources)
    classification = classify_all_files(completeness_sources)
    print("\nChain classification (clean / runaway / incomplete):")
    for name, v in classification["per_file"].items():
        print(f"  {name:<26} {v['clean']:>4} / {v['runaway']:>4} / "
              f"{v['incomplete']:>3}  (runaway rate {v['runaway_rate']:.3f})")
    t = classification["totals"]
    print(f"  {'TOTAL':<26} {t['clean']:>4} / {t['runaway']:>4} / "
          f"{t['incomplete']:>3}  (runaway rate {t['runaway_rate']:.3f})")

    # --- Compute stats ---
    print("\nComputing statistics...")
    rq1_stats = compute_rq1_stats(exp1_models)
    print(f"  RQ1 spaghetti: {len(rq1_stats['per_model'])} models")

    rq1_las_stats = compute_rq1_lasagna_stats(exp1_models)
    print(f"  RQ1 lasagna: {len(rq1_las_stats['per_model'])} models")

    quant_stats = compute_quant_comparison_stats(exp1_models, exp3a_models)
    print(f"  RQ2 quantization: {sum(1 for v in quant_stats.values() if v)} recipes")

    temp_stats = compute_temperature_stats(exp2_models, exp3b_models)
    print(f"  Temperature: {len(temp_stats['per_model'])} models")

    sysprompt_stats = compute_sysprompt_stats(exp4_models, flat_sysprompt_sources)
    print(f"  Sysprompt: {len(sysprompt_stats['conditions'])} conditions")

    cross_gen_stats = compute_cross_gen_stats(exp1_models, qwen3_pilot)
    n_valid = sum(1 for p in cross_gen_stats["pairs"]
                  if p["qwen3_final"] is not None and p["qwen35_mean"] is not None)
    print(f"  Cross-gen: {n_valid} valid pairs")

    # --- Generate figures ---
    print("\nGenerating figures...")
    if rq1_stats["per_model"]:
        plot_fig1_size_quant_lines(exp1_models, rq1_stats)
    if rq1_stats["per_model"] and rq1_las_stats["per_model"]:
        plot_fig2_boxplot(rq1_stats, rq1_las_stats)
    if rq1_stats["per_model"]:
        plot_fig3_quant_heatmap(rq1_stats, rq1_las_stats)
    if exp3a_models:
        plot_fig4_quant_comparison(quant_stats, exp1_models, exp3a_models)
    if temp_stats["per_model"]:
        plot_fig5_temperature(temp_stats)
    if rq1_stats["per_model"]:
        plot_fig6_complexity(rq1_stats, rq1_las_stats)
    if exp4_models:
        plot_fig7_sysprompt_hero(sysprompt_stats, exp4_models)
    if sysprompt_stats["conditions"]:
        plot_fig8_sysprompt_cross_model(sysprompt_stats)
    if sysprompt_stats["conditions"]:
        plot_fig9_category_heatmap(sysprompt_stats)
    if cross_gen_stats["pairs"]:
        plot_fig10_cross_gen(cross_gen_stats)
    if rq1_las_stats["per_model"]:
        plot_figA1_lasagna_lines(exp1_models, rq1_las_stats)
    if rq1_stats["per_model"]:
        plot_figA2_seed_variance(exp1_models)
    if rq1_stats["per_model"]:
        plot_fig11_runaway_rates(rq1_stats, rq1_las_stats, sysprompt_stats)
    if temp_stats["per_model"]:
        plot_figA3_temperature_multiples(temp_stats)

    # --- Export tables ---
    print("\nExporting LaTeX tables...")
    if rq1_stats["per_model"]:
        export_table1_size_quant(rq1_stats, rq1_las_stats)
    if sysprompt_stats["conditions"]:
        export_table2_sysprompt(sysprompt_stats)
        export_table3_rq4(sysprompt_stats)
    if rq1_stats["per_model"]:
        export_table4_runaway(rq1_stats, rq1_las_stats, sysprompt_stats)

    # --- Save stats ---
    print("\nSaving statistics...")
    save_all_stats(rq1_stats, rq1_las_stats, quant_stats, temp_stats,
                   sysprompt_stats, cross_gen_stats, completeness, classification)
    write_summary_for_text(rq1_stats, rq1_las_stats, quant_stats, temp_stats,
                           sysprompt_stats, cross_gen_stats, completeness,
                           classification)

    # --- Console summary ---
    print_console_summary(rq1_stats, rq1_las_stats, quant_stats, temp_stats,
                          sysprompt_stats, cross_gen_stats, completeness)

    print(f"\n{'='*80}")
    print(f"All outputs in: {OUT_DIR}")
    print("Tables in: " + ", ".join(str(d) for d in TABLE_DIRS))
    print("Done!")


if __name__ == "__main__":
    main()
