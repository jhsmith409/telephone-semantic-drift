#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Analyze KV cache precision impact on semantic drift across 22 Qwen3 models.

Loads q8_0 KV and fp16 KV drift experiment data, computes deltas, and generates:
  1. Hero overlay plot (selected models, dashed=q8 vs solid=fp16)
  2. Delta bar chart (all 22 models, sorted by delta)
  3. Heatmap (size × weight quant grid, delta values)
  4. Small multiples (22 subplots, one per model)

Usage:
    uv run python scripts/analyze_kv_cache_comparison.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# --- Paths ---
Q8_DATA = Path("results/drift_experiment_qwen3/drift_data.json")
FP16_DATA = Path("results/drift_experiment_qwen3_fp16kv/drift_data.json")
OUT_DIR = Path("results/kv_cache_comparison")
BLOG_IMG_DIR = Path("../j8web/images/llm-telephone-drift-kv")

ITERATIONS = 30

# --- Model metadata ---
# Maps model key -> (display_name, size_group, weight_quant)
MODEL_META = {
    "qwen3:0.6b-fp16":                      ("0.6B fp16",       "0.6B",       "fp16"),
    "qwen3:0.6b-q8_0":                      ("0.6B q8_0",       "0.6B",       "q8_0"),
    "qwen3:0.6b-q4_K_M":                    ("0.6B q4_K_M",     "0.6B",       "q4_K_M"),
    "qwen3:1.7b-fp16":                      ("1.7B fp16",       "1.7B",       "fp16"),
    "qwen3:1.7b-q8_0":                      ("1.7B q8_0",       "1.7B",       "q8_0"),
    "qwen3:1.7b-q4_K_M":                    ("1.7B q4_K_M",     "1.7B",       "q4_K_M"),
    "qwen3:4b-instruct-2507-fp16":           ("4B inst fp16",    "4B inst",    "fp16"),
    "qwen3:4b-instruct-2507-q8_0":           ("4B inst q8_0",    "4B inst",    "q8_0"),
    "qwen3:4b-instruct-2507-q4_K_M":         ("4B inst q4_K_M",  "4B inst",    "q4_K_M"),
    "qwen3:8b-fp16":                         ("8B fp16",         "8B",         "fp16"),
    "qwen3:8b-q8_0":                         ("8B q8_0",         "8B",         "q8_0"),
    "qwen3:8b-q4_K_M":                       ("8B q4_K_M",       "8B",         "q4_K_M"),
    "qwen3:30b-a3b-instruct-2507-fp16":      ("30B inst fp16",   "30B inst",   "fp16"),
    "qwen3:30b-a3b-instruct-2507-q8_0":      ("30B inst q8_0",   "30B inst",   "q8_0"),
    "qwen3:30b-a3b-instruct-2507-q4_K_M":    ("30B inst q4_K_M", "30B inst",   "q4_K_M"),
    "qwen3:30b-a3b-thinking-2507-fp16":      ("30B think fp16",  "30B think",  "fp16"),
    "qwen3:30b-a3b-thinking-2507-q8_0":      ("30B think q8_0",  "30B think",  "q8_0"),
    "qwen3:30b-a3b-thinking-2507-q4_K_M":    ("30B think q4_K_M","30B think",  "q4_K_M"),
    "qwen3:32b-fp16":                        ("32B fp16",        "32B",        "fp16"),
    "qwen3:32b-q8_0":                        ("32B q8_0",        "32B",        "q8_0"),
    "qwen3:32b-q4_K_M":                      ("32B q4_K_M",      "32B",        "q4_K_M"),
    "qwen3-next:80b-a3b-instruct-q4_K_M":    ("80B next q4_K_M", "80B next",   "q4_K_M"),
}

SIZE_ORDER = ["0.6B", "1.7B", "4B inst", "8B", "30B inst", "30B think", "32B", "80B next"]
QUANT_ORDER = ["fp16", "q8_0", "q4_K_M"]

# Hero plot: models showing the most dramatic/interesting differences
HERO_MODELS = [
    "qwen3:0.6b-fp16",                       # biggest winner
    "qwen3:0.6b-q8_0",                       # huge winner
    "qwen3:1.7b-q8_0",                       # big winner
    "qwen3:4b-instruct-2507-q4_K_M",         # immune/stable
    "qwen3:32b-q4_K_M",                      # nice win
    "qwen3:30b-a3b-thinking-2507-q4_K_M",    # notable regression
    "qwen3-next:80b-a3b-instruct-q4_K_M",    # catastrophic regression
]

# Consistent colors per hero model
HERO_COLORS = [
    "#4fc3f7",  # light blue
    "#81c784",  # green
    "#ffb74d",  # orange
    "#ce93d8",  # purple
    "#e57373",  # red
    "#fff176",  # yellow
    "#ff8a65",  # deep orange
]


def load_data() -> tuple[dict, dict]:
    """Load both datasets and return {model_key: [cosine_sim_per_iter]}."""
    q8_raw = json.loads(Q8_DATA.read_text())
    fp16_raw = json.loads(FP16_DATA.read_text())

    def extract_sims(raw: dict) -> dict[str, list[float | None]]:
        result = {}
        for model_key, iterations in raw["models"].items():
            sims = []
            for r in iterations:
                sims.append(r.get("cosine_similarity"))
            result[model_key] = sims
        return result

    return extract_sims(q8_raw), extract_sims(fp16_raw)


def final_sim(sims: list[float | None]) -> float:
    """Get last non-None similarity value."""
    valid = [s for s in sims if s is not None]
    return valid[-1] if valid else 0.0


def compute_deltas(q8: dict, fp16: dict) -> dict[str, dict]:
    """Compute per-model summary stats and deltas."""
    results = {}
    for model_key in sorted(q8.keys()):
        if model_key not in fp16:
            continue
        q8_final = final_sim(q8[model_key])
        fp16_final = final_sim(fp16[model_key])
        delta = fp16_final - q8_final
        meta = MODEL_META.get(model_key, (model_key, "?", "?"))
        results[model_key] = {
            "display_name": meta[0],
            "size_group": meta[1],
            "weight_quant": meta[2],
            "q8_final": q8_final,
            "fp16_final": fp16_final,
            "delta": delta,
        }
    return results


def plot_hero_overlay(q8: dict, fp16: dict, deltas: dict) -> None:
    """Plot 1: Selected models overlaid, dashed=q8 KV vs solid=fp16 KV."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, model_key in enumerate(HERO_MODELS):
        color = HERO_COLORS[i]
        display = deltas[model_key]["display_name"]
        delta = deltas[model_key]["delta"]
        sign = "+" if delta >= 0 else ""

        # q8_0 KV (dashed)
        q8_sims = q8[model_key]
        q8_valid = [(j + 1, s) for j, s in enumerate(q8_sims) if s is not None]
        if q8_valid:
            xs, ys = zip(*q8_valid)
            ax.plot(xs, ys, linestyle="--", linewidth=1.5, color=color, alpha=0.7,
                    marker="o", markersize=2.5)

        # fp16 KV (solid)
        fp16_sims = fp16[model_key]
        fp16_valid = [(j + 1, s) for j, s in enumerate(fp16_sims) if s is not None]
        if fp16_valid:
            xs, ys = zip(*fp16_valid)
            ax.plot(xs, ys, linestyle="-", linewidth=2.2, color=color,
                    marker="o", markersize=3,
                    label=f"{display} ({sign}{delta:.2f})")

    # Legend entries for line styles
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="gray", linewidth=2.2, linestyle="-", label="fp16 KV cache"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", alpha=0.7, label="q8_0 KV cache"),
    ]

    main_legend = ax.legend(fontsize=9, loc="lower left", title="Model (\u0394 final sim)")
    ax.add_artist(main_legend)
    ax.legend(handles=style_handles, fontsize=9, loc="upper right")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax.set_title(
        "KV Cache Precision Impact on Semantic Drift\n"
        "Selected Qwen3 Models — Dashed = q8_0 KV, Solid = fp16 KV",
        fontsize=13,
    )
    ax.set_xlim(0.5, ITERATIONS + 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, ITERATIONS + 1, 2))

    path = OUT_DIR / "drift_overlay_selected.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Hero overlay: {path}")

    # Blog-sized copy
    blog_path = BLOG_IMG_DIR / "drift_overlay_selected.png"
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    for i, model_key in enumerate(HERO_MODELS):
        color = HERO_COLORS[i]
        display = deltas[model_key]["display_name"]
        delta = deltas[model_key]["delta"]
        sign = "+" if delta >= 0 else ""

        q8_sims = q8[model_key]
        q8_valid = [(j + 1, s) for j, s in enumerate(q8_sims) if s is not None]
        if q8_valid:
            xs, ys = zip(*q8_valid)
            ax2.plot(xs, ys, linestyle="--", linewidth=1.5, color=color, alpha=0.7,
                     marker="o", markersize=2.5)

        fp16_sims = fp16[model_key]
        fp16_valid = [(j + 1, s) for j, s in enumerate(fp16_sims) if s is not None]
        if fp16_valid:
            xs, ys = zip(*fp16_valid)
            ax2.plot(xs, ys, linestyle="-", linewidth=2.2, color=color,
                     marker="o", markersize=3,
                     label=f"{display} ({sign}{delta:.2f})")

    main_legend2 = ax2.legend(fontsize=9, loc="lower left", title="Model (\u0394 final sim)")
    ax2.add_artist(main_legend2)
    ax2.legend(handles=style_handles, fontsize=9, loc="upper right")

    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Cosine Similarity to Original", fontsize=12)
    ax2.set_title(
        "KV Cache Precision Impact on Semantic Drift\n"
        "Selected Qwen3 Models — Dashed = q8_0 KV, Solid = fp16 KV",
        fontsize=13,
    )
    ax2.set_xlim(0.5, ITERATIONS + 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, ITERATIONS + 1, 2))

    fig2.savefig(blog_path, dpi=100, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Hero overlay (blog): {blog_path}")


def plot_delta_bar_chart(deltas: dict) -> None:
    """Plot 2: All 22 models sorted by delta, green/red coloring."""
    sorted_models = sorted(deltas.items(), key=lambda x: x[1]["delta"], reverse=True)

    names = [d["display_name"] for _, d in sorted_models]
    delta_vals = [d["delta"] for _, d in sorted_models]
    colors = ["#4caf50" if d >= 0 else "#f44336" for d in delta_vals]

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(range(len(names)), delta_vals, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Delta Final Similarity (fp16 KV - q8_0 KV)", fontsize=12)
    ax.set_title(
        "Impact of KV Cache Precision on Final Cosine Similarity\n"
        "22 Qwen3 Models — Green = Improved, Red = Regressed",
        fontsize=13,
    )
    ax.axvline(x=0, color="white", linewidth=1.0, alpha=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate bars with values
    for i, (bar, val) in enumerate(zip(bars, delta_vals)):
        sign = "+" if val >= 0 else ""
        x_pos = val + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, i, f"{sign}{val:.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")

    fig.tight_layout()
    path = OUT_DIR / "delta_bar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Delta bar chart: {path}")

    # Blog copy
    blog_path = BLOG_IMG_DIR / "delta_bar_chart.png"
    fig.savefig(blog_path, dpi=100, bbox_inches="tight") if False else None
    # Re-render for blog
    fig2, ax2 = plt.subplots(figsize=(14, 9))
    bars2 = ax2.barh(range(len(names)), delta_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Delta Final Similarity (fp16 KV - q8_0 KV)", fontsize=12)
    ax2.set_title(
        "Impact of KV Cache Precision on Final Cosine Similarity\n"
        "22 Qwen3 Models — Green = Improved, Red = Regressed",
        fontsize=13,
    )
    ax2.axvline(x=0, color="white", linewidth=1.0, alpha=0.8)
    ax2.grid(True, axis="x", alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, delta_vals)):
        sign = "+" if val >= 0 else ""
        x_pos = val + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax2.text(x_pos, i, f"{sign}{val:.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(BLOG_IMG_DIR / "delta_bar_chart.png", dpi=100, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Delta bar chart (blog): {BLOG_IMG_DIR / 'delta_bar_chart.png'}")


def plot_heatmap(deltas: dict) -> None:
    """Plot 3: Size × weight quant grid showing delta values."""
    # Build the grid
    grid = np.full((len(SIZE_ORDER), len(QUANT_ORDER)), np.nan)

    for model_key, info in deltas.items():
        size = info["size_group"]
        quant = info["weight_quant"]
        if size in SIZE_ORDER and quant in QUANT_ORDER:
            row = SIZE_ORDER.index(size)
            col = QUANT_ORDER.index(quant)
            grid[row, col] = info["delta"]

    # Diverging colormap centered on 0
    max_abs = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(QUANT_ORDER)))
    ax.set_xticklabels(QUANT_ORDER, fontsize=11)
    ax.set_yticks(range(len(SIZE_ORDER)))
    ax.set_yticklabels(SIZE_ORDER, fontsize=11)
    ax.set_xlabel("Weight Quantization", fontsize=12)
    ax.set_ylabel("Model Size", fontsize=12)
    ax.set_title(
        "KV Cache Precision Delta (fp16 - q8_0)\n"
        "Green = fp16 KV Helped, Red = fp16 KV Hurt",
        fontsize=13,
    )

    # Annotate cells
    for i in range(len(SIZE_ORDER)):
        for j in range(len(QUANT_ORDER)):
            val = grid[i, j]
            if not np.isnan(val):
                sign = "+" if val >= 0 else ""
                text_color = "black" if abs(val) < 0.3 else "white"
                ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="gray")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Delta (fp16 KV - q8_0 KV)")

    fig.tight_layout()
    path = OUT_DIR / "delta_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap: {path}")

    # Blog copy
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    im2 = ax2.imshow(grid, cmap="RdYlGn", norm=norm, aspect="auto")
    ax2.set_xticks(range(len(QUANT_ORDER)))
    ax2.set_xticklabels(QUANT_ORDER, fontsize=11)
    ax2.set_yticks(range(len(SIZE_ORDER)))
    ax2.set_yticklabels(SIZE_ORDER, fontsize=11)
    ax2.set_xlabel("Weight Quantization", fontsize=12)
    ax2.set_ylabel("Model Size", fontsize=12)
    ax2.set_title(
        "KV Cache Precision Delta (fp16 - q8_0)\n"
        "Green = fp16 KV Helped, Red = fp16 KV Hurt",
        fontsize=13,
    )
    for i in range(len(SIZE_ORDER)):
        for j in range(len(QUANT_ORDER)):
            val = grid[i, j]
            if not np.isnan(val):
                sign = "+" if val >= 0 else ""
                text_color = "black" if abs(val) < 0.3 else "white"
                ax2.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                         fontsize=11, fontweight="bold", color=text_color)
            else:
                ax2.text(j, i, "—", ha="center", va="center",
                         fontsize=11, color="gray")
    fig2.colorbar(im2, ax=ax2, shrink=0.8, label="Delta (fp16 KV - q8_0 KV)")
    fig2.tight_layout()
    fig2.savefig(BLOG_IMG_DIR / "delta_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Heatmap (blog): {BLOG_IMG_DIR / 'delta_heatmap.png'}")


def plot_small_multiples(q8: dict, fp16: dict, deltas: dict) -> None:
    """Plot 4: 22-subplot grid, one per model, q8 vs fp16."""
    # Sort by delta descending for visual flow
    sorted_models = sorted(deltas.items(), key=lambda x: x[1]["delta"], reverse=True)
    n = len(sorted_models)

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.2), squeeze=False)

    for idx, (model_key, info) in enumerate(sorted_models):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        # q8_0 KV
        q8_sims = q8.get(model_key, [])
        q8_valid = [(j + 1, s) for j, s in enumerate(q8_sims) if s is not None]
        if q8_valid:
            xs, ys = zip(*q8_valid)
            ax.plot(xs, ys, linestyle="--", linewidth=1.2, color="#f44336", alpha=0.8,
                    marker="o", markersize=1.5, label="q8_0 KV")

        # fp16 KV
        fp16_sims = fp16.get(model_key, [])
        fp16_valid = [(j + 1, s) for j, s in enumerate(fp16_sims) if s is not None]
        if fp16_valid:
            xs, ys = zip(*fp16_valid)
            ax.plot(xs, ys, linestyle="-", linewidth=1.5, color="#4fc3f7",
                    marker="o", markersize=1.5, label="fp16 KV")

        delta = info["delta"]
        sign = "+" if delta >= 0 else ""
        title_color = "#4caf50" if delta >= 0 else "#f44336"
        ax.set_title(f"{info['display_name']} ({sign}{delta:.3f})",
                     fontsize=9, fontweight="bold", color=title_color)
        ax.set_xlim(0.5, ITERATIONS + 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.set_xticks([1, 10, 20, 30])
        ax.tick_params(labelsize=7)

        if idx == 0:
            ax.legend(fontsize=7, loc="lower left")

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "KV Cache Precision: All 22 Qwen3 Models\n"
        "Blue solid = fp16 KV, Red dashed = q8_0 KV",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()

    path = OUT_DIR / "all_models_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Small multiples: {path}")

    # Blog copy (lower DPI)
    fig2, axes2 = plt.subplots(rows, cols, figsize=(20, rows * 3.2), squeeze=False)
    for idx, (model_key, info) in enumerate(sorted_models):
        row, col = divmod(idx, cols)
        ax = axes2[row][col]
        q8_sims = q8.get(model_key, [])
        q8_valid = [(j + 1, s) for j, s in enumerate(q8_sims) if s is not None]
        if q8_valid:
            xs, ys = zip(*q8_valid)
            ax.plot(xs, ys, linestyle="--", linewidth=1.2, color="#f44336", alpha=0.8,
                    marker="o", markersize=1.5, label="q8_0 KV")
        fp16_sims = fp16.get(model_key, [])
        fp16_valid = [(j + 1, s) for j, s in enumerate(fp16_sims) if s is not None]
        if fp16_valid:
            xs, ys = zip(*fp16_valid)
            ax.plot(xs, ys, linestyle="-", linewidth=1.5, color="#4fc3f7",
                    marker="o", markersize=1.5, label="fp16 KV")
        delta = info["delta"]
        sign = "+" if delta >= 0 else ""
        title_color = "#4caf50" if delta >= 0 else "#f44336"
        ax.set_title(f"{info['display_name']} ({sign}{delta:.3f})",
                     fontsize=9, fontweight="bold", color=title_color)
        ax.set_xlim(0.5, ITERATIONS + 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.set_xticks([1, 10, 20, 30])
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower left")
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes2[row][col].set_visible(False)
    fig2.suptitle(
        "KV Cache Precision: All 22 Qwen3 Models\n"
        "Blue solid = fp16 KV, Red dashed = q8_0 KV",
        fontsize=14, y=1.01,
    )
    fig2.tight_layout()
    fig2.savefig(BLOG_IMG_DIR / "all_models_comparison.png", dpi=100, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Small multiples (blog): {BLOG_IMG_DIR / 'all_models_comparison.png'}")


def print_summary(deltas: dict) -> None:
    """Print console summary table."""
    print(f"\n{'='*80}")
    print("KV CACHE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Size':<12} {'Quant':<8} {'q8 KV':>8} {'fp16 KV':>8} {'Delta':>8}")
    print(f"{'-'*25} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    sorted_models = sorted(deltas.items(), key=lambda x: x[1]["delta"], reverse=True)
    for model_key, info in sorted_models:
        sign = "+" if info["delta"] >= 0 else ""
        print(
            f"  {info['display_name']:<23} {info['size_group']:<12} {info['weight_quant']:<8} "
            f"{info['q8_final']:>7.4f} {info['fp16_final']:>8.4f} {sign}{info['delta']:>7.4f}"
        )

    # Summary stats
    all_deltas = [d["delta"] for d in deltas.values()]
    improved = sum(1 for d in all_deltas if d > 0.01)
    regressed = sum(1 for d in all_deltas if d < -0.01)
    neutral = len(all_deltas) - improved - regressed

    print(f"\n  Models improved (delta > +0.01):  {improved}")
    print(f"  Models regressed (delta < -0.01): {regressed}")
    print(f"  Models neutral (|delta| <= 0.01): {neutral}")
    print(f"  Mean delta: {np.mean(all_deltas):+.4f}")
    print(f"  Max improvement: {max(all_deltas):+.4f}")
    print(f"  Max regression: {min(all_deltas):+.4f}")

    # Save summary stats as JSON
    stats = {
        "models": {k: v for k, v in sorted_models},
        "summary": {
            "improved": improved,
            "regressed": regressed,
            "neutral": neutral,
            "mean_delta": round(float(np.mean(all_deltas)), 4),
            "max_improvement": round(float(max(all_deltas)), 4),
            "max_regression": round(float(min(all_deltas)), 4),
        },
    }
    stats_path = OUT_DIR / "summary_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\n  Summary stats saved: {stats_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BLOG_IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    print(f"  q8_0 KV: {Q8_DATA}")
    print(f"  fp16 KV: {FP16_DATA}")
    q8, fp16 = load_data()
    print(f"  q8_0 KV models: {len(q8)}")
    print(f"  fp16 KV models: {len(fp16)}")

    common = set(q8.keys()) & set(fp16.keys())
    print(f"  Common models: {len(common)}")

    print("\nComputing deltas...")
    deltas = compute_deltas(q8, fp16)

    print("\nGenerating plots...")
    plot_hero_overlay(q8, fp16, deltas)
    plot_delta_bar_chart(deltas)
    plot_heatmap(deltas)
    plot_small_multiples(q8, fp16, deltas)

    print_summary(deltas)

    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"Blog images in: {BLOG_IMG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
