#!/usr/bin/env python3
# Copyright (c) 2026 James H. Smith. MIT License.
"""Unified analysis script for Paper B (Paper 3): Qwen3.6 / Qwen3.8 drift study.

Reads results/paper3/<label>/*.json (read-only) plus the Paper A anchor datasets,
and writes:
  - 14 figures (12 main + 2 appendix) at 200 DPI into results/paper3_figures/
  - 9 LaTeX tables (booktabs) into paper3/tables/
  - results/paper3_figures/all_stats.json
  - results/paper3_figures/summary_for_text.md

Usage:
    uv run python scripts/analyze_paper3.py

Never prints IP addresses or lab hostnames: the two Qwen3.6-35B NVFP4 deployments
are called "host A" and "host B".
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import mannwhitneyu, rankdata, pearsonr

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "legacy"))
sys.path.insert(0, str(ROOT / "scripts" / "paper3"))

from paper3.common import PROMPTS, PROMPT_NAMES, _slug  # noqa: E402
from paper3.concrete_prompts import (  # noqa: E402
    CONCRETE_PROMPTS, DOMAIN_NEUTRAL_SYSPROMPT_IDS,
)
from paper3.domain_prompts import (  # noqa: E402
    ANCHORS, DOMAIN_ORDER, DOMAIN_PROMPTS, FIDELITY_PROMPT_ID,
)

# =============================================================================
# 1. CONSTANTS
# =============================================================================

DATA_ROOT = ROOT / "results" / "paper3"
OUT_DIR = ROOT / "results" / "paper3_figures"
TABLE_DIR = ROOT / "paper3" / "tables"

PAPER_A_QWEN3_PILOT = ROOT / "results" / "drift_experiment_qwen3" / "drift_data.json"
PAPER_A_QWEN35_GGUF = ROOT / "results" / "qwen35_size_quant" / "drift_data.json"
PAPER_A_QWEN35_NVFP4 = ROOT / "results" / "qwen35_nvfp4" / "exp1_drift_data.json"

ITERATIONS = 30
BOOTSTRAP_N = 10_000
RNG = np.random.default_rng(0)
DPI = 200

# Label -> display name (spec section "Labels and display names").
DISPLAY = {
    "qwen3.6-35b-nvfp4": "Qwen3.6-35B NVFP4 (host A)",
    "qwen3.6-35b-nvfp4-gpu-host-d": "Qwen3.6-35B NVFP4 (host B)",
    "qwen3.8-27b-nvfp4": "Qwen3.8-27B NVFP4",
    "qwen3.8-flash-next-nvfp4": "Qwen3.8-Flash-Next NVFP4",
    "qwen3.6-35b-q4kxl": "Qwen3.6-35B GGUF",
    "qwen3.6-27b-q4kxl": "Qwen3.6-27B GGUF",
    "qwen3.8-27b-q4kxl": "Qwen3.8-27B GGUF",
    "qwen3.8-27b-q4kxl-mtp": "Qwen3.8-27B GGUF+MTP",
}

# Analysis "configs": the two Qwen3.6-35B NVFP4 hosts are pooled.
POOLED_35B = ["qwen3.6-35b-nvfp4", "qwen3.6-35b-nvfp4-gpu-host-d"]
CONFIGS = [
    ("35b-nvfp4", "Qwen3.6-35B NVFP4", POOLED_35B),
    ("27b-nvfp4", "Qwen3.8-27B NVFP4", ["qwen3.8-27b-nvfp4"]),
    ("flash-nvfp4", "Qwen3.8-Flash-Next NVFP4", ["qwen3.8-flash-next-nvfp4"]),
    ("35b-gguf", "Qwen3.6-35B GGUF", ["qwen3.6-35b-q4kxl"]),
    ("27b36-gguf", "Qwen3.6-27B GGUF", ["qwen3.6-27b-q4kxl"]),
    ("27b38-gguf", "Qwen3.8-27B GGUF", ["qwen3.8-27b-q4kxl"]),
]
CONFIG_DISPLAY = {cid: disp for cid, disp, _ in CONFIGS}
CONFIG_LABELS = {cid: labs for cid, _, labs in CONFIGS}
MTP_CONFIG = ("27b38-gguf-mtp", "Qwen3.8-27B GGUF+MTP", ["qwen3.8-27b-q4kxl-mtp"])

CONFIG_COLORS = {
    "35b-nvfp4": "#1f77b4",
    "27b-nvfp4": "#d62728",
    "flash-nvfp4": "#9467bd",
    "35b-gguf": "#2ca02c",
    "27b36-gguf": "#8c564b",
    "27b38-gguf": "#ff7f0e",
    "27b38-gguf-mtp": "#7f7f7f",
}

RECIPES = ["spaghetti", "lasagna"]
RECIPE_DISPLAY = {"spaghetti": "Spaghetti (short)", "lasagna": "Lasagna (long)"}
CONCRETE_RECIPES = ["concrete_short", "concrete_long"]
CONCRETE_DISPLAY = {"concrete_short": "SOW short", "concrete_long": "SOW long"}

# Configs that carry a thinking ON/OFF pair.
THINKING_CONFIGS = [
    ("hostA", "Qwen3.6-35B NVFP4 (host A)", "qwen3.6-35b-nvfp4"),
    ("hostB", "Qwen3.6-35B NVFP4 (host B)", "qwen3.6-35b-nvfp4-gpu-host-d"),
    ("27b", "Qwen3.8-27B NVFP4", "qwen3.8-27b-nvfp4"),
    ("flash", "Qwen3.8-Flash-Next NVFP4", "qwen3.8-flash-next-nvfp4"),
]

# Sysprompt configs (label, display) -- host A and host B kept separate on purpose.
SYSPROMPT_CONFIGS = [
    ("qwen3.6-35b-nvfp4", "3.6-35B (A)"),
    ("qwen3.6-35b-nvfp4-gpu-host-d", "3.6-35B (B)"),
    ("qwen3.8-27b-nvfp4", "3.8-27B"),
    ("qwen3.8-flash-next-nvfp4", "Flash-Next"),
]

# 6 categories from the legacy PROMPT_NAMES mapping (spec F8c).
PROMPT_CATEGORY = {}
for _pid in [1, 2]:
    PROMPT_CATEGORY[_pid] = "Baseline"
for _pid in [3, 4, 7, 12, 16, 19]:
    PROMPT_CATEGORY[_pid] = "Anti-Drift"
for _pid in [6, 8, 13, 14]:
    PROMPT_CATEGORY[_pid] = "Constrained"
for _pid in [9, 10, 15, 20]:
    PROMPT_CATEGORY[_pid] = "Creative"
for _pid in [5, 11, 18]:
    PROMPT_CATEGORY[_pid] = "VL-Leveraged"
for _pid in [17]:
    PROMPT_CATEGORY[_pid] = "Thinking"
CATEGORY_ORDER = ["Baseline", "Anti-Drift", "Constrained", "Creative",
                  "VL-Leveraged", "Thinking"]

META_MARKERS = ["next agent", "here are", "here is", "paraphrase", "option 1",
                "**option", "rephrase", "version 1", "let me know"]
GENRE_MARKERS = ["subject:", "hi ", "dear ", "[your", "memo", "attached",
                 "action item", "to:"]
EXCLUSION_MARKERS = ["by others", "exclusion"]
NUMERIC_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")

# All initial prompts, keyed by the recipe/domain token used in condition keys.
ALL_PROMPTS: dict[str, str] = {}
ALL_PROMPTS.update(PROMPTS)
ALL_PROMPTS.update(CONCRETE_PROMPTS)
ALL_PROMPTS.update(DOMAIN_PROMPTS)

PAPER_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

ANOMALIES: list[str] = []


def anomaly(msg: str) -> None:
    ANOMALIES.append(msg)
    print(f"  ANOMALY: {msg}")


# =============================================================================
# 2. STATISTICS HELPERS
# =============================================================================

def bootstrap_ci(values, stat_fn=np.mean, n_boot: int = BOOTSTRAP_N,
                 alpha: float = 0.05):
    """(point, ci_low, ci_high) via percentile bootstrap; deterministic (seed 0)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None, None
    point = float(stat_fn(arr))
    if arr.size < 2:
        return point, point, point
    idx = RNG.integers(0, arr.size, size=(n_boot, arr.size))
    boot = stat_fn(arr[idx], axis=1)
    return (point,
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def mean_ci(values) -> dict:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "std": None}
    point, lo, hi = bootstrap_ci(vals)
    return {"mean": point, "ci_low": lo, "ci_high": hi, "n": len(vals),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)), "max": float(np.max(vals))}


def bootstrap_ci_curves(curves, n_boot: int = 2000, alpha: float = 0.05):
    """(mean_curve, lo_curve, hi_curve) aligned to the shortest curve."""
    min_len = min(len(c) for c in curves)
    aligned = np.array([np.asarray(c[:min_len], dtype=float) for c in curves])
    mean_curve = aligned.mean(axis=0)
    n = aligned.shape[0]
    if n < 2:
        return mean_curve, mean_curve, mean_curve
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot = aligned[idx].mean(axis=1)
    return (mean_curve,
            np.percentile(boot, 100 * alpha / 2, axis=0),
            np.percentile(boot, 100 * (1 - alpha / 2), axis=0))


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, (centre - half) / denom)),
            float(min(1.0, (centre + half) / denom)))


def rate_stat(flags) -> dict:
    flags = [bool(f) for f in flags]
    n = len(flags)
    k = sum(flags)
    lo, hi = wilson_ci(k, n)
    return {"rate": (k / n) if n else None, "k": k, "n": n,
            "ci_low": lo if n else None, "ci_high": hi if n else None}


def mwu(a, b) -> dict:
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if len(a) < 2 or len(b) < 2:
        return {"u": None, "p": None, "n1": len(a), "n2": len(b)}
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"u": float(u), "p": float(p), "n1": len(a), "n2": len(b)}


def _spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(pearsonr(rx, ry)[0])


def spearman_ci(x, y, n_boot: int = 2000) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return {"rho": None, "ci_low": None, "ci_high": None, "n": n, "p": None}
    rho = _spearman(x, y)
    # two-sided p from the t-approximation on the rank correlation
    p = None
    if n > 3 and np.isfinite(rho) and abs(rho) < 1.0:
        from scipy.stats import t as _t
        tstat = rho * np.sqrt((n - 2) / (1 - rho ** 2))
        p = float(2 * _t.sf(abs(tstat), n - 2))
    elif np.isfinite(rho):
        p = 0.0
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        r = _spearman(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(r)
    if len(boots) < 20:
        return {"rho": rho, "ci_low": None, "ci_high": None, "n": n, "p": p}
    return {"rho": rho, "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": n, "p": p}


# =============================================================================
# 3. PAPER A chain classification (imported logic from analyze_qwen35_paper.py)
# =============================================================================

def _is_empty_output(rec: dict) -> bool:
    return not (rec.get("output_message") or "").strip()


def _is_padding(rec: dict) -> bool:
    return _is_empty_output(rec) and not (rec.get("elapsed_seconds") or 0)


def classify_chain(iterations: list[dict], min_iters: int = ITERATIONS) -> dict:
    """clean / runaway / incomplete -- verbatim port of Paper A's classify_chain."""
    succ = [r for r in iterations
            if r.get("status") == "success" and r.get("cosine_similarity") is not None]
    last_ok = bool(iterations) and iterations[-1].get("status") == "success"
    info = {"n_success": len(succ), "runaway_onset": None,
            "n_runaway_events": 0, "n_padding": 0}
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
    info["runaway_onset"] = (real_empty[0] if real_empty else any_empty[0]) + 1
    return info


def paper_a_finals(models: dict, key: str, clean_only: bool = True) -> list[float]:
    """Final similarities for one Paper A condition, optionally clean chains only.

    Paper A key layouts: `<model>` / `lasagna_<model>` (single chain per key in the
    pilot) and `<model>_s<seed>` / `lasagna_<model>_s<seed>` for seeded runs.
    """
    finals = []
    pat = re.compile(rf"^{re.escape(key)}(_s\d+)?$")
    for k, iters in models.items():
        if not pat.match(k):
            continue
        info = classify_chain(iters)
        if info["status"] == "incomplete":
            continue
        if clean_only and info["status"] != "clean":
            continue
        sims = [r["cosine_similarity"] for r in iters
                if r.get("status") == "success" and r.get("cosine_similarity") is not None]
        finals.append(float(sims[ITERATIONS - 1]))
    return finals


# =============================================================================
# 4. PAPER B CHAIN REGISTRY
# =============================================================================

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _md5(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


class Chain:
    """One telephone chain, reduced to the fields the analysis needs."""

    __slots__ = ("label", "stem", "key", "seed", "recipe", "pid", "thinking",
                 "temp", "sims", "hashes", "final_text", "words", "ctokens",
                 "rtokens_est", "rtokens_src", "finish", "n_empty", "n_length",
                 "n_iter", "locked_end", "lock_onset", "inflation", "meta",
                 "numeric_survival", "exclusions_kept", "genre_drift",
                 "anchor_survival", "anchor_by_type")

    def __init__(self, label, stem, key, seed, recipe, pid, thinking, temp, iters):
        self.label, self.stem, self.key = label, stem, key
        self.seed, self.recipe, self.pid = seed, recipe, pid
        self.thinking, self.temp = thinking, temp
        self.n_iter = len(iters)

        self.sims, self.hashes, self.words = [], [], []
        self.ctokens, self.rtokens_est, self.finish = [], [], []
        self.n_empty = self.n_length = 0
        srcs = set()
        outs = []
        for r in iters:
            out = r.get("output_message") or ""
            outs.append(out)
            self.hashes.append(_md5(out))
            sim = r.get("cosine_similarity")
            self.sims.append(float(sim) if sim is not None else float("nan"))
            self.words.append(r.get("output_words") if r.get("output_words") is not None
                              else len(out.split()))
            ct = r.get("completion_tokens")
            self.ctokens.append(ct)
            fr = r.get("finish_reason")
            self.finish.append(fr)
            if not out.strip():
                self.n_empty += 1
            if fr == "length":
                self.n_length += 1
            rt = r.get("reasoning_tokens")
            if rt is not None:
                self.rtokens_est.append(float(rt))
                srcs.add("reported")
            elif ct is not None:
                # spec fallback: completion_tokens minus tokens of the output;
                # output tokens estimated at 4 characters per token.
                self.rtokens_est.append(max(0.0, float(ct) - len(out) / 4.0))
                srcs.add("estimated")
            else:
                self.rtokens_est.append(float("nan"))
        self.rtokens_src = ("reported" if srcs == {"reported"}
                            else "estimated" if srcs == {"estimated"}
                            else "mixed" if srcs else "none")

        self.final_text = outs[-1] if outs else ""
        # locking
        self.locked_end = (len(self.hashes) >= 3
                           and self.hashes[-1] == self.hashes[-2] == self.hashes[-3])
        onset = None
        if self.locked_end:
            i = len(self.hashes) - 1
            while i > 0 and self.hashes[i - 1] == self.hashes[-1]:
                i -= 1
            onset = i + 1  # 1-based
        self.lock_onset = onset

        prompt = ALL_PROMPTS.get(recipe)
        pw = len(prompt.split()) if prompt else None
        fw = self.words[-1] if self.words else 0
        self.inflation = (fw / pw) if pw else None

        final_norm = _norm(self.final_text)
        self.meta = any(m in final_norm for m in META_MARKERS)

        # concrete-domain metrics
        self.numeric_survival = None
        self.exclusions_kept = None
        self.genre_drift = None
        if prompt is not None and recipe in CONCRETE_PROMPTS:
            toks = NUMERIC_RE.findall(prompt)
            uniq = sorted(set(t.lower() for t in toks))
            if uniq:
                self.numeric_survival = sum(1 for t in uniq if t in final_norm) / len(uniq)
            self.exclusions_kept = any(m in final_norm for m in EXCLUSION_MARKERS)
            self.genre_drift = any(m in final_norm for m in GENRE_MARKERS)

        # breadth anchors
        self.anchor_survival = None
        self.anchor_by_type = None
        if recipe in ANCHORS:
            facts = ANCHORS[recipe]
            hits = [(ft, _norm(sub) in final_norm) for ft, sub in facts]
            self.anchor_survival = sum(1 for _, h in hits if h) / len(hits)
            by_type = defaultdict(list)
            for ft, h in hits:
                by_type[ft].append(h)
            self.anchor_by_type = {ft: sum(v) / len(v) for ft, v in by_type.items()}

    @property
    def final_sim(self) -> float:
        return self.sims[ITERATIONS - 1] if len(self.sims) >= ITERATIONS else self.sims[-1]

    @property
    def last_sim(self) -> float:
        return self.sims[-1]


_PID_SLUGS = {pid: _slug(name) for pid, name in PROMPT_NAMES.items()}


def parse_key(label: str, key: str):
    """(recipe, pid, thinking, temp, seed) or None if the key does not parse."""
    if not key.startswith(label + "_"):
        return None
    rest = key[len(label) + 1:]
    m = re.search(r"_s(\d+)$", rest)
    if not m:
        return None
    seed = int(m.group(1))
    rest = rest[:m.start()]
    thinking = False
    if rest.endswith("_think"):
        thinking = True
        rest = rest[:-len("_think")]
    pid = None
    temp = 0.7
    m = re.search(r"_(\d{2})_(.+)$", rest)
    if m and int(m.group(1)) in _PID_SLUGS and _PID_SLUGS[int(m.group(1))] == m.group(2):
        pid = int(m.group(1))
        rest = rest[:m.start()]
    else:
        m = re.search(r"_t(\d+\.\d+)$", rest)
        if m:
            temp = float(m.group(1))
            rest = rest[:m.start()]
    if rest not in ALL_PROMPTS:
        return None
    return rest, pid, thinking, temp, seed


CHAINS: list[Chain] = []
FILE_SUMMARY: list[dict] = []


def load_all() -> None:
    files = sorted(DATA_ROOT.glob("*/*.json"))
    print(f"Loading {len(files)} Paper B data files ...")
    for path in files:
        label = path.parent.name
        stem = path.stem
        if label not in DISPLAY:
            anomaly(f"unknown label directory {label}")
            continue
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            anomaly(f"{label}/{stem}.json does not parse: {exc}")
            continue
        models = raw.get("models", {})
        n_ok = n_bad = 0
        recs = 0
        for key, iters in models.items():
            parsed = parse_key(label, key)
            if parsed is None:
                anomaly(f"unparsable condition key in {label}/{stem}.json: {key}")
                n_bad += 1
                continue
            recipe, pid, thinking, temp, seed = parsed
            if any(r.get("status") != "success" for r in iters):
                anomaly(f"non-success iteration in {label}/{stem}.json key {key}")
            if len(iters) not in (30, 100):
                anomaly(f"{label}/{stem}.json key {key} has {len(iters)} iterations")
            # thinking is also implied by the file stem for the *_think experiments
            CHAINS.append(Chain(label, stem, key, seed, recipe, pid, thinking, temp, iters))
            n_ok += 1
            recs += len(iters)
        FILE_SUMMARY.append({"label": label, "stem": stem, "chains": n_ok,
                             "unparsed": n_bad, "inferences": recs})
    print(f"  {len(CHAINS)} chains, "
          f"{sum(f['inferences'] for f in FILE_SUMMARY)} model calls")


def sel(labels=None, stems=None, recipe=None, recipes=None, pid=None,
        thinking=None, temp=None, seeds=None, n_iter=None) -> list[Chain]:
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(stems, str):
        stems = [stems]
    if recipe is not None:
        recipes = [recipe]
    out = []
    for c in CHAINS:
        if labels is not None and c.label not in labels:
            continue
        if stems is not None and c.stem not in stems:
            continue
        if recipes is not None and c.recipe not in recipes:
            continue
        if pid is not None and c.pid != pid:
            continue
        if pid is None and thinking is not None and c.pid is not None:
            continue
        if thinking is not None and c.thinking != thinking:
            continue
        if temp is not None and abs(c.temp - temp) > 1e-9:
            continue
        if seeds is not None and c.seed not in seeds:
            continue
        if n_iter is not None and c.n_iter != n_iter:
            continue
        out.append(c)
    return out


def baseline_chains(labels, recipe) -> list[Chain]:
    """Thinking-OFF, T=0.7, no system prompt, 30-iteration baseline chains."""
    return [c for c in sel(labels=labels, stems="baseline", recipe=recipe,
                           thinking=False, temp=0.7, n_iter=30) if c.pid is None]


def finals(chains) -> list[float]:
    return [c.final_sim for c in chains]


# =============================================================================
# 5. OUTPUT HELPERS
# =============================================================================

def escape_latex(text: str) -> str:
    text = str(text)
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("_", r"\_")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    return text


def write_table(name: str, body: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / name).write_text(body)
    print(f"  {name} -> {TABLE_DIR}")


def save_figure(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def empty_panel(ax, title: str) -> None:
    ax.set_title(f"{title}\n(no data)", fontsize=8)
    ax.text(0.5, 0.5, "no data available", ha="center", va="center",
            transform=ax.transAxes, fontsize=8, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# Lab hostnames must never appear in an output artefact; the deployment suffix in
# the raw condition keys is rewritten to the neutral host A / host B naming.
_HOSTNAME_SUBS = [("gpu-host-d", "hostB"), ("gpu-host-e", "hostC"), ("gpu-host-f", "hostA"),
                  ("gpu-host-b", "hostD")]


def sanitize(text: str) -> str:
    for bad, good in _HOSTNAME_SUBS:
        text = text.replace(bad, good)
    return text


def numpy_to_python(obj):
    if isinstance(obj, str):
        return sanitize(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return numpy_to_python(obj.tolist())
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {sanitize(str(k)): numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    return obj


def fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    return f"{v:.{nd}f}"


def fmt_ci(st: dict, nd: int = 3) -> str:
    """Bracketed CI for either a mean_ci dict or a spearman_ci dict."""
    if not st:
        return "---"
    point = st.get("mean", st.get("rho"))
    if point is None or st.get("ci_low") is None or st.get("ci_high") is None:
        return "---"
    return f"[{st['ci_low']:.{nd}f}, {st['ci_high']:.{nd}f}]"


def fmt_p(p) -> str:
    if p is None:
        return "---"
    if p < 1e-4:
        return "$<$0.0001"
    return f"{p:.4f}"


def errbars(st: dict):
    """Asymmetric yerr array for one mean_ci dict."""
    if st["mean"] is None:
        return None
    return np.array([[st["mean"] - st["ci_low"]], [st["ci_high"] - st["mean"]]])


STATS: dict = {}


# =============================================================================
# 6. INTEGRITY CHECK (empty outputs / finish_reason == length)
# =============================================================================

def compute_integrity() -> dict:
    total_calls = sum(len(c.sims) for c in CHAINS)
    empty_chains = [c for c in CHAINS if c.n_empty]
    length_chains = [c for c in CHAINS if c.n_length]
    n_empty_calls = sum(c.n_empty for c in CHAINS)
    n_length_calls = sum(c.n_length for c in CHAINS)

    def detail(chs, attr):
        return [{"label": c.label, "stem": c.stem, "key": c.key,
                 "thinking": bool(c.thinking or "think" in c.stem),
                 "count": getattr(c, attr)} for c in chs]

    off_empty = [c for c in empty_chains if not (c.thinking or "think" in c.stem)]
    off_length = [c for c in length_chains if not (c.thinking or "think" in c.stem)]
    rate_all = rate_stat([bool(c.n_empty) for c in CHAINS])
    off_chains = [c for c in CHAINS if not (c.thinking or "think" in c.stem)]
    rate_off = rate_stat([bool(c.n_empty) for c in off_chains])
    out = {
        "n_chains": len(CHAINS),
        "n_calls": total_calls,
        "n_empty_calls": n_empty_calls,
        "n_empty_chains": len(empty_chains),
        "n_length_calls": n_length_calls,
        "n_length_chains": len(length_chains),
        "runaway_rate_all": rate_all,
        "runaway_rate_thinking_off": rate_off,
        "n_chains_thinking_off": len(off_chains),
        "empty_chain_detail": detail(empty_chains, "n_empty"),
        "length_chain_detail": detail(length_chains, "n_length"),
        "thinking_off_empty_chains": len(off_empty),
        "thinking_off_length_chains": len(off_length),
    }
    if n_empty_calls:
        anomaly(f"{n_empty_calls} empty outputs across {len(empty_chains)} chains "
                f"(spec expected zero); all in thinking-ON conditions: "
                f"{len(off_empty) == 0}")
    if n_length_calls:
        anomaly(f"{n_length_calls} iterations with finish_reason=='length' across "
                f"{len(length_chains)} chains (spec expected zero); all in "
                f"thinking-ON conditions: {len(off_length) == 0}")
    return out


# =============================================================================
# 7. F1 / T1  BASELINE
# =============================================================================

def compute_baseline() -> dict:
    out = {}
    for cid, disp, labs in CONFIGS:
        for rec in RECIPES:
            chs = baseline_chains(labs, rec)
            if not chs:
                continue
            st = mean_ci(finals(chs))
            st["locked"] = rate_stat([c.locked_end for c in chs])
            st["lock_onsets"] = [c.lock_onset for c in chs if c.lock_onset]
            st["inflation"] = mean_ci([c.inflation for c in chs])
            st["meta"] = rate_stat([c.meta for c in chs])
            st["words"] = mean_ci([c.words[ITERATIONS - 1] for c in chs])
            st["finals"] = finals(chs)
            out[f"{cid}|{rec}"] = st
    # MTP as its own config (used by F6/F12)
    cid, disp, labs = MTP_CONFIG
    for rec in RECIPES:
        chs = baseline_chains(labs, rec)
        if not chs:
            continue
        st = mean_ci(finals(chs))
        st["locked"] = rate_stat([c.locked_end for c in chs])
        st["lock_onsets"] = [c.lock_onset for c in chs if c.lock_onset]
        st["inflation"] = mean_ci([c.inflation for c in chs])
        st["meta"] = rate_stat([c.meta for c in chs])
        st["words"] = mean_ci([c.words[ITERATIONS - 1] for c in chs])
        st["finals"] = finals(chs)
        out[f"{cid}|{rec}"] = st
    return out


def plot_f1(bstats: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rec in zip(axes, RECIPES):
        any_data = False
        for cid, disp, labs in CONFIGS:
            chs = baseline_chains(labs, rec)
            if not chs:
                continue
            any_data = True
            curves = [np.array(c.sims[:ITERATIONS]) for c in chs]
            mean_c, lo, hi = bootstrap_ci_curves(curves)
            x = np.arange(1, len(mean_c) + 1)
            col = CONFIG_COLORS[cid]
            ax.plot(x, mean_c, color=col, lw=1.6, label=f"{disp} (n={len(chs)})")
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        if not any_data:
            empty_panel(ax, RECIPE_DISPLAY[rec])
            continue
        ax.set_title(f"{RECIPE_DISPLAY[rec]} -- baseline, T=0.7, thinking OFF", fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine similarity to original")
        ax.set_ylim(0.0, 1.02)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("F1. Baseline drift trajectories (mean $\\pm$ bootstrap 95% CI)",
                 fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig1_baseline_trajectories.png")


def table1(bstats: dict) -> None:
    rows = []
    for cid, disp, _ in CONFIGS + [MTP_CONFIG]:
        for rec in RECIPES:
            st = bstats.get(f"{cid}|{rec}")
            if not st:
                continue
            rows.append(
                f"{escape_latex(disp)} & {RECIPE_DISPLAY[rec].split(' ')[0]} & "
                f"{fmt(st['mean'])} & {fmt_ci(st)} & {st['n']} & "
                f"{fmt(st['locked']['rate'], 2)} & {fmt(st['inflation']['mean'], 2)} & "
                f"{fmt(st['meta']['rate'], 2)} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Baseline drift at iteration 30 (T=0.7, no system prompt, thinking OFF).
Means over independent chains with percentile bootstrap 95\% CIs ($B$=10{,}000).
Locked = last three outputs byte-identical; inflation = final output words divided
by prompt words; meta = final output contains agent-handoff / meta-commentary
markers. The two Qwen3.6-35B NVFP4 hosts are pooled.}
\label{tab:paper3-baseline}
\begin{tabular}{llrlrrrr}
\toprule
Configuration & Recipe & Mean & 95\% CI & $n$ & Locked & Inflation & Meta \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table1_baseline.tex", body)


# =============================================================================
# 8. F2 / T2  CROSS-GENERATION
# =============================================================================

def compute_cross_gen() -> dict:
    pilot = json.loads(PAPER_A_QWEN3_PILOT.read_text()).get("models", {})
    gguf = json.loads(PAPER_A_QWEN35_GGUF.read_text()).get("models", {})
    nvfp4 = json.loads(PAPER_A_QWEN35_NVFP4.read_text()).get("models", {})

    def pa(models, model_key, rec, clean=True):
        key = model_key if rec == "spaghetti" else f"lasagna_{model_key}"
        return paper_a_finals(models, key, clean_only=clean)

    groups = {
        "27B dense": [
            ("Qwen3-32B (pilot)", lambda r: pa(pilot, "qwen3:32b-q4_K_M", r, clean=False)),
            ("Qwen3.5-27B GGUF", lambda r: pa(gguf, "qwen3.5:27b-q4_K_M", r)),
            ("Qwen3.6-27B GGUF", lambda r: finals(baseline_chains(["qwen3.6-27b-q4kxl"], r))),
            ("Qwen3.8-27B GGUF", lambda r: finals(baseline_chains(["qwen3.8-27b-q4kxl"], r))),
            ("Qwen3.8-27B NVFP4", lambda r: finals(baseline_chains(["qwen3.8-27b-nvfp4"], r))),
        ],
        "35B-class MoE": [
            ("Qwen3-30B MoE (pilot)",
             lambda r: pa(pilot, "qwen3:30b-a3b-instruct-2507-q4_K_M", r, clean=False)),
            ("Qwen3.5-35B GGUF", lambda r: pa(gguf, "qwen3.5:35b-q4_K_M", r)),
            ("Qwen3.5-35B NVFP4", lambda r: pa(nvfp4, "qwen3.5:35b-nvfp4", r)),
            ("Qwen3.6-35B GGUF", lambda r: finals(baseline_chains(["qwen3.6-35b-q4kxl"], r))),
            ("Qwen3.6-35B NVFP4", lambda r: finals(baseline_chains(POOLED_35B, r))),
        ],
        "122B-class": [
            ("Qwen3.5-122B GGUF", lambda r: pa(gguf, "qwen3.5:122b-q4_K_M", r)),
            ("Qwen3.8-Flash-Next NVFP4",
             lambda r: finals(baseline_chains(["qwen3.8-flash-next-nvfp4"], r))),
        ],
    }
    out = {}
    for gname, entries in groups.items():
        out[gname] = []
        for name, fn in entries:
            item = {"name": name}
            for rec in RECIPES:
                vals = fn(rec)
                item[rec] = mean_ci(vals)
            if item["spaghetti"]["n"] == 0 and item["lasagna"]["n"] == 0:
                anomaly(f"cross-generation entry '{name}' has no usable chains")
            out[gname].append(item)
    return out


def plot_f2(cg: dict) -> None:
    gnames = list(cg)
    fig, axes = plt.subplots(2, len(gnames), figsize=(13, 7), squeeze=False)
    for j, gname in enumerate(gnames):
        for i, rec in enumerate(RECIPES):
            ax = axes[i][j]
            entries = cg[gname]
            xs, means, errs, labels, ns, tops = [], [], [], [], [], []
            for k, e in enumerate(entries):
                st = e[rec]
                labels.append(e["name"])
                ns.append(st["n"])
                if st["mean"] is None:
                    continue
                xs.append(k)
                means.append(st["mean"])
                errs.append([st["mean"] - st["ci_low"], st["ci_high"] - st["mean"]])
                tops.append(st["ci_high"])
            if not xs:
                empty_panel(ax, f"{gname} -- {RECIPE_DISPLAY[rec]}")
                continue
            err = np.array(errs).T
            ax.bar(xs, means, yerr=err, capsize=3, color="#4c72b0", alpha=0.85,
                   error_kw={"lw": 1})
            for x, top, n in zip(xs, tops, [ns[k] for k in xs]):
                ax.text(x, min(top + 0.03, 1.0), f"n={n}", ha="center", fontsize=6.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)],
                               rotation=30, ha="right", fontsize=6.5)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Final similarity" if j == 0 else "")
            ax.set_title(f"{gname} -- {RECIPE_DISPLAY[rec]}", fontsize=9)
    fig.suptitle("F2. Cross-generation comparison at matched size "
                 "(Paper A chains: clean only)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig2_cross_generation.png")


def table2(cg: dict) -> None:
    rows = []
    for gname, entries in cg.items():
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{escape_latex(gname)}}}}} \\")
        for e in entries:
            s, l = e["spaghetti"], e["lasagna"]
            rows.append(
                f"\\quad {escape_latex(e['name'])} & {fmt(s['mean'])} & {fmt_ci(s)} & "
                f"{s['n']} & {fmt(l['mean'])} & {fmt_ci(l)} \\\\")
        rows.append(r"\addlinespace")
    body = r"""\begin{table}[t]
\centering
\caption{Cross-generation drift at matched model size. Paper~A rows use clean chains
only (no empty output in any successful iteration); the Qwen3 pilot is a single
unseeded chain per model. Bootstrap 95\% CIs, $B$=10{,}000.}
\label{tab:paper3-crossgen}
\begin{tabular}{lrlrrl}
\toprule
Model & \multicolumn{3}{c}{Spaghetti} & \multicolumn{2}{c}{Lasagna} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
 & Mean & 95\% CI & $n$ & Mean & 95\% CI \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table2_cross_generation.tex", body)


# =============================================================================
# 9. F3 / T3  DETERMINISM
# =============================================================================

def _pair_chains(a_chains, b_chains):
    """Match two chain lists on (recipe, seed)."""
    ai = {(c.recipe, c.seed): c for c in a_chains}
    bi = {(c.recipe, c.seed): c for c in b_chains}
    keys = sorted(set(ai) & set(bi))
    return [(ai[k], bi[k]) for k in keys]


def _pair_stats(pairs, name):
    if not pairs:
        return {"name": name, "n_pairs": 0}
    identical_iter1 = sum(1 for a, b in pairs if a.hashes[0] == b.hashes[0])
    ident_counts = [sum(1 for x, y in zip(a.hashes[:ITERATIONS], b.hashes[:ITERATIONS])
                        if x == y) for a, b in pairs]
    deltas = [abs(a.final_sim - b.final_sim) for a, b in pairs]
    return {
        "name": name,
        "n_pairs": len(pairs),
        "identical_at_iter1": identical_iter1,
        "identical_at_iter1_rate": rate_stat([a.hashes[0] == b.hashes[0]
                                              for a, b in pairs]),
        "mean_identical_iterations": float(np.mean(ident_counts)),
        "max_identical_iterations": int(np.max(ident_counts)),
        "abs_delta_final": mean_ci(deltas),
        "max_abs_delta_final": float(np.max(deltas)),
    }


def compute_determinism() -> dict:
    run1_all = sel(labels="qwen3.6-35b-nvfp4", stems="baseline", thinking=False,
                   temp=0.7, recipes=RECIPES, n_iter=30)
    run1_all = [c for c in run1_all if c.pid is None]
    rep2 = sel(labels="qwen3.6-35b-nvfp4", stems="baseline_rep2", n_iter=30)
    hostB = [c for c in sel(labels="qwen3.6-35b-nvfp4-gpu-host-d", stems="baseline",
                            thinking=False, temp=0.7, recipes=RECIPES, n_iter=30)
             if c.pid is None]
    rep_seeds = sorted({c.seed for c in rep2})
    run1 = [c for c in run1_all if c.seed in rep_seeds]
    hostB_m = [c for c in hostB if c.seed in rep_seeds]

    w1a = sel(labels="qwen3.6-35b-nvfp4", stems="baseline_w1a")
    w1b = sel(labels="qwen3.6-35b-nvfp4", stems="baseline_w1b")

    out = {
        "rep_seeds": rep_seeds,
        "within_host": _pair_stats(_pair_chains(run1, rep2), "host A run1 vs run2"),
        "cross_host": _pair_stats(_pair_chains(run1, hostB_m), "host A run1 vs host B"),
        "single_worker": _pair_stats(_pair_chains(w1a, w1b),
                                     "host A workers=1 run a vs run b"),
        "per_seed": {},
        "t0_vs_t07": {},
        "mwu_host_a_vs_b": {},
    }
    for name, chs in [("run1", run1), ("run2", rep2), ("hostB", hostB_m)]:
        out["per_seed"][name] = [
            {"recipe": c.recipe, "seed": c.seed, "final": c.final_sim} for c in
            sorted(chs, key=lambda c: (c.recipe, c.seed))]
    out["per_seed"]["w1a"] = [{"recipe": c.recipe, "seed": c.seed, "final": c.final_sim}
                              for c in sorted(w1a, key=lambda c: c.seed)]
    out["per_seed"]["w1b"] = [{"recipe": c.recipe, "seed": c.seed, "final": c.final_sim}
                              for c in sorted(w1b, key=lambda c: c.seed)]

    gpu = [("qwen3.6-35b-nvfp4-gpu-host-d", "Qwen3.6-35B NVFP4 (host B)"),
           ("qwen3.8-27b-nvfp4", "Qwen3.8-27B NVFP4"),
           ("qwen3.8-flash-next-nvfp4", "Qwen3.8-Flash-Next NVFP4")]
    for lab, disp in gpu:
        for rec in RECIPES:
            t0 = [c for c in sel(labels=lab, stems="temperature", recipe=rec,
                                 temp=0.0, thinking=False) if c.pid is None]
            t07 = baseline_chains([lab], rec)
            out["t0_vs_t07"][f"{lab}|{rec}"] = {
                "display": disp, "recipe": rec,
                "t0": mean_ci(finals(t0)), "t07": mean_ci(finals(t07)),
                "mwu": mwu(finals(t0), finals(t07)),
            }
    for rec in RECIPES:
        a = finals(baseline_chains(["qwen3.6-35b-nvfp4"], rec))
        b = finals(baseline_chains(["qwen3.6-35b-nvfp4-gpu-host-d"], rec))
        out["mwu_host_a_vs_b"][rec] = {
            "host_a": mean_ci(a), "host_b": mean_ci(b), "mwu": mwu(a, b)}
    return out


def plot_f3(det: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) per-seed dot plot run1 / run2 / hostB
    ax = axes[0]
    series = [("run1", "host A run 1", "#1f77b4", "o"),
              ("run2", "host A run 2", "#ff7f0e", "s"),
              ("hostB", "host B", "#2ca02c", "^")]
    base = det["per_seed"].get("run1", [])
    if base:
        labels = [f"{d['recipe'][:4]} s{d['seed']}" for d in base]
        xs = np.arange(len(base))
        for k, disp, col, mk in series:
            data = det["per_seed"].get(k, [])
            idx = {(d["recipe"], d["seed"]): d["final"] for d in data}
            ys = [idx.get((d["recipe"], d["seed"]), np.nan) for d in base]
            ax.plot(xs, ys, mk, color=col, label=disp, ms=6, alpha=0.85, ls="none")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        ax.set_ylabel("Final similarity")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.set_title("(a) Same seed, repeated runs and hosts", fontsize=9)
    else:
        empty_panel(ax, "(a) repeated runs")

    # (b) single-worker
    ax = axes[1]
    a = det["per_seed"].get("w1a", [])
    b = det["per_seed"].get("w1b", [])
    if a and b:
        bi = {(d["recipe"], d["seed"]): d["final"] for d in b}
        xs = np.arange(len(a))
        ax.plot(xs, [d["final"] for d in a], "o", color="#1f77b4", label="workers=1 run a")
        ax.plot(xs, [bi.get((d["recipe"], d["seed"]), np.nan) for d in a], "s",
                color="#d62728", label="workers=1 run b")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"s{d['seed']}" for d in a], rotation=45, ha="right",
                           fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Final similarity")
        ax.legend(fontsize=7)
        sw = det["single_worker"]
        ax.set_title(f"(b) Single-worker serving (spaghetti)\n"
                     f"{sw.get('identical_at_iter1', 0)}/{sw.get('n_pairs', 0)} identical "
                     f"at iteration 1", fontsize=9)
    else:
        empty_panel(ax, "(b) single-worker control")

    # (c) T=0 vs T=0.7 distributions
    ax = axes[2]
    entries = [(k, v) for k, v in det["t0_vs_t07"].items()
               if v["t0"]["n"] or v["t07"]["n"]]
    if entries:
        pos = 0
        ticks, ticklabels, ns = [], [], set()
        for k, v in entries:
            for temp_key, col in (("t0", "#4c72b0"), ("t07", "#dd8452")):
                st = v[temp_key]
                if st["n"]:
                    ax.errorbar([pos], [st["mean"]], yerr=errbars(st), fmt="o",
                                color=col, capsize=3, ms=5)
                    ns.add(st["n"])
                pos += 0.6
            ticks.append(pos - 0.9)
            ticklabels.append(f"{v['display'].replace(' NVFP4', '')}\n{v['recipe'][:4]}")
            pos += 0.6
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=40, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Final similarity")
        ax.legend(handles=[Line2D([], [], color="#4c72b0", marker="o", ls="none",
                                  label="T=0.0"),
                           Line2D([], [], color="#dd8452", marker="o", ls="none",
                                  label="T=0.7")], fontsize=7)
        nlab = (f"n={sorted(ns)[0]} per point" if len(ns) == 1
                else "n=" + "/".join(str(x) for x in sorted(ns)) + " per point")
        ax.set_title("(c) Greedy decoding is not deterministic\n"
                     f"(T=0 sample distributions vs T=0.7, {nlab})", fontsize=9)
    else:
        empty_panel(ax, "(c) T=0 vs T=0.7")

    fig.suptitle("F3. Seeds do not pin chains: run-to-run and host-to-host "
                 "reproducibility", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig3_determinism.png")


def table3(det: dict) -> None:
    rows = []
    for key in ["within_host", "cross_host", "single_worker"]:
        d = det[key]
        if not d.get("n_pairs"):
            continue
        rows.append(
            f"{escape_latex(d['name'])} & {d['n_pairs']} & "
            f"{d['identical_at_iter1']}/{d['n_pairs']} & "
            f"{fmt(d['mean_identical_iterations'], 2)} & "
            f"{fmt(d['abs_delta_final']['mean'])} & "
            f"{fmt(d['max_abs_delta_final'])} \\\\")
    mrows = []
    for rec, d in det["mwu_host_a_vs_b"].items():
        mrows.append(
            f"{rec} & {fmt(d['host_a']['mean'])} ({d['host_a']['n']}) & "
            f"{fmt(d['host_b']['mean'])} ({d['host_b']['n']}) & "
            f"{fmt_p(d['mwu']['p'])} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Seed reproducibility. Chains repeated with identical seed, prompt and
deployment diverge immediately; ``identical at iter.\ 1'' counts pairs whose first
outputs are byte-identical, and $|\Delta|$ is the absolute difference in final
similarity. The lower block compares the two Qwen3.6-35B NVFP4 hosts as
distributions (Mann--Whitney U, $n$=15 per host per recipe).}
\label{tab:paper3-determinism}
\begin{tabular}{lrrrrr}
\toprule
Comparison & pairs & identical at iter.\ 1 & mean identical iters & mean $|\Delta|$ & max $|\Delta|$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{1em}
\begin{tabular}{lrrr}
\toprule
Recipe & host A mean ($n$) & host B mean ($n$) & MWU $p$ \\
\midrule
""" + "\n".join(mrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table3_determinism.tex", body)


# =============================================================================
# 10. F4 / T4  THINKING
# =============================================================================

def compute_thinking() -> dict:
    out = {"per_config": {}, "reasoning_source": {}, "long_horizon": {}}
    for cid, disp, lab in THINKING_CONFIGS:
        for rec in RECIPES:
            off = baseline_chains([lab], rec)
            on = [c for c in sel(labels=lab, stems="thinking", recipe=rec,
                                 thinking=True, n_iter=30) if c.pid is None]
            if not off and not on:
                continue
            rt = [np.nanmean(c.rtokens_est) for c in on] if on else []
            srcs = {c.rtokens_src for c in on}
            out["per_config"][f"{cid}|{rec}"] = {
                "display": disp, "recipe": rec,
                "off": mean_ci(finals(off)), "on": mean_ci(finals(on)),
                "mwu": mwu(finals(off), finals(on)),
                "delta": (float(np.mean(finals(on))) - float(np.mean(finals(off))))
                if on and off else None,
                "reasoning_tokens_per_iter": mean_ci(rt),
                "reasoning_tokens_all": [float(x) for c in on for x in c.rtokens_est
                                         if np.isfinite(x)],
                "reasoning_source": "/".join(sorted(srcs)) if srcs else "none",
                "locked_off": rate_stat([c.locked_end for c in off]),
                "locked_on": rate_stat([c.locked_end for c in on]),
            }
            out["reasoning_source"][f"{cid}|{rec}"] = "/".join(sorted(srcs))
    # 100-iteration trajectories
    for cid, disp, lab in THINKING_CONFIGS:
        for rec in RECIPES:
            off = [c for c in sel(labels=lab, stems="baseline_100iter", recipe=rec,
                                  n_iter=100) if c.pid is None]
            on = [c for c in sel(labels=lab, stems="thinking_100iter", recipe=rec,
                                 n_iter=100) if c.pid is None]
            if not off and not on:
                continue
            entry = {"display": disp, "recipe": rec}
            for arm, chs in (("off", off), ("on", on)):
                if not chs:
                    entry[arm] = {"n": 0}
                    continue
                s30 = [c.sims[29] for c in chs]
                s100 = [c.sims[99] for c in chs]
                mins = [float(np.nanmin(c.sims[29:])) for c in chs]
                dips = [(c.sims[29] - float(np.nanmin(c.sims[29:]))) > 0.05 for c in chs]
                entry[arm] = {
                    "n": len(chs),
                    "at30": mean_ci(s30), "at100": mean_ci(s100),
                    "delta_30_100": float(np.mean(s100) - np.mean(s30)),
                    "min_after_30": mean_ci(mins),
                    "dip_rate": rate_stat(dips),
                }
            out["long_horizon"][f"{cid}|{rec}"] = entry
    return out


_SRC_TAG = {"reported": "rep", "estimated": "est",
            "estimated/reported": "mix", "none": "n/a"}


def plot_f4(th: dict) -> None:
    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    # (a) OFF vs ON final sims
    ax = fig.add_subplot(gs[0, 0])
    keys = list(th["per_config"])
    if keys:
        xs = np.arange(len(keys))
        for off_on, dx, col in (("off", -0.16, "#4c72b0"), ("on", 0.16, "#c44e52")):
            means, errs, valid = [], [], []
            for i, k in enumerate(keys):
                st = th["per_config"][k][off_on]
                if st["mean"] is None:
                    continue
                valid.append(i + dx)
                means.append(st["mean"])
                errs.append([st["mean"] - st["ci_low"], st["ci_high"] - st["mean"]])
            if valid:
                ax.bar(valid, means, width=0.3, yerr=np.array(errs).T, capsize=2,
                       color=col, label=f"thinking {off_on.upper()}",
                       error_kw={"lw": 0.8})
        for i, k in enumerate(keys):
            p = th["per_config"][k]["mwu"]["p"]
            if p is not None:
                ax.text(i, 1.0, f"p={p:.3g}", ha="center", fontsize=6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{th['per_config'][k]['display'].replace(' NVFP4','')}\n"
                            f"{th['per_config'][k]['recipe'][:4]}" for k in keys],
                           rotation=35, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Final similarity")
        ax.legend(fontsize=7)
        ax.set_title("(a) Thinking OFF vs ON (Mann--Whitney $p$)", fontsize=9)
    else:
        empty_panel(ax, "(a) thinking OFF vs ON")

    # (b) reasoning tokens per iteration
    ax = fig.add_subplot(gs[0, 1])
    data, labels = [], []
    for k in keys:
        v = th["per_config"][k]["reasoning_tokens_all"]
        if v:
            data.append(v)
            labels.append(f"{th['per_config'][k]['display'].replace(' NVFP4','')}\n"
                          f"{th['per_config'][k]['recipe'][:4]} "
                          f"[{_SRC_TAG.get(th['per_config'][k]['reasoning_source'], '?')}]")
    if data:
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#c44e52")
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
        ax.set_ylabel("Reasoning tokens per iteration")
        ax.set_yscale("log")
        ax.set_title("(b) Reasoning budget per iteration\n"
                     "(rep = reported, est = completion tokens $-$ output tokens)",
                     fontsize=9)
    else:
        empty_panel(ax, "(b) reasoning tokens")

    # (c) 100-iteration trajectories
    lh_keys = list(th["long_horizon"])
    ncol = max(1, len(lh_keys))
    sub = gs[1, :].subgridspec(1, ncol)
    if not lh_keys:
        ax = fig.add_subplot(gs[1, :])
        empty_panel(ax, "(c) 100-iteration trajectories")
    for j, k in enumerate(lh_keys):
        ax = fig.add_subplot(sub[0, j])
        v = th["long_horizon"][k]
        plotted = []
        for arm, col, lab in (("off", "#4c72b0", "OFF"), ("on", "#c44e52", "ON")):
            cid = k.split("|")[0]
            labname = dict((c[0], c[2]) for c in THINKING_CONFIGS)[cid]
            stem = "baseline_100iter" if arm == "off" else "thinking_100iter"
            chs = [c for c in sel(labels=labname, stems=stem, recipe=v["recipe"],
                                  n_iter=100) if c.pid is None]
            if not chs:
                continue
            mean_c, lo, hi = bootstrap_ci_curves([np.array(c.sims) for c in chs])
            x = np.arange(1, len(mean_c) + 1)
            ax.plot(x, mean_c, color=col, lw=1.3, label=f"{lab} (n={len(chs)})")
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
            plotted.append(arm)
        title = f"{v['display'].replace(' NVFP4','')} {v['recipe'][:4]}"
        if not plotted:
            empty_panel(ax, f"(c) {title}")
            continue
        if len(plotted) == 1:
            title += f"\n(only thinking {plotted[0].upper()} run to 100 iters)"
        ax.axvline(30, color="#888888", ls=":", lw=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Iteration")
        if j == 0:
            ax.set_ylabel("Cosine similarity")
        ax.legend(fontsize=6)
        ax.set_title(title, fontsize=8)
    fig.suptitle("F4. Thinking mode: effect on drift, reasoning budget and long horizon",
                 fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig4_thinking.png")


def table4(th: dict) -> None:
    rows = []
    for k, v in th["per_config"].items():
        rows.append(
            f"{escape_latex(v['display'])} & {v['recipe']} & "
            f"{fmt(v['off']['mean'])} ({v['off']['n']}) & "
            f"{fmt(v['on']['mean'])} ({v['on']['n']}) & "
            f"{fmt(v['delta'])} & {fmt_p(v['mwu']['p'])} & "
            f"{fmt(v['reasoning_tokens_per_iter']['mean'], 0)} & "
            f"{escape_latex(v['reasoning_source'])} \\\\")
    lrows = []
    for k, v in th["long_horizon"].items():
        for arm in ("off", "on"):
            a = v[arm]
            if not a.get("n"):
                continue
            lrows.append(
                f"{escape_latex(v['display'])} & {v['recipe']} & {arm.upper()} & "
                f"{a['n']} & {fmt(a['at30']['mean'])} & {fmt(a['at100']['mean'])} & "
                f"{fmt(a['delta_30_100'])} & {fmt(a['dip_rate']['rate'], 2)} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Thinking mode. Upper block: final similarity with thinking OFF and ON
(mean with $n$ chains), the ON$-$OFF difference, Mann--Whitney $p$, and the mean
reasoning tokens per iteration. ``rep'' = reasoning token counts reported by the
server; ``est'' = completion tokens minus output tokens (4 characters per token).
Lower block: 100-iteration runs, similarity at iteration 30 and 100 and the fraction
of chains dipping more than 0.05 below their iteration-30 value.}
\label{tab:paper3-thinking}
\begin{tabular}{llrrrrrl}
\toprule
Configuration & Recipe & OFF ($n$) & ON ($n$) & $\Delta$ & $p$ & Reas.\ tok/iter & Source \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{1em}
\begin{tabular}{lllrrrrr}
\toprule
Configuration & Recipe & Think & $n$ & @30 & @100 & $\Delta_{30\to100}$ & dip rate \\
\midrule
""" + "\n".join(lrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table4_thinking.tex", body)


# =============================================================================
# 11. F5 / T5  FORMAT / STACK, F6 MTP
# =============================================================================

FORMAT_PAIRS = [
    ("Qwen3.6-35B", POOLED_35B, ["qwen3.6-35b-q4kxl"]),
    ("Qwen3.8-27B", ["qwen3.8-27b-nvfp4"], ["qwen3.8-27b-q4kxl"]),
]


def compute_format() -> dict:
    out = {}
    for name, nv_labs, gg_labs in FORMAT_PAIRS:
        for rec in RECIPES:
            nv = finals(baseline_chains(nv_labs, rec))
            gg = finals(baseline_chains(gg_labs, rec))
            out[f"{name}|{rec}"] = {
                "model": name, "recipe": rec,
                "nvfp4": mean_ci(nv), "gguf": mean_ci(gg), "mwu": mwu(nv, gg),
                "delta": (float(np.mean(nv)) - float(np.mean(gg))) if nv and gg else None,
            }
    out["_note"] = ("Qwen3.8-Flash-Next has no GGUF Q4_K_XL run; that comparison "
                    "is omitted as the spec directs.")
    return out


def plot_f5(fmt_stats: dict) -> None:
    keys = [k for k in fmt_stats if k != "_note"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if keys:
        xs = np.arange(len(keys))
        for arm, dx, col, lab in (("nvfp4", -0.17, "#4c72b0", "NVFP4 (vLLM/SGLang)"),
                                  ("gguf", 0.17, "#55a868", "GGUF Q4\\_K\\_XL (llama.cpp)")):
            valid, means, errs = [], [], []
            for i, k in enumerate(keys):
                st = fmt_stats[k][arm]
                if st["mean"] is None:
                    continue
                valid.append(i + dx)
                means.append(st["mean"])
                errs.append([st["mean"] - st["ci_low"], st["ci_high"] - st["mean"]])
            if valid:
                ax.bar(valid, means, width=0.32, yerr=np.array(errs).T, capsize=3,
                       color=col, label=lab.replace("\\_", "_"), error_kw={"lw": 0.9})
        for i, k in enumerate(keys):
            p = fmt_stats[k]["mwu"]["p"]
            if p is not None:
                ax.text(i, 1.02, f"p={p:.3g}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{fmt_stats[k]['model']}\n{fmt_stats[k]['recipe']}"
                            for k in keys], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Final similarity")
        ax.legend(fontsize=8)
        ax.set_title("F5. Serving format / stack: NVFP4 vs GGUF Q4_K_XL\n"
                     "(Flash-Next has no GGUF run and is omitted)", fontsize=10)
    else:
        empty_panel(ax, "F5. format / stack")
    fig.tight_layout()
    save_figure(fig, "fig5_format_stack.png")


def table5(fmt_stats: dict) -> None:
    rows = []
    for k, v in fmt_stats.items():
        if k == "_note":
            continue
        rows.append(
            f"{escape_latex(v['model'])} & {v['recipe']} & "
            f"{fmt(v['nvfp4']['mean'])} {fmt_ci(v['nvfp4'])} & {v['nvfp4']['n']} & "
            f"{fmt(v['gguf']['mean'])} {fmt_ci(v['gguf'])} & {v['gguf']['n']} & "
            f"{fmt(v['delta'])} & {fmt_p(v['mwu']['p'])} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Serving format and stack. Same weights family, two quantisation formats and
two inference stacks (NVFP4 on vLLM/SGLang vs GGUF Q4\_K\_XL on llama.cpp).
Qwen3.8-Flash-Next has no GGUF run and is omitted.}
\label{tab:paper3-format}
\begin{tabular}{llrrrrrr}
\toprule
Model & Recipe & NVFP4 mean [CI] & $n$ & GGUF mean [CI] & $n$ & $\Delta$ & MWU $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table5_format_stack.tex", body)


def compute_mtp() -> dict:
    out = {}
    for rec in RECIPES:
        no = baseline_chains(["qwen3.8-27b-q4kxl"], rec)
        yes = baseline_chains(["qwen3.8-27b-q4kxl-mtp"], rec)
        out[rec] = {
            "no_mtp": mean_ci(finals(no)), "mtp": mean_ci(finals(yes)),
            "mwu": mwu(finals(no), finals(yes)),
            "no_mtp_words": mean_ci([c.words[ITERATIONS - 1] for c in no]),
            "mtp_words": mean_ci([c.words[ITERATIONS - 1] for c in yes]),
            "no_mtp_finals": finals(no), "mtp_finals": finals(yes),
        }
    return out


def plot_f6(mtp: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    keys = [r for r in RECIPES if mtp[r]["no_mtp"]["n"] or mtp[r]["mtp"]["n"]]
    if keys:
        xs = np.arange(len(keys))
        for arm, dx, col, lab in (("no_mtp", -0.17, "#ff7f0e", "GGUF"),
                                  ("mtp", 0.17, "#7f7f7f", "GGUF + MTP")):
            valid, means, errs = [], [], []
            for i, r in enumerate(keys):
                st = mtp[r][arm]
                if st["mean"] is None:
                    continue
                valid.append(i + dx)
                means.append(st["mean"])
                errs.append([st["mean"] - st["ci_low"], st["ci_high"] - st["mean"]])
            ax.bar(valid, means, width=0.32, yerr=np.array(errs).T, capsize=3,
                   color=col, label=lab, error_kw={"lw": 0.9})
        for i, r in enumerate(keys):
            p = mtp[r]["mwu"]["p"]
            if p is not None:
                ax.text(i, 1.02, f"p={p:.3g}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([RECIPE_DISPLAY[r] for r in keys], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Final similarity")
        ax.legend(fontsize=8)
        ax.set_title("(a) Qwen3.8-27B GGUF, speculative decoding off/on", fontsize=9)
    else:
        empty_panel(ax, "(a) MTP means")

    ax = axes[1]
    plotted = False
    for i, r in enumerate(RECIPES):
        for arm, dx, col in (("no_mtp_finals", -0.12, "#ff7f0e"),
                             ("mtp_finals", 0.12, "#7f7f7f")):
            vals = mtp[r][arm]
            if not vals:
                continue
            plotted = True
            jitter = (RNG.random(len(vals)) - 0.5) * 0.12
            ax.plot(np.full(len(vals), i + dx) + jitter, vals, "o", color=col,
                    ms=4, alpha=0.75)
    if plotted:
        ax.set_xticks(range(len(RECIPES)))
        ax.set_xticklabels([RECIPE_DISPLAY[r] for r in RECIPES], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Final similarity (per chain)")
        ax.legend(handles=[Line2D([], [], color="#ff7f0e", marker="o", ls="none",
                                  label="GGUF"),
                           Line2D([], [], color="#7f7f7f", marker="o", ls="none",
                                  label="GGUF + MTP")], fontsize=7)
        ax.set_title("(b) Per-chain strip plot", fontsize=9)
    else:
        empty_panel(ax, "(b) per-chain strip plot")
    fig.suptitle("F6. Multi-token prediction (speculative decoding)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig6_mtp.png")


# =============================================================================
# 12. F7  TEMPERATURE
# =============================================================================

TEMPS = [0.0, 0.3, 0.5, 0.7, 1.0]
TEMP_CONFIGS = [
    ("qwen3.6-35b-nvfp4", "Qwen3.6-35B NVFP4 (host A)", "#1f77b4"),
    ("qwen3.6-35b-nvfp4-gpu-host-d", "Qwen3.6-35B NVFP4 (host B)", "#17becf"),
    ("qwen3.8-27b-nvfp4", "Qwen3.8-27B NVFP4", "#d62728"),
    ("qwen3.8-flash-next-nvfp4", "Qwen3.8-Flash-Next NVFP4", "#9467bd"),
]


def compute_temperature() -> dict:
    out = {}
    for lab, disp, _ in TEMP_CONFIGS:
        for rec in RECIPES:
            series = {}
            for t in TEMPS:
                if abs(t - 0.7) < 1e-9:
                    chs = baseline_chains([lab], rec)
                else:
                    chs = [c for c in sel(labels=lab, stems="temperature", recipe=rec,
                                          temp=t, thinking=False) if c.pid is None]
                series[f"{t:.1f}"] = mean_ci(finals(chs))
            out[f"{lab}|{rec}"] = {"display": disp, "recipe": rec, "series": series}
    return out


def plot_f7(temp: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rec in zip(axes, RECIPES):
        any_data = False
        for lab, disp, col in TEMP_CONFIGS:
            v = temp.get(f"{lab}|{rec}")
            if not v:
                continue
            xs, ys, errs, ns = [], [], [], []
            for t in TEMPS:
                st = v["series"][f"{t:.1f}"]
                if st["mean"] is None:
                    continue
                xs.append(t)
                ys.append(st["mean"])
                errs.append([st["mean"] - st["ci_low"], st["ci_high"] - st["mean"]])
                ns.append(st["n"])
            if not xs:
                continue
            any_data = True
            ax.errorbar(xs, ys, yerr=np.array(errs).T, marker="o", color=col,
                        capsize=2, lw=1.3, ms=4, label=disp)
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(n), (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=5.5, color=col)
        if not any_data:
            empty_panel(ax, RECIPE_DISPLAY[rec])
            continue
        ax.set_xlabel("Sampling temperature")
        ax.set_ylabel("Final similarity")
        ax.set_ylim(0, 1.05)
        ax.set_title(RECIPE_DISPLAY[rec], fontsize=10)
        ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("F7. Temperature sweep (point labels give $n$ chains)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig7_temperature.png")


# =============================================================================
# 13. F8 / T6  SYSTEM PROMPTS
# =============================================================================

def compute_sysprompt() -> dict:
    per = {}
    for lab, disp in SYSPROMPT_CONFIGS:
        for rec in RECIPES:
            prompts = {}
            for pid in sorted(PROMPT_NAMES):
                chs = sel(labels=lab, stems="sysprompt", recipe=rec, pid=pid,
                          thinking=False, n_iter=30)
                if not chs:
                    continue
                st = mean_ci(finals(chs))
                st["locked"] = rate_stat([c.locked_end for c in chs])
                st["n_chains"] = len(chs)
                prompts[pid] = st
            if not prompts:
                continue
            means = {pid: v["mean"] for pid, v in prompts.items() if v["mean"] is not None}
            best = max(means, key=means.get)
            worst = min(means, key=means.get)
            per[f"{lab}|{rec}"] = {
                "display": disp, "label": lab, "recipe": rec, "prompts": prompts,
                "n_prompts": len(prompts),
                "chains_per_prompt": sorted({v["n_chains"] for v in prompts.values()}),
                "n_chains_total": sum(v["n_chains"] for v in prompts.values()),
                "spread": means[best] - means[worst],
                "best_pid": best, "best_name": PROMPT_NAMES[best], "best": means[best],
                "worst_pid": worst, "worst_name": PROMPT_NAMES[worst],
                "worst": means[worst],
            }
    # Spearman matrix across config x recipe
    keys = list(per)
    rho_matrix = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            common = sorted(set(per[a]["prompts"]) & set(per[b]["prompts"]))
            if len(common) < 3:
                continue
            xs = [per[a]["prompts"][p]["mean"] for p in common]
            ys = [per[b]["prompts"][p]["mean"] for p in common]
            rho_matrix[f"{a}||{b}"] = spearman_ci(xs, ys)
    # categories
    cats = {}
    for k, v in per.items():
        row = {}
        for cat in CATEGORY_ORDER:
            vals = [v["prompts"][p]["mean"] for p in v["prompts"]
                    if PROMPT_CATEGORY.get(p) == cat and v["prompts"][p]["mean"] is not None]
            row[cat] = float(np.mean(vals)) if vals else None
        cats[k] = row
    # thinking x sysprompt on host A lasagna
    think_sys = {}
    for pid in sorted(PROMPT_NAMES):
        off = sel(labels="qwen3.6-35b-nvfp4", stems="sysprompt", recipe="lasagna",
                  pid=pid, thinking=False)
        on = sel(labels="qwen3.6-35b-nvfp4", stems="sysprompt_think", recipe="lasagna",
                 pid=pid, thinking=True)
        if not off and not on:
            continue
        think_sys[pid] = {"name": PROMPT_NAMES[pid],
                          "off": mean_ci(finals(off)), "on": mean_ci(finals(on))}
    ts_rho = None
    pairs = [(v["off"]["mean"], v["on"]["mean"]) for v in think_sys.values()
             if v["off"]["mean"] is not None and v["on"]["mean"] is not None]
    if len(pairs) >= 3:
        ts_rho = spearman_ci([p[0] for p in pairs], [p[1] for p in pairs])
    deltas = [p[1] - p[0] for p in pairs]
    return {"per_config": per, "rho_matrix": rho_matrix, "categories": cats,
            "thinking_sysprompt": think_sys,
            "thinking_sysprompt_rho": ts_rho,
            "thinking_sysprompt_mean_delta": float(np.mean(deltas)) if deltas else None,
            "thinking_sysprompt_n_worse": int(sum(1 for d in deltas if d < 0)),
            "thinking_sysprompt_n": len(deltas)}


def plot_f8(sp: dict) -> None:
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0])

    # (a) per-prompt means
    ax = fig.add_subplot(gs[0, :])
    keys = list(sp["per_config"])
    if keys:
        pids = sorted(PROMPT_NAMES)
        markers = {"spaghetti": "o", "lasagna": "s"}
        colors = {"qwen3.6-35b-nvfp4": "#1f77b4",
                  "qwen3.6-35b-nvfp4-gpu-host-d": "#17becf",
                  "qwen3.8-27b-nvfp4": "#d62728",
                  "qwen3.8-flash-next-nvfp4": "#9467bd"}
        for k in keys:
            v = sp["per_config"][k]
            xs = [p for p in pids if p in v["prompts"]]
            ys = [v["prompts"][p]["mean"] for p in xs]
            ax.plot(xs, ys, markers[v["recipe"]], color=colors[v["label"]], ms=4.5,
                    alpha=0.8, ls="none",
                    label=f"{v['display']} {v['recipe'][:4]} (n$_p$={len(xs)})")
        ax.set_xticks(pids)
        ax.set_xticklabels([f"{p:02d} {PROMPT_NAMES[p]}" for p in pids],
                           rotation=55, ha="right", fontsize=6.5)
        ax.set_ylabel("Final similarity")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, ncol=4, loc="lower left")
        npc = sorted({n for k in keys
                      for n in sp["per_config"][k]["chains_per_prompt"]})
        ax.set_title("(a) Per-prompt mean final similarity "
                     f"({'/'.join(str(x) for x in npc)} chains per point)", fontsize=9)
    else:
        empty_panel(ax, "(a) per-prompt means")

    # (b) Spearman matrix
    ax = fig.add_subplot(gs[1, 0])
    if keys and sp["rho_matrix"]:
        n = len(keys)
        M = np.full((n, n), np.nan)
        for i in range(n):
            M[i, i] = 1.0
        for pair, v in sp["rho_matrix"].items():
            a, b = pair.split("||")
            i, j = keys.index(a), keys.index(b)
            M[i, j] = M[j, i] = v["rho"] if v["rho"] is not None else np.nan
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=-1, vmax=1)
        short = [f"{sp['per_config'][k]['display']} {sp['per_config'][k]['recipe'][:4]}"
                 for k in keys]
        ax.set_xticks(range(n))
        ax.set_xticklabels(short, rotation=55, ha="right", fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(short, fontsize=6)
        for i in range(n):
            for j in range(n):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045)
        ax.set_title("(b) Spearman $\\rho$ of prompt rankings", fontsize=9)
    else:
        empty_panel(ax, "(b) Spearman matrix")

    # (c) category heatmap
    ax = fig.add_subplot(gs[1, 1])
    if sp["categories"]:
        ck = list(sp["categories"])
        M = np.array([[sp["categories"][k][c] if sp["categories"][k][c] is not None
                       else np.nan for c in CATEGORY_ORDER] for k in ck])
        im = ax.imshow(M, cmap="viridis", vmin=np.nanmin(M), vmax=np.nanmax(M))
        ax.set_xticks(range(len(CATEGORY_ORDER)))
        ax.set_xticklabels(CATEGORY_ORDER, rotation=35, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(ck)))
        ax.set_yticklabels([f"{sp['per_config'][k]['display']} "
                            f"{sp['per_config'][k]['recipe'][:4]}" for k in ck],
                           fontsize=6)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.5, color="white")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045)
        ax.set_title("(c) Mean final similarity by prompt category", fontsize=9)
    else:
        empty_panel(ax, "(c) category heatmap")

    # (d) thinking x sysprompt
    ax = fig.add_subplot(gs[2, :])
    ts = sp["thinking_sysprompt"]
    if ts:
        pids = sorted(ts)
        off = [ts[p]["off"]["mean"] for p in pids]
        on = [ts[p]["on"]["mean"] for p in pids]
        ax.plot(pids, off, "o-", color="#4c72b0", ms=4, lw=1, label="thinking OFF")
        ax.plot(pids, on, "s-", color="#c44e52", ms=4, lw=1, label="thinking ON")
        ax.set_xticks(pids)
        ax.set_xticklabels([f"{p:02d} {PROMPT_NAMES[p]}" for p in pids],
                           rotation=55, ha="right", fontsize=6.5)
        ax.set_ylabel("Final similarity")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        d = sp["thinking_sysprompt_mean_delta"]
        rho = sp["thinking_sysprompt_rho"]
        ax.set_title(f"(d) Qwen3.6-35B (host A) lasagna: system prompt $\\times$ "
                     f"thinking (mean $\\Delta$={fmt(d)}, "
                     f"{sp['thinking_sysprompt_n_worse']}/{sp['thinking_sysprompt_n']} "
                     f"prompts worse, rank $\\rho$="
                     f"{fmt(rho['rho'] if rho else None, 2)})", fontsize=9)
    else:
        empty_panel(ax, "(d) thinking x sysprompt")

    fig.suptitle("F8. System prompts dominate drift in the Qwen3.6/3.8 generations",
                 fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig8_sysprompt.png")


def table6(sp: dict) -> None:
    rows = []
    for k, v in sp["per_config"].items():
        rows.append(
            f"{escape_latex(v['display'])} & {v['recipe']} & {v['n_prompts']} & "
            f"{'/'.join(str(x) for x in v['chains_per_prompt'])} & "
            f"{fmt(v['spread'])} & {escape_latex(v['best_name'])} ({fmt(v['best'])}) & "
            f"{escape_latex(v['worst_name'])} ({fmt(v['worst'])}) \\\\")
    rrows = []
    for pair, v in sp["rho_matrix"].items():
        a, b = pair.split("||")
        da = sp["per_config"][a]
        db = sp["per_config"][b]
        rrows.append(
            f"{escape_latex(da['display'])} {da['recipe'][:4]} vs "
            f"{escape_latex(db['display'])} {db['recipe'][:4]} & {v['n']} & "
            f"{fmt(v['rho'], 2)} & [{fmt(v['ci_low'], 2)}, {fmt(v['ci_high'], 2)}] & "
            f"{fmt_p(v['p'])} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{System-prompt sensitivity. Spread is the best minus the worst prompt mean
for that configuration and recipe; ``chains'' is the number of independent chains
behind each prompt mean. The lower block gives the
Spearman rank correlation of prompt orderings between every pair of
configuration$\times$recipe cells, with bootstrap 95\% CIs.}
\label{tab:paper3-sysprompt}
\begin{tabular}{llrrrll}
\toprule
Configuration & Recipe & prompts & chains & Spread & Best prompt & Worst prompt \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{1em}
\begin{tabular}{lrrlr}
\toprule
Pair & $n$ prompts & $\rho$ & 95\% CI & $p$ \\
\midrule
""" + "\n".join(rrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table6_sysprompt.tex", body)


# =============================================================================
# 14. F9  LOCKING
# =============================================================================

def compute_locking() -> dict:
    out = {"baseline": {}, "thinking": {}}
    for cid, disp, labs in CONFIGS + [MTP_CONFIG]:
        for rec in RECIPES:
            chs = baseline_chains(labs, rec)
            if not chs:
                continue
            out["baseline"][f"{cid}|{rec}"] = {
                "display": disp, "recipe": rec,
                "locked": rate_stat([c.locked_end for c in chs]),
                "onsets": [c.lock_onset for c in chs if c.lock_onset],
                "onset_stat": mean_ci([c.lock_onset for c in chs if c.lock_onset]),
            }
    for cid, disp, lab in THINKING_CONFIGS:
        for rec in RECIPES:
            chs = [c for c in sel(labels=lab, stems="thinking", recipe=rec,
                                  thinking=True, n_iter=30) if c.pid is None]
            if not chs:
                continue
            out["thinking"][f"{cid}|{rec}"] = {
                "display": disp, "recipe": rec,
                "locked": rate_stat([c.locked_end for c in chs]),
                "onsets": [c.lock_onset for c in chs if c.lock_onset],
                "onset_stat": mean_ci([c.lock_onset for c in chs if c.lock_onset]),
            }
    return out


def plot_f9(lock: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    keys = list(lock["baseline"])
    tkeys = list(lock["thinking"])
    if keys or tkeys:
        allk = [("OFF", k) for k in keys] + [("ON", k) for k in tkeys]
        xs = np.arange(len(allk))
        vals, errs, cols = [], [], []
        for arm, k in allk:
            st = lock["OFF" == arm and "baseline" or "thinking"][k]["locked"]
            vals.append(st["rate"])
            errs.append([st["rate"] - st["ci_low"], st["ci_high"] - st["rate"]])
            cols.append("#4c72b0" if arm == "OFF" else "#c44e52")
        ax.bar(xs, vals, yerr=np.array(errs).T, color=cols, capsize=2,
               error_kw={"lw": 0.8})
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{lock['baseline' if a == 'OFF' else 'thinking'][k]['display'].replace(' NVFP4','')}"
             f" {k.split('|')[1][:4]} {a}" for a, k in allk],
            rotation=55, ha="right", fontsize=6)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Locked-at-end rate")
        ax.legend(handles=[Line2D([], [], color="#4c72b0", lw=6, label="thinking OFF"),
                           Line2D([], [], color="#c44e52", lw=6, label="thinking ON")],
                  fontsize=7)
        ax.set_title("(a) Fraction of chains whose last three outputs are identical\n"
                     "(Wilson 95% CI)", fontsize=9)
    else:
        empty_panel(ax, "(a) locking rate")

    ax = axes[1]
    data, labels = [], []
    for src, arm in (("baseline", "OFF"), ("thinking", "ON")):
        for k, v in lock[src].items():
            if v["onsets"]:
                data.append(v["onsets"])
                labels.append(f"{v['display'].replace(' NVFP4','')} "
                              f"{v['recipe'][:4]} {arm}")
    if data:
        bp = ax.boxplot(data, showfliers=True, patch_artist=True, vert=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#55a868")
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
        ax.set_ylabel("Lock onset (iteration)")
        ax.set_title("(b) Onset of the final byte-identical run", fontsize=9)
    else:
        empty_panel(ax, "(b) lock onset")
    fig.suptitle("F9. Byte-level locking", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig9_locking.png")


# =============================================================================
# 15. F10 / T7  CONCRETE
# =============================================================================

CONCRETE_CONFIGS = [
    ("35b", "Qwen3.6-35B NVFP4", POOLED_35B),
    ("27b", "Qwen3.8-27B NVFP4", ["qwen3.8-27b-nvfp4"]),
    ("flash", "Qwen3.8-Flash-Next NVFP4", ["qwen3.8-flash-next-nvfp4"]),
]


def _concrete_block(chs) -> dict:
    if not chs:
        return {"n": 0}
    return {
        "n": len(chs),
        "sim": mean_ci(finals(chs)),
        "numeric": mean_ci([c.numeric_survival for c in chs]),
        "inflation": mean_ci([c.inflation for c in chs]),
        "exclusions": rate_stat([c.exclusions_kept for c in chs]),
        "genre": rate_stat([c.genre_drift for c in chs]),
        "meta": rate_stat([c.meta for c in chs]),
        "locked": rate_stat([c.locked_end for c in chs]),
    }


def compute_concrete() -> dict:
    out = {"baseline": {}, "thinking": {}, "sysprompt": {}, "scatter": {}}
    for cid, disp, labs in CONCRETE_CONFIGS:
        for rec in CONCRETE_RECIPES:
            chs = [c for c in sel(labels=labs, stems="baseline_concrete", recipe=rec,
                                  thinking=False) if c.pid is None]
            b = _concrete_block(chs)
            if b["n"]:
                b.update({"display": disp, "recipe": rec})
                out["baseline"][f"{cid}|{rec}"] = b
    for cid, disp, labs in CONCRETE_CONFIGS:
        for rec in CONCRETE_RECIPES:
            on = [c for c in sel(labels=labs, stems="thinking_concrete", recipe=rec,
                                 thinking=True) if c.pid is None]
            off = [c for c in sel(labels=labs, stems="baseline_concrete", recipe=rec,
                                  thinking=False) if c.pid is None]
            if not on:
                continue
            out["thinking"][f"{cid}|{rec}"] = {
                "display": disp, "recipe": rec,
                "off": _concrete_block(off), "on": _concrete_block(on),
                "mwu": mwu(finals(off), finals(on)),
            }
    for rec in CONCRETE_RECIPES:
        for pid in DOMAIN_NEUTRAL_SYSPROMPT_IDS:
            chs = sel(labels="qwen3.6-35b-nvfp4-gpu-host-d", stems="sysprompt_concrete",
                      recipe=rec, pid=pid)
            if not chs:
                continue
            b = _concrete_block(chs)
            b.update({"pid": pid, "name": PROMPT_NAMES[pid], "recipe": rec,
                      "display": "Qwen3.6-35B NVFP4 (host B)"})
            out["sysprompt"][f"{rec}|{pid:02d}"] = b
    # scatter: every concrete chain with a numeric-survival score
    pts = [c for c in CHAINS if c.recipe in CONCRETE_PROMPTS
           and c.numeric_survival is not None]
    sims = [c.final_sim for c in pts]
    nums = [c.numeric_survival for c in pts]
    out["scatter"] = {
        "n": len(pts),
        "rho": spearman_ci(sims, nums) if len(pts) >= 3 else None,
        "sims": sims, "numeric": nums,
        "kinds": [("thinking" if (c.thinking or "think" in c.stem) else
                   "sysprompt" if c.pid is not None else "baseline") for c in pts],
        "recipes": [c.recipe for c in pts],
    }
    # baseline-only rho (the Phase-1 headline used baseline chains)
    bpts = [c for c in pts if c.pid is None and not (c.thinking or "think" in c.stem)]
    out["scatter"]["rho_baseline_only"] = (
        spearman_ci([c.final_sim for c in bpts], [c.numeric_survival for c in bpts])
        if len(bpts) >= 3 else None)
    out["scatter"]["n_baseline_only"] = len(bpts)
    return out


def plot_f10(con: dict) -> None:
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 3)

    # (a) baseline metrics per config x recipe
    ax = fig.add_subplot(gs[0, :2])
    keys = list(con["baseline"])
    if keys:
        metrics = [("sim", "Final similarity"), ("numeric", "Numeric survival"),
                   ("inflation", "Inflation (/3)")]
        rates = [("exclusions", "Exclusions kept"), ("genre", "Genre drift")]
        width = 0.15
        xs = np.arange(len(keys))
        for m, (mk, mlab) in enumerate(metrics):
            vals, errs = [], []
            for k in keys:
                st = con["baseline"][k][mk]
                v = st["mean"]
                if mk == "inflation" and v is not None:
                    v = v / 3.0
                vals.append(v if v is not None else np.nan)
                if st["mean"] is None:
                    errs.append([0, 0])
                else:
                    sc = 3.0 if mk == "inflation" else 1.0
                    errs.append([(st["mean"] - st["ci_low"]) / sc,
                                 (st["ci_high"] - st["mean"]) / sc])
            ax.bar(xs + (m - 2) * width, vals, width=width, yerr=np.array(errs).T,
                   capsize=2, label=mlab, error_kw={"lw": 0.7})
        for m, (rk, rlab) in enumerate(rates):
            vals = [con["baseline"][k][rk]["rate"] for k in keys]
            ax.bar(xs + (m + 1) * width, vals, width=width, label=rlab, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{con['baseline'][k]['display'].replace(' NVFP4','')}\n"
                            f"{CONCRETE_DISPLAY[con['baseline'][k]['recipe']]} "
                            f"(n={con['baseline'][k]['n']})" for k in keys],
                           rotation=25, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.25)
        ax.legend(fontsize=6.5, ncol=5, loc="upper left")
        ax.set_title("(a) Concrete SOW, thinking OFF (inflation divided by 3 to fit)",
                     fontsize=9)
    else:
        empty_panel(ax, "(a) concrete baseline")

    # (b) scatter sim vs numeric survival
    ax = fig.add_subplot(gs[0, 2])
    sc = con["scatter"]
    if sc["n"]:
        colmap = {"baseline": "#4c72b0", "thinking": "#c44e52", "sysprompt": "#55a868"}
        for kind in ["baseline", "thinking", "sysprompt"]:
            idx = [i for i, k in enumerate(sc["kinds"]) if k == kind]
            if not idx:
                continue
            ax.scatter([sc["sims"][i] for i in idx], [sc["numeric"][i] for i in idx],
                       s=14, alpha=0.65, color=colmap[kind], label=kind)
        r = sc["rho"]
        ax.set_xlabel("Final cosine similarity")
        ax.set_ylabel("Numeric-token survival")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=6.5, loc="lower left")
        ax.set_title(f"(b) Similarity vs numeric fidelity\n"
                     f"all chains n={sc['n']}, $\\rho$={fmt(r['rho'] if r else None, 2)}; "
                     f"baseline only n={sc['n_baseline_only']}, $\\rho$="
                     f"{fmt(sc['rho_baseline_only']['rho'] if sc['rho_baseline_only'] else None, 2)}",
                     fontsize=8)
    else:
        empty_panel(ax, "(b) similarity vs numeric survival")

    # (c) thinking OFF vs ON
    ax = fig.add_subplot(gs[1, 0])
    tkeys = list(con["thinking"])
    if tkeys:
        xs = np.arange(len(tkeys))
        for arm, dx, col in (("off", -0.17, "#4c72b0"), ("on", 0.17, "#c44e52")):
            vals, errs, valid = [], [], []
            for i, k in enumerate(tkeys):
                st = con["thinking"][k][arm]
                if not st.get("n"):
                    continue
                valid.append(i + dx)
                vals.append(st["sim"]["mean"])
                errs.append([st["sim"]["mean"] - st["sim"]["ci_low"],
                             st["sim"]["ci_high"] - st["sim"]["mean"]])
            if valid:
                ax.bar(valid, vals, width=0.32, yerr=np.array(errs).T, capsize=2,
                       color=col, label=f"thinking {arm.upper()}", error_kw={"lw": 0.7})
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{con['thinking'][k]['display'].replace(' NVFP4','')}\n"
                            f"{CONCRETE_DISPLAY[con['thinking'][k]['recipe']]}"
                            for k in tkeys], rotation=30, ha="right", fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Final similarity")
        ax.legend(fontsize=6.5)
        ax.set_title("(c) Concrete SOW: thinking OFF vs ON", fontsize=9)
    else:
        empty_panel(ax, "(c) concrete thinking")

    # (d) exclusions kept OFF vs ON
    ax = fig.add_subplot(gs[1, 1])
    if tkeys:
        xs = np.arange(len(tkeys))
        for arm, dx, col in (("off", -0.17, "#4c72b0"), ("on", 0.17, "#c44e52")):
            vals, valid = [], []
            for i, k in enumerate(tkeys):
                st = con["thinking"][k][arm]
                if not st.get("n"):
                    continue
                valid.append(i + dx)
                vals.append(st["exclusions"]["rate"])
            if valid:
                ax.bar(valid, vals, width=0.32, color=col,
                       label=f"thinking {arm.upper()}")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{con['thinking'][k]['display'].replace(' NVFP4','')}\n"
                            f"{CONCRETE_DISPLAY[con['thinking'][k]['recipe']]}"
                            for k in tkeys], rotation=30, ha="right", fontsize=6)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Exclusions-kept rate")
        ax.legend(fontsize=6.5)
        ax.set_title("(d) Survival of the ``by others'' exclusion line", fontsize=9)
    else:
        empty_panel(ax, "(d) exclusion survival")

    # (e) sysprompt subset on host B
    ax = fig.add_subplot(gs[1, 2])
    skeys = list(con["sysprompt"])
    if skeys:
        pids = sorted({con["sysprompt"][k]["pid"] for k in skeys})
        width = 0.35
        xs = np.arange(len(pids))
        for m, rec in enumerate(CONCRETE_RECIPES):
            vals, errs = [], []
            for p in pids:
                b = con["sysprompt"].get(f"{rec}|{p:02d}")
                if not b:
                    vals.append(np.nan)
                    errs.append([0, 0])
                    continue
                vals.append(b["sim"]["mean"])
                errs.append([b["sim"]["mean"] - b["sim"]["ci_low"],
                             b["sim"]["ci_high"] - b["sim"]["mean"]])
            ax.bar(xs + (m - 0.5) * width, vals, width=width, yerr=np.array(errs).T,
                   capsize=2, label=CONCRETE_DISPLAY[rec], error_kw={"lw": 0.7})
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p:02d} {PROMPT_NAMES[p]}" for p in pids],
                           rotation=55, ha="right", fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Final similarity")
        ax.legend(fontsize=6.5)
        ax.set_title("(e) Domain-neutral system prompts (host B)", fontsize=9)
    else:
        empty_panel(ax, "(e) concrete sysprompt subset")

    fig.suptitle("F10. Concrete statement-of-work domain: similarity vs factual fidelity",
                 fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig10_concrete.png")


def table7(con: dict) -> None:
    rows = []
    for k, b in con["baseline"].items():
        rows.append(
            f"{escape_latex(b['display'])} & {CONCRETE_DISPLAY[b['recipe']]} & OFF & "
            f"{b['n']} & {fmt(b['sim']['mean'])} & {fmt(b['numeric']['mean'], 2)} & "
            f"{fmt(b['inflation']['mean'], 2)} & {fmt(b['exclusions']['rate'], 2)} & "
            f"{fmt(b['genre']['rate'], 2)} \\\\")
    for k, v in con["thinking"].items():
        b = v["on"]
        rows.append(
            f"{escape_latex(v['display'])} & {CONCRETE_DISPLAY[v['recipe']]} & ON & "
            f"{b['n']} & {fmt(b['sim']['mean'])} & {fmt(b['numeric']['mean'], 2)} & "
            f"{fmt(b['inflation']['mean'], 2)} & {fmt(b['exclusions']['rate'], 2)} & "
            f"{fmt(b['genre']['rate'], 2)} \\\\")
    srows = []
    for k, b in con["sysprompt"].items():
        srows.append(
            f"{escape_latex(b['name'])} & {CONCRETE_DISPLAY[b['recipe']]} & {b['n']} & "
            f"{fmt(b['sim']['mean'])} & {fmt(b['numeric']['mean'], 2)} & "
            f"{fmt(b['inflation']['mean'], 2)} & {fmt(b['exclusions']['rate'], 2)} & "
            f"{fmt(b['genre']['rate'], 2)} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Construction statement-of-work domain. Numeric survival is the fraction of
the prompt's distinct numeric tokens still present in the iteration-30 output;
exclusions kept is the fraction of chains whose final output still contains the
``by others'' / ``exclusion'' language; genre drift is the fraction reframed as an
email or memo. The lower block is the domain-neutral system-prompt subset on host B.}
\label{tab:paper3-concrete}
\begin{tabular}{lllrrrrrr}
\toprule
Configuration & Document & Think & $n$ & Sim & Numeric & Inflation & Excl.\ kept & Genre drift \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{1em}
\begin{tabular}{llrrrrrr}
\toprule
System prompt & Document & $n$ & Sim & Numeric & Inflation & Excl.\ kept & Genre drift \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table7_concrete.tex", body)


# =============================================================================
# 16. F11 / T8 / T9  BREADTH
# =============================================================================

def _domain_block(chs) -> dict:
    if not chs:
        return {"n": 0}
    by_type = defaultdict(list)
    for c in chs:
        for ft, v in (c.anchor_by_type or {}).items():
            by_type[ft].append(v)
    return {
        "n": len(chs),
        "sim": mean_ci(finals(chs)),
        "anchors": mean_ci([c.anchor_survival for c in chs]),
        "inflation": mean_ci([c.inflation for c in chs]),
        "meta": rate_stat([c.meta for c in chs]),
        "locked": rate_stat([c.locked_end for c in chs]),
        "by_type": {ft: mean_ci(v) for ft, v in by_type.items()},
    }


def compute_breadth() -> dict:
    out = {"pooled_35b": {}, "flash": {}, "host_a": {}, "host_b": {},
           "thinking": {}, "fidelity": {}, "rank_rho": {}, "chain_scatter": {}}
    for dom in DOMAIN_ORDER:
        a = sel(labels="qwen3.6-35b-nvfp4", stems="baseline_breadth", recipe=dom)
        b = sel(labels="qwen3.6-35b-nvfp4-gpu-host-d", stems="baseline_breadth", recipe=dom)
        out["host_a"][dom] = _domain_block(a)
        out["host_b"][dom] = _domain_block(b)
        out["pooled_35b"][dom] = _domain_block(a + b)
        out["flash"][dom] = _domain_block(
            sel(labels="qwen3.8-flash-next-nvfp4", stems="baseline_breadth", recipe=dom))
        out["thinking"][dom] = _domain_block(
            sel(labels="qwen3.6-35b-nvfp4-gpu-host-d", stems="thinking_breadth", recipe=dom))
        out["fidelity"][dom] = _domain_block(
            sel(labels="qwen3.6-35b-nvfp4", stems="sysprompt_breadth", recipe=dom,
                pid=FIDELITY_PROMPT_ID))

    def series(src, field):
        return [out[src][d][field]["mean"] if out[src][d].get("n") else np.nan
                for d in DOMAIN_ORDER]

    def rho_between(s1, s2, field):
        x = series(s1, field)
        y = series(s2, field)
        pairs = [(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
        if len(pairs) < 3:
            return None
        return spearman_ci([p[0] for p in pairs], [p[1] for p in pairs])

    for f in ("sim", "anchors"):
        out["rank_rho"][f"host_a_vs_host_b|{f}"] = rho_between("host_a", "host_b", f)
        out["rank_rho"][f"pooled35b_vs_flash|{f}"] = rho_between("pooled_35b", "flash", f)
        out["rank_rho"][f"pooled35b_vs_thinking|{f}"] = rho_between(
            "pooled_35b", "thinking", f)
        out["rank_rho"][f"pooled35b_vs_fidelity|{f}"] = rho_between(
            "pooled_35b", "fidelity", f)

    # chain-level scatter across all breadth chains of the pooled 3.6-35B baseline
    chs = [c for c in CHAINS if c.recipe in ANCHORS and c.anchor_survival is not None]
    base = [c for c in chs if c.stem == "baseline_breadth" and c.label in POOLED_35B]
    out["chain_scatter"] = {
        "n": len(base),
        "sims": [c.final_sim for c in base],
        "anchors": [c.anchor_survival for c in base],
        "domains": [c.recipe for c in base],
        "rho": spearman_ci([c.final_sim for c in base],
                           [c.anchor_survival for c in base]) if len(base) >= 3 else None,
    }
    dom_sim = [out["pooled_35b"][d]["sim"]["mean"] for d in DOMAIN_ORDER
               if out["pooled_35b"][d].get("n")]
    dom_anc = [out["pooled_35b"][d]["anchors"]["mean"] for d in DOMAIN_ORDER
               if out["pooled_35b"][d].get("n")]
    out["domain_level_rho"] = (spearman_ci(dom_sim, dom_anc)
                               if len(dom_sim) >= 3 else None)
    # overall arms
    for name, (src_label, stem, pid) in {
        "overall_off": (POOLED_35B, "baseline_breadth", None),
        "overall_think": (["qwen3.6-35b-nvfp4-gpu-host-d"], "thinking_breadth", None),
        "overall_fidelity": (["qwen3.6-35b-nvfp4"], "sysprompt_breadth",
                             FIDELITY_PROMPT_ID),
        "overall_flash": (["qwen3.8-flash-next-nvfp4"], "baseline_breadth", None),
    }.items():
        chs = [c for c in sel(labels=src_label, stems=stem, recipes=DOMAIN_ORDER)
               if c.pid == pid]
        out[name] = {"n": len(chs), "sim": mean_ci(finals(chs)),
                     "anchors": mean_ci([c.anchor_survival for c in chs]),
                     "inflation": mean_ci([c.inflation for c in chs]),
                     "meta": rate_stat([c.meta for c in chs]),
                     "locked": rate_stat([c.locked_end for c in chs])}
    off = [c for c in sel(labels=["qwen3.6-35b-nvfp4-gpu-host-d"], stems="baseline_breadth",
                          recipes=DOMAIN_ORDER) if c.pid is None]
    on = [c for c in sel(labels=["qwen3.6-35b-nvfp4-gpu-host-d"], stems="thinking_breadth",
                         recipes=DOMAIN_ORDER) if c.pid is None]
    out["mwu_think"] = mwu(finals(off), finals(on))
    offA = [c for c in sel(labels=["qwen3.6-35b-nvfp4"], stems="baseline_breadth",
                           recipes=DOMAIN_ORDER) if c.pid is None]
    fid = [c for c in sel(labels=["qwen3.6-35b-nvfp4"], stems="sysprompt_breadth",
                          recipes=DOMAIN_ORDER) if c.pid == FIDELITY_PROMPT_ID]
    out["mwu_fidelity"] = mwu(finals(offA), finals(fid))
    out["fact_types"] = sorted({ft for d in DOMAIN_ORDER
                                for ft in (out["pooled_35b"][d].get("by_type") or {})})
    return out


def plot_f11(br: dict) -> None:
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0])

    order = [d for d in DOMAIN_ORDER if br["pooled_35b"][d].get("n")]
    order = sorted(order, key=lambda d: br["pooled_35b"][d]["sim"]["mean"])

    # (a) per-domain sim and anchors
    ax = fig.add_subplot(gs[0, :])
    if order:
        xs = np.arange(len(order))
        for src, dx, col, lab in (("pooled_35b", -0.2, "#1f77b4",
                                   "3.6-35B sim (n=%d)" % br["pooled_35b"][order[0]]["n"]),
                                  ("flash", 0.0, "#9467bd", "Flash-Next sim")):
            vals, errs, valid = [], [], []
            for i, d in enumerate(order):
                b = br[src][d]
                if not b.get("n"):
                    continue
                valid.append(i + dx)
                vals.append(b["sim"]["mean"])
                errs.append([b["sim"]["mean"] - b["sim"]["ci_low"],
                             b["sim"]["ci_high"] - b["sim"]["mean"]])
            if valid:
                ax.bar(valid, vals, width=0.2, yerr=np.array(errs).T, capsize=1.5,
                       color=col, label=lab, error_kw={"lw": 0.6})
        for src, dx, col, lab in (("pooled_35b", 0.2, "#ff7f0e", "3.6-35B anchors"),
                                  ("flash", 0.4, "#e377c2", "Flash-Next anchors")):
            vals, valid = [], []
            for i, d in enumerate(order):
                b = br[src][d]
                if not b.get("n"):
                    continue
                valid.append(i + dx)
                vals.append(b["anchors"]["mean"])
            if valid:
                ax.bar(valid, vals, width=0.2, color=col, label=lab, alpha=0.9)
        ax.set_xticks(xs + 0.1)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Final similarity / anchor survival")
        ax.legend(fontsize=7, ncol=4)
        ax.set_title("(a) Per-domain final similarity and anchor-fact survival "
                     "(sorted by 3.6-35B similarity)", fontsize=9)
    else:
        empty_panel(ax, "(a) per-domain breadth")

    # (b) chain-level scatter
    ax = fig.add_subplot(gs[1, 0])
    cs = br["chain_scatter"]
    if cs["n"]:
        ax.scatter(cs["sims"], cs["anchors"], s=10, alpha=0.4, color="#4c72b0")
        r = cs["rho"]
        dr = br["domain_level_rho"]
        ax.set_xlabel("Final cosine similarity")
        ax.set_ylabel("Anchor-fact survival")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"(b) Chain-level similarity vs fact survival\n"
                     f"n={cs['n']}, $\\rho$={fmt(r['rho'] if r else None, 2)}; "
                     f"domain-level $\\rho$={fmt(dr['rho'] if dr else None, 2)}",
                     fontsize=9)
    else:
        empty_panel(ax, "(b) chain-level scatter")

    # (c) fact-type heatmap
    ax = fig.add_subplot(gs[1, 1])
    fts = br["fact_types"]
    if order and fts:
        M = np.full((len(order), len(fts)), np.nan)
        for i, d in enumerate(order):
            bt = br["pooled_35b"][d].get("by_type") or {}
            for j, ft in enumerate(fts):
                if ft in bt and bt[ft]["mean"] is not None:
                    M[i, j] = bt[ft]["mean"]
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(fts)))
        ax.set_xticklabels(fts, fontsize=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=6.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045)
        ax.set_title("(c) Anchor survival by fact type (Qwen3.6-35B, both hosts)",
                     fontsize=9)
    else:
        empty_panel(ax, "(c) fact-type heatmap")

    # (d) OFF vs THINK vs FIDELITY
    ax = fig.add_subplot(gs[2, 0])
    if order:
        xs = np.arange(len(order))
        for src, dx, col, lab in (("pooled_35b", -0.25, "#4c72b0", "OFF"),
                                  ("thinking", 0.0, "#c44e52", "THINK"),
                                  ("fidelity", 0.25, "#55a868", "FIDELITY prompt")):
            vals, valid = [], []
            for i, d in enumerate(order):
                b = br[src][d]
                if not b.get("n"):
                    continue
                valid.append(i + dx)
                vals.append(b["sim"]["mean"])
            if valid:
                ax.bar(valid, vals, width=0.24, color=col, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Final similarity")
        ax.legend(fontsize=7)
        ax.set_title("(d) Qwen3.6-35B: thinking OFF vs ON vs "
                     "Ultra-Constrained Fidelity prompt", fontsize=9)
    else:
        empty_panel(ax, "(d) OFF / THINK / FIDELITY")

    # (e) rank correlations
    ax = fig.add_subplot(gs[2, 1])
    items = [(k, v) for k, v in br["rank_rho"].items() if v and v["rho"] is not None]
    if items:
        names = [k.replace("|", " -- ").replace("_", " ") for k, _ in items]
        vals = [v["rho"] for _, v in items]
        errs = [[v["rho"] - v["ci_low"], v["ci_high"] - v["rho"]] for _, v in items]
        ax.barh(range(len(items)), vals, xerr=np.array(errs).T, color="#8172b2",
                capsize=2, error_kw={"lw": 0.8})
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlim(-0.2, 1.05)
        ax.set_xlabel("Spearman $\\rho$ of domain ranking")
        ax.set_title("(e) Domain rankings transfer across hosts, models and arms",
                     fontsize=9)
    else:
        empty_panel(ax, "(e) domain-ranking rho")

    fig.suptitle("F11. Breadth across 16 knowledge domains", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig11_breadth.png")


def table8(br: dict) -> None:
    rows = []
    order = sorted([d for d in DOMAIN_ORDER if br["pooled_35b"][d].get("n")],
                   key=lambda d: -br["pooled_35b"][d]["sim"]["mean"])
    for d in order:
        b = br["pooled_35b"][d]
        f = br["flash"][d]
        t = br["thinking"][d]
        fi = br["fidelity"][d]
        rows.append(
            f"{escape_latex(d)} & {b['n']} & {fmt(b['sim']['mean'])} & {fmt_ci(b['sim'])} & "
            f"{fmt(b['anchors']['mean'], 2)} & {fmt(b['inflation']['mean'], 2)} & "
            f"{fmt(b['meta']['rate'], 2)} & {fmt(b['locked']['rate'], 2)} & "
            f"{fmt(t['sim']['mean']) if t.get('n') else '---'} & "
            f"{fmt(fi['sim']['mean']) if fi.get('n') else '---'} & "
            f"{fmt(f['sim']['mean']) if f.get('n') else '---'} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{Breadth experiment, 16 knowledge domains, 30 iterations. The first block is
Qwen3.6-35B NVFP4 with both hosts pooled and thinking OFF; THINK is thinking ON,
FIDELITY is the domain-neutral Ultra-Constrained Fidelity system prompt, and Flash
is Qwen3.8-Flash-Next NVFP4. Anchors = fraction of the 10 anchor facts surviving
verbatim in the final output.}
\label{tab:paper3-breadth}
\begin{tabular}{lrrlrrrrrrr}
\toprule
Domain & $n$ & Sim & 95\% CI & Anchors & Infl. & Meta & Locked & THINK & FIDELITY & Flash \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table8_breadth_domain.tex", body)


def table9(br: dict) -> None:
    fts = br["fact_types"]
    rows = []
    order = sorted([d for d in DOMAIN_ORDER if br["pooled_35b"][d].get("n")])
    for d in order:
        bt = br["pooled_35b"][d].get("by_type") or {}
        cells = []
        for ft in fts:
            st = bt.get(ft)
            cells.append(fmt(st["mean"], 2) if st and st["mean"] is not None else "---")
        rows.append(f"{escape_latex(d)} & " + " & ".join(cells) + " \\\\")
    # overall per fact type
    agg = defaultdict(list)
    for c in CHAINS:
        if c.stem == "baseline_breadth" and c.label in POOLED_35B and c.anchor_by_type:
            for ft, v in c.anchor_by_type.items():
                agg[ft].append(v)
    overall = []
    for ft in fts:
        st = mean_ci(agg.get(ft, []))
        overall.append(fmt(st["mean"], 2) if st["mean"] is not None else "---")
    rows.append(r"\midrule")
    rows.append(r"\textbf{All domains} & " + " & ".join(overall) + r" \\")
    body = r"""\begin{table}[t]
\centering
\caption{Anchor-fact survival by fact type (Qwen3.6-35B NVFP4, both hosts pooled,
thinking OFF, $n$=30 chains per domain). Each domain prompt carries exactly ten
tagged anchor facts.}
\label{tab:paper3-facttype}
\begin{tabular}{l""" + "r" * len(fts) + r"""}
\toprule
Domain & """ + " & ".join(escape_latex(f) for f in fts) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("table9_breadth_facttype.tex", body)


# =============================================================================
# 17. F12  INFLATION AND META-COMMENTARY
# =============================================================================

def compute_inflation_meta() -> dict:
    groups = {}
    for cid, disp, labs in CONFIGS + [MTP_CONFIG]:
        for rec in RECIPES:
            chs = baseline_chains(labs, rec)
            if chs:
                groups[f"recipe:{cid}|{rec}"] = {
                    "display": disp, "arm": RECIPE_DISPLAY[rec], "n": len(chs),
                    "inflation": mean_ci([c.inflation for c in chs]),
                    "meta": rate_stat([c.meta for c in chs]),
                    "words": mean_ci([c.words[ITERATIONS - 1] for c in chs])}
    for cid, disp, labs in CONCRETE_CONFIGS:
        for rec in CONCRETE_RECIPES:
            chs = [c for c in sel(labels=labs, stems="baseline_concrete", recipe=rec,
                                  thinking=False) if c.pid is None]
            if chs:
                groups[f"concrete:{cid}|{rec}"] = {
                    "display": disp, "arm": CONCRETE_DISPLAY[rec], "n": len(chs),
                    "inflation": mean_ci([c.inflation for c in chs]),
                    "meta": rate_stat([c.meta for c in chs]),
                    "words": mean_ci([c.words[ITERATIONS - 1] for c in chs])}
    for lab, disp in [("qwen3.6-35b-nvfp4", "Qwen3.6-35B NVFP4 (host A)"),
                      ("qwen3.6-35b-nvfp4-gpu-host-d", "Qwen3.6-35B NVFP4 (host B)"),
                      ("qwen3.8-flash-next-nvfp4", "Qwen3.8-Flash-Next NVFP4")]:
        chs = [c for c in sel(labels=lab, stems="baseline_breadth",
                              recipes=DOMAIN_ORDER) if c.pid is None]
        if chs:
            groups[f"breadth:{lab}"] = {
                "display": disp, "arm": "Breadth (16 domains)", "n": len(chs),
                "inflation": mean_ci([c.inflation for c in chs]),
                "meta": rate_stat([c.meta for c in chs]),
                "words": mean_ci([c.words[ITERATIONS - 1] for c in chs])}
    return groups


def plot_f12(im_stats: dict) -> None:
    keys = list(im_stats)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    if not keys:
        for ax in axes:
            empty_panel(ax, "F12. inflation and meta-commentary")
        fig.tight_layout()
        save_figure(fig, "fig12_inflation_meta.png")
        return
    xs = np.arange(len(keys))
    labels = [f"{im_stats[k]['display'].replace(' NVFP4','')}\n{im_stats[k]['arm']} "
              f"(n={im_stats[k]['n']})" for k in keys]
    ax = axes[0]
    vals = [im_stats[k]["inflation"]["mean"] for k in keys]
    errs = [[im_stats[k]["inflation"]["mean"] - im_stats[k]["inflation"]["ci_low"],
             im_stats[k]["inflation"]["ci_high"] - im_stats[k]["inflation"]["mean"]]
            for k in keys]
    cols = ["#4c72b0" if k.startswith("recipe") else
            "#dd8452" if k.startswith("concrete") else "#55a868" for k in keys]
    ax.bar(xs, vals, yerr=np.array(errs).T, color=cols, capsize=2, error_kw={"lw": 0.7})
    ax.axhline(1.0, color="#888888", ls=":", lw=0.9)
    ax.set_ylabel("Word inflation (final / prompt)")
    ax.set_title("Word inflation at iteration 30 (bootstrap 95% CI)", fontsize=9)
    ax = axes[1]
    vals = [im_stats[k]["meta"]["rate"] for k in keys]
    errs = [[im_stats[k]["meta"]["rate"] - im_stats[k]["meta"]["ci_low"],
             im_stats[k]["meta"]["ci_high"] - im_stats[k]["meta"]["rate"]]
            for k in keys]
    ax.bar(xs, vals, yerr=np.array(errs).T, color=cols, capsize=2, error_kw={"lw": 0.7})
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Meta-commentary rate")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_title("Fraction of chains whose final output carries meta-commentary "
                 "(Wilson 95% CI)", fontsize=9)
    ax.legend(handles=[Line2D([], [], color="#4c72b0", lw=6, label="recipe prompts"),
                       Line2D([], [], color="#dd8452", lw=6, label="concrete SOW"),
                       Line2D([], [], color="#55a868", lw=6, label="breadth domains")],
              fontsize=7)
    fig.suptitle("F12. Verbosity and meta-commentary across all baseline conditions",
                 fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig12_inflation_meta.png")


# =============================================================================
# 18. A1 / A2  PER-SEED SMALL MULTIPLES
# =============================================================================

def _small_multiples(name: str, title: str, getter, configs=None) -> None:
    entries = []
    for cid, disp, labs in (configs if configs is not None else CONFIGS + [MTP_CONFIG]):
        for rec in RECIPES:
            chs = getter(labs, rec)
            entries.append((f"{disp}\n{RECIPE_DISPLAY[rec]}", chs))
    ncol = 4
    nrow = int(np.ceil(len(entries) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for i, (lab, chs) in enumerate(entries):
        ax = axes[i // ncol][i % ncol]
        if not chs:
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{lab}\n(no data)", fontsize=7)
            ax.text(0.5, 0.5, "no data available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="#888888")
            if i % ncol == 0:
                ax.set_ylabel("Cos. sim.", fontsize=7)
            continue
        for c in chs:
            ax.plot(np.arange(1, len(c.sims[:ITERATIONS]) + 1), c.sims[:ITERATIONS],
                    lw=0.7, alpha=0.7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{lab} (n={len(chs)})", fontsize=7)
        if i % ncol == 0:
            ax.set_ylabel("Cos. sim.", fontsize=7)
        if i // ncol == nrow - 1:
            ax.set_xlabel("Iteration", fontsize=7)
    for j in range(len(entries), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_figure(fig, name)


def plot_a1() -> None:
    _small_multiples("figA1_baseline_per_chain.png",
                     "A1. Per-chain baseline trajectories (thinking OFF, T=0.7)",
                     lambda labs, rec: baseline_chains(labs, rec))


def plot_a2() -> None:
    def getter(labs, rec):
        return [c for c in sel(labels=labs, stems="thinking", recipe=rec,
                               thinking=True, n_iter=30) if c.pid is None]
    _small_multiples("figA2_thinking_per_chain.png",
                     "A2. Per-chain trajectories with thinking ON", getter,
                     configs=[(cid, disp, [lab])
                              for cid, disp, lab in THINKING_CONFIGS])


# =============================================================================
# 19. SUMMARY
# =============================================================================

def write_summary(stats: dict) -> None:
    L: list[str] = []
    a = L.append
    a("# Paper B numbers summary (auto-generated by scripts/analyze_paper3.py)")
    a("")
    a("All means are the final (iteration 30) cosine similarity of the chain output to")
    a("the original prompt, unless stated otherwise. A condition's samples are its")
    a("chains: seeds are NOT replicates (see section 3). All CIs on means are")
    a("percentile bootstrap 95% intervals with B=10,000 and seed 0; rates use Wilson")
    a("95% intervals; two-group contrasts use Mann-Whitney U; rank comparisons use")
    a("Spearman rho with a bootstrap 95% CI. The two Qwen3.6-35B NVFP4 deployments are")
    a("called host A and host B.")
    a("")

    # 0 completeness
    ig = stats["integrity"]
    a("## 0. Data completeness and integrity")
    a("")
    a(f"- Paper B chains loaded: **{ig['n_chains']}**; model calls: **{ig['n_calls']}**.")
    a(f"- Chains with at least one empty output: **{ig['n_empty_chains']}** "
      f"({ig['n_empty_calls']} calls).")
    a(f"- Iterations with `finish_reason == \"length\"`: **{ig['n_length_calls']}** "
      f"across {ig['n_length_chains']} chains.")
    a(f"- Restricted to thinking-OFF chains ({ig['n_chains_thinking_off']} chains): "
      f"empty-output chains = **{ig['thinking_off_empty_chains']}**, "
      f"`length` chains = **{ig['thinking_off_length_chains']}** "
      f"-> runaway rate **{ig['runaway_rate_thinking_off']['rate']:.4f}** "
      f"[{ig['runaway_rate_thinking_off']['ci_low']:.4f}, "
      f"{ig['runaway_rate_thinking_off']['ci_high']:.4f}].")
    a("")
    if ig["n_empty_calls"] or ig["n_length_calls"]:
        a("**Correction to the spec's expectation.** The runaway rate is zero for every")
        a("thinking-OFF condition, but it is *not* zero overall: thinking-ON chains on")
        a("Qwen3.6-35B do hit the 16,384-token cap and return an empty assistant")
        a("message. Every empty output in the dataset is also a `finish_reason ==")
        a("\"length\"` iteration, and every one of them is a thinking-ON condition on")
        a("Qwen3.6-35B NVFP4. Affected chains:")
        a("")
        a("| label | file | condition key | empty iters | length iters |")
        a("|---|---|---|---|---|")
        byk = {}
        for d in ig["empty_chain_detail"]:
            byk[(d["label"], d["stem"], d["key"])] = [d["count"], 0]
        for d in ig["length_chain_detail"]:
            byk.setdefault((d["label"], d["stem"], d["key"]), [0, 0])[1] = d["count"]
        for (lab, stem, key), (ne, nl) in sorted(byk.items()):
            a(f"| {DISPLAY.get(lab, lab)} | {stem}.json | `{key}` | {ne} | {nl} |")
        a("")
    else:
        a("Runaway rate is 0 across the whole Paper B dataset and no iteration ended")
        a("with `finish_reason == \"length\"`.")
        a("")
    a("| label | file | chains | model calls |")
    a("|---|---|---|---|")
    for f in stats["files"]:
        a(f"| {DISPLAY.get(f['label'], f['label'])} | {f['stem']}.json | "
          f"{f['chains']} | {f['inferences']} |")
    a("")

    # 1 baseline
    a("## 1. Baseline drift (F1, T1)")
    a("")
    a("| Configuration | Recipe | mean | 95% CI | n | locked-at-end | inflation | meta rate |")
    a("|---|---|---|---|---|---|---|---|")
    for k, v in stats["baseline"].items():
        cid, rec = k.split("|")
        disp = CONFIG_DISPLAY.get(cid, MTP_CONFIG[1])
        a(f"| {disp} | {rec} | {fmt(v['mean'])} | "
          f"[{fmt(v['ci_low'])}, {fmt(v['ci_high'])}] | {v['n']} | "
          f"{fmt(v['locked']['rate'], 2)} ({v['locked']['k']}/{v['locked']['n']}) | "
          f"{fmt(v['inflation']['mean'], 2)} | "
          f"{fmt(v['meta']['rate'], 2)} ({v['meta']['k']}/{v['meta']['n']}) |")
    a("")
    a("Per-condition spread (min / max final similarity):")
    a("")
    for k, v in stats["baseline"].items():
        cid, rec = k.split("|")
        disp = CONFIG_DISPLAY.get(cid, MTP_CONFIG[1])
        a(f"- {disp} {rec}: min {fmt(v['min'])}, max {fmt(v['max'])}, "
          f"sd {fmt(v['std'])}, mean final output {fmt(v['words']['mean'], 1)} words")
    a("")

    # 2 cross-generation
    a("## 2. Cross-generation at matched size (F2, T2)")
    a("")
    a("Paper A rows use clean chains only (Paper A `classify_chain`); the Qwen3 pilot")
    a("is a single unseeded chain per model and has no lasagna run.")
    a("")
    a("| Group | Model | spaghetti mean [CI] (n) | lasagna mean [CI] (n) |")
    a("|---|---|---|---|")
    for g, entries in stats["cross_gen"].items():
        for e in entries:
            s, l = e["spaghetti"], e["lasagna"]
            a(f"| {g} | {e['name']} | {fmt(s['mean'])} {fmt_ci(s)} ({s['n']}) | "
              f"{fmt(l['mean'])} {fmt_ci(l)} ({l['n']}) |")
    a("")

    # 3 determinism
    det = stats["determinism"]
    a("## 3. Seeds do not pin chains (F3, T3)")
    a("")
    for key in ["within_host", "cross_host", "single_worker"]:
        d = det[key]
        if not d.get("n_pairs"):
            continue
        a(f"- **{d['name']}**: {d['n_pairs']} seed-matched pairs; "
          f"{d['identical_at_iter1']}/{d['n_pairs']} identical at iteration 1 "
          f"(rate {fmt(d['identical_at_iter1_rate']['rate'], 3)}); mean identical "
          f"iterations per pair {fmt(d['mean_identical_iterations'], 2)} of 30; "
          f"mean |delta final| {fmt(d['abs_delta_final']['mean'])} "
          f"{fmt_ci(d['abs_delta_final'])}, max {fmt(d['max_abs_delta_final'])}.")
    a("")
    a("Host A vs host B as distributions (baseline, n=15 per host per recipe):")
    a("")
    for rec, d in det["mwu_host_a_vs_b"].items():
        a(f"- {rec}: host A {fmt(d['host_a']['mean'])} {fmt_ci(d['host_a'])} "
          f"(n={d['host_a']['n']}), host B {fmt(d['host_b']['mean'])} "
          f"{fmt_ci(d['host_b'])} (n={d['host_b']['n']}), MWU U={d['mwu']['u']}, "
          f"p={fmt(d['mwu']['p'], 4)}.")
    a("")
    a("T=0 vs T=0.7 (greedy decoding is not reproducible either):")
    a("")
    a("| Model | Recipe | T=0 mean [CI] (n) | sd | T=0.7 mean [CI] (n) | sd | MWU p |")
    a("|---|---|---|---|---|---|---|")
    for k, v in det["t0_vs_t07"].items():
        t0, t7 = v["t0"], v["t07"]
        a(f"| {v['display']} | {v['recipe']} | {fmt(t0['mean'])} {fmt_ci(t0)} "
          f"({t0['n']}) | {fmt(t0['std'])} | {fmt(t7['mean'])} {fmt_ci(t7)} "
          f"({t7['n']}) | {fmt(t7['std'])} | {fmt(v['mwu']['p'], 4)} |")
    a("")

    # 4 thinking
    th = stats["thinking"]
    a("## 4. Thinking mode (F4, T4)")
    a("")
    a("| Configuration | Recipe | OFF mean (n) | ON mean (n) | delta | MWU p | reasoning tok/iter | source |")
    a("|---|---|---|---|---|---|---|---|")
    for k, v in th["per_config"].items():
        a(f"| {v['display']} | {v['recipe']} | {fmt(v['off']['mean'])} "
          f"({v['off']['n']}) | {fmt(v['on']['mean'])} ({v['on']['n']}) | "
          f"{fmt(v['delta'])} | {fmt(v['mwu']['p'], 4)} | "
          f"{fmt(v['reasoning_tokens_per_iter']['mean'], 0)} | {v['reasoning_source']} |")
    a("")
    a("`rep` = the server reported `completion_tokens_details.reasoning_tokens`; "
      "`est` = completion tokens minus output tokens, with output tokens estimated at "
      "4 characters per token.")
    a("")
    a("Long horizon (100 iterations):")
    a("")
    a("| Configuration | Recipe | Think | n | @30 | @100 | delta | min after 30 | dip rate (>0.05) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for k, v in th["long_horizon"].items():
        for arm in ("off", "on"):
            d = v[arm]
            if not d.get("n"):
                continue
            a(f"| {v['display']} | {v['recipe']} | {arm.upper()} | {d['n']} | "
              f"{fmt(d['at30']['mean'])} | {fmt(d['at100']['mean'])} | "
              f"{fmt(d['delta_30_100'])} | {fmt(d['min_after_30']['mean'])} | "
              f"{fmt(d['dip_rate']['rate'], 2)} ({d['dip_rate']['k']}/{d['dip_rate']['n']}) |")
    a("")
    missing_off = [k for k, v in th["long_horizon"].items() if not v["off"].get("n")]
    if missing_off:
        a("Only Qwen3.6-35B NVFP4 (host B) has a thinking-OFF 100-iteration run; the "
          "other 100-iteration runs are thinking-ON only, so their OFF/ON contrast at "
          "100 iterations cannot be drawn.")
        a("")

    # 5 format
    a("## 5. Serving format and stack (F5, T5)")
    a("")
    a("| Model | Recipe | NVFP4 mean [CI] (n) | GGUF mean [CI] (n) | delta | MWU p |")
    a("|---|---|---|---|---|---|")
    for k, v in stats["format"].items():
        if k == "_note":
            continue
        a(f"| {v['model']} | {v['recipe']} | {fmt(v['nvfp4']['mean'])} "
          f"{fmt_ci(v['nvfp4'])} ({v['nvfp4']['n']}) | {fmt(v['gguf']['mean'])} "
          f"{fmt_ci(v['gguf'])} ({v['gguf']['n']}) | {fmt(v['delta'])} | "
          f"{fmt(v['mwu']['p'], 4)} |")
    a("")
    a(stats["format"]["_note"])
    a("")

    # 6 MTP
    a("## 6. Speculative decoding / MTP (F6)")
    a("")
    a("| Recipe | no MTP mean [CI] (n) | MTP mean [CI] (n) | MWU p | words no MTP | words MTP |")
    a("|---|---|---|---|---|---|")
    for rec, v in stats["mtp"].items():
        a(f"| {rec} | {fmt(v['no_mtp']['mean'])} {fmt_ci(v['no_mtp'])} "
          f"({v['no_mtp']['n']}) | {fmt(v['mtp']['mean'])} {fmt_ci(v['mtp'])} "
          f"({v['mtp']['n']}) | {fmt(v['mwu']['p'], 4)} | "
          f"{fmt(v['no_mtp_words']['mean'], 1)} | {fmt(v['mtp_words']['mean'], 1)} |")
    a("")

    # 7 temperature
    a("## 7. Temperature (F7)")
    a("")
    a("| Configuration | Recipe | " + " | ".join(f"T={t:.1f}" for t in TEMPS) + " |")
    a("|---|---|" + "---|" * len(TEMPS))
    for k, v in stats["temperature"].items():
        cells = []
        for t in TEMPS:
            st = v["series"][f"{t:.1f}"]
            cells.append(f"{fmt(st['mean'])} (n={st['n']})" if st["n"] else "---")
        a(f"| {v['display']} | {v['recipe']} | " + " | ".join(cells) + " |")
    a("")

    # 8 sysprompt
    sp = stats["sysprompt"]
    a("## 8. System prompts (F8, T6)")
    a("")
    a("| Configuration | Recipe | prompts | chains/prompt | spread | best | worst |")
    a("|---|---|---|---|---|---|---|")
    for k, v in sp["per_config"].items():
        a(f"| {v['display']} | {v['recipe']} | {v['n_prompts']} | "
          f"{'/'.join(str(x) for x in v['chains_per_prompt'])} | {fmt(v['spread'])} | "
          f"{v['best_name']} ({fmt(v['best'])}) | {v['worst_name']} ({fmt(v['worst'])}) |")
    a("")
    a("Rank transfer (Spearman rho over the shared prompts, bootstrap 95% CI):")
    a("")
    a("| Pair | n prompts | rho | 95% CI | p |")
    a("|---|---|---|---|---|")
    for pair, v in sp["rho_matrix"].items():
        x, y = pair.split("||")
        dx, dy = sp["per_config"][x], sp["per_config"][y]
        a(f"| {dx['display']} {dx['recipe']} vs {dy['display']} {dy['recipe']} | "
          f"{v['n']} | {fmt(v['rho'], 3)} | [{fmt(v['ci_low'], 3)}, "
          f"{fmt(v['ci_high'], 3)}] | {fmt(v['p'], 4)} |")
    a("")
    a("Mean final similarity by prompt category:")
    a("")
    a("| Configuration | Recipe | " + " | ".join(CATEGORY_ORDER) + " |")
    a("|---|---|" + "---|" * len(CATEGORY_ORDER))
    for k, row in sp["categories"].items():
        v = sp["per_config"][k]
        a(f"| {v['display']} | {v['recipe']} | " +
          " | ".join(fmt(row[c]) for c in CATEGORY_ORDER) + " |")
    a("")
    ts_rho = sp["thinking_sysprompt_rho"]
    a(f"Thinking x system prompt, Qwen3.6-35B (host A) lasagna, all 20 prompts: mean "
      f"delta(ON-OFF) = {fmt(sp['thinking_sysprompt_mean_delta'])}, "
      f"{sp['thinking_sysprompt_n_worse']}/{sp['thinking_sysprompt_n']} prompts worse "
      f"with thinking on, OFF/ON prompt-ranking rho = "
      f"{fmt(ts_rho['rho'] if ts_rho else None, 3)} "
      f"{fmt_ci(ts_rho, 3) if ts_rho else ''}.")
    a("")
    a("| Prompt | OFF mean (n) | ON mean (n) | delta |")
    a("|---|---|---|---|")
    for pid, v in sp["thinking_sysprompt"].items():
        d = (v["on"]["mean"] - v["off"]["mean"]
             if v["on"]["mean"] is not None and v["off"]["mean"] is not None else None)
        a(f"| {pid:02d} {v['name']} | {fmt(v['off']['mean'])} ({v['off']['n']}) | "
          f"{fmt(v['on']['mean'])} ({v['on']['n']}) | {fmt(d)} |")
    a("")

    # 9 locking
    a("## 9. Byte-level locking (F9)")
    a("")
    a("| Configuration | Recipe | Think | locked-at-end | 95% CI | mean lock onset |")
    a("|---|---|---|---|---|---|")
    for src, arm in (("baseline", "OFF"), ("thinking", "ON")):
        for k, v in stats["locking"][src].items():
            lk = v["locked"]
            a(f"| {v['display']} | {v['recipe']} | {arm} | "
              f"{fmt(lk['rate'], 2)} ({lk['k']}/{lk['n']}) | "
              f"[{fmt(lk['ci_low'], 2)}, {fmt(lk['ci_high'], 2)}] | "
              f"{fmt(v['onset_stat']['mean'], 1)} |")
    a("")

    # 10 concrete
    con = stats["concrete"]
    a("## 10. Concrete statement of work (F10, T7)")
    a("")
    a("| Configuration | Document | Think | n | sim | numeric survival | inflation | exclusions kept | genre drift | meta |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for k, b in con["baseline"].items():
        a(f"| {b['display']} | {CONCRETE_DISPLAY[b['recipe']]} | OFF | {b['n']} | "
          f"{fmt(b['sim']['mean'])} | {fmt(b['numeric']['mean'], 3)} | "
          f"{fmt(b['inflation']['mean'], 2)} | {fmt(b['exclusions']['rate'], 2)} | "
          f"{fmt(b['genre']['rate'], 2)} | {fmt(b['meta']['rate'], 2)} |")
    for k, v in con["thinking"].items():
        b = v["on"]
        a(f"| {v['display']} | {CONCRETE_DISPLAY[v['recipe']]} | ON | {b['n']} | "
          f"{fmt(b['sim']['mean'])} | {fmt(b['numeric']['mean'], 3)} | "
          f"{fmt(b['inflation']['mean'], 2)} | {fmt(b['exclusions']['rate'], 2)} | "
          f"{fmt(b['genre']['rate'], 2)} | {fmt(b['meta']['rate'], 2)} "
          f"(MWU p={fmt(v['mwu']['p'], 4)}) |")
    a("")
    a("Domain-neutral system prompts on host B:")
    a("")
    a("| System prompt | Document | n | sim | numeric | inflation | exclusions kept | genre drift |")
    a("|---|---|---|---|---|---|---|---|")
    for k, b in con["sysprompt"].items():
        a(f"| {b['pid']:02d} {b['name']} | {CONCRETE_DISPLAY[b['recipe']]} | {b['n']} | "
          f"{fmt(b['sim']['mean'])} | {fmt(b['numeric']['mean'], 3)} | "
          f"{fmt(b['inflation']['mean'], 2)} | {fmt(b['exclusions']['rate'], 2)} | "
          f"{fmt(b['genre']['rate'], 2)} |")
    a("")
    sc = con["scatter"]
    r, rb = sc["rho"], sc["rho_baseline_only"]
    a(f"Similarity vs numeric-token survival across all {sc['n']} concrete chains: "
      f"Spearman rho = {fmt(r['rho'] if r else None, 3)} "
      f"{fmt_ci(r, 3) if r else ''}, p={fmt(r['p'] if r else None, 4)}. "
      f"Baseline (thinking OFF, no system prompt) chains only, n={sc['n_baseline_only']}: "
      f"rho = {fmt(rb['rho'] if rb else None, 3)} {fmt_ci(rb, 3) if rb else ''}, "
      f"p={fmt(rb['p'] if rb else None, 4)}.")
    a("")

    # 11 breadth
    br = stats["breadth"]
    a("## 11. Breadth across 16 domains (F11, T8, T9)")
    a("")
    for arm, key in (("Qwen3.6-35B, thinking OFF, both hosts pooled", "overall_off"),
                     ("Qwen3.6-35B, thinking ON (host B)", "overall_think"),
                     ("Qwen3.6-35B, Ultra-Constrained Fidelity prompt (host A)",
                      "overall_fidelity"),
                     ("Qwen3.8-Flash-Next, thinking OFF", "overall_flash")):
        v = br[key]
        a(f"- {arm}: n={v['n']}, sim {fmt(v['sim']['mean'])} "
          f"{fmt_ci(v['sim'])}, anchor survival {fmt(v['anchors']['mean'], 3)} "
          f"{fmt_ci(v['anchors'], 3)}, inflation {fmt(v['inflation']['mean'], 2)}, "
          f"meta rate {fmt(v['meta']['rate'], 2)}, locked {fmt(v['locked']['rate'], 2)}.")
    a("")
    a(f"- Thinking ON vs OFF on host B, all domains pooled: MWU p="
      f"{fmt(br['mwu_think']['p'], 4)} (n={br['mwu_think']['n1']} vs "
      f"{br['mwu_think']['n2']}).")
    a(f"- Fidelity prompt vs no prompt on host A, all domains pooled: MWU p="
      f"{fmt(br['mwu_fidelity']['p'], 6)} (n={br['mwu_fidelity']['n1']} vs "
      f"{br['mwu_fidelity']['n2']}).")
    a("")
    a("| Domain | n | sim | 95% CI | anchors | inflation | meta | locked | THINK sim | FIDELITY sim | Flash sim | Flash anchors |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in sorted(DOMAIN_ORDER,
                    key=lambda d: -(br["pooled_35b"][d]["sim"]["mean"]
                                    if br["pooled_35b"][d].get("n") else -1)):
        b = br["pooled_35b"][d]
        if not b.get("n"):
            continue
        t, fi, fl = br["thinking"][d], br["fidelity"][d], br["flash"][d]
        a(f"| {d} | {b['n']} | {fmt(b['sim']['mean'])} | {fmt_ci(b['sim'])} | "
          f"{fmt(b['anchors']['mean'], 3)} | {fmt(b['inflation']['mean'], 2)} | "
          f"{fmt(b['meta']['rate'], 2)} | {fmt(b['locked']['rate'], 2)} | "
          f"{fmt(t['sim']['mean']) if t.get('n') else '---'} | "
          f"{fmt(fi['sim']['mean']) if fi.get('n') else '---'} | "
          f"{fmt(fl['sim']['mean']) if fl.get('n') else '---'} | "
          f"{fmt(fl['anchors']['mean'], 3) if fl.get('n') else '---'} |")
    a("")
    fts = br["fact_types"]
    a("Anchor survival by fact type (Qwen3.6-35B pooled, thinking OFF):")
    a("")
    a("| Domain | " + " | ".join(fts) + " |")
    a("|---|" + "---|" * len(fts))
    for d in DOMAIN_ORDER:
        b = br["pooled_35b"][d]
        if not b.get("n"):
            continue
        bt = b["by_type"]
        a(f"| {d} | " + " | ".join(
            fmt(bt[ft]["mean"], 2) if ft in bt and bt[ft]["mean"] is not None else "---"
            for ft in fts) + " |")
    agg = defaultdict(list)
    for c in CHAINS:
        if c.stem == "baseline_breadth" and c.label in POOLED_35B and c.anchor_by_type:
            for ft, v in c.anchor_by_type.items():
                agg[ft].append(v)
    a("| **all domains** | " + " | ".join(
        fmt(mean_ci(agg.get(ft, []))["mean"], 3) for ft in fts) + " |")
    a("")
    cs = br["chain_scatter"]
    dr = br["domain_level_rho"]
    a(f"Similarity vs anchor survival: chain level n={cs['n']}, rho="
      f"{fmt(cs['rho']['rho'] if cs['rho'] else None, 3)} "
      f"{fmt_ci(cs['rho'], 3) if cs['rho'] else ''}; domain level n=16, rho="
      f"{fmt(dr['rho'] if dr else None, 3)} {fmt_ci(dr, 3) if dr else ''}.")
    a("")
    a("Domain-ranking transfer (Spearman rho with bootstrap CI):")
    a("")
    for k, v in br["rank_rho"].items():
        if not v:
            continue
        a(f"- {k.replace('|', ' on ')}: rho={fmt(v['rho'], 3)} {fmt_ci(v, 3)} "
          f"(n={v['n']} domains)")
    a("")

    # 12 inflation / meta
    a("## 12. Inflation and meta-commentary (F12)")
    a("")
    a("| Configuration | Condition | n | inflation [CI] | meta rate [CI] | mean final words |")
    a("|---|---|---|---|---|---|")
    for k, v in stats["inflation_meta"].items():
        a(f"| {v['display']} | {v['arm']} | {v['n']} | "
          f"{fmt(v['inflation']['mean'], 2)} {fmt_ci(v['inflation'], 2)} | "
          f"{fmt(v['meta']['rate'], 2)} [{fmt(v['meta']['ci_low'], 2)}, "
          f"{fmt(v['meta']['ci_high'], 2)}] | {fmt(v['words']['mean'], 1)} |")
    a("")

    # anomalies
    a("## 13. Data anomalies found by the analysis script")
    a("")
    if ANOMALIES:
        for m in ANOMALIES:
            a(f"- {m}")
    else:
        a("- None.")
    a("")

    (OUT_DIR / "summary_for_text.md").write_text(sanitize("\n".join(L)) + "\n")
    print(f"  summary_for_text.md -> {OUT_DIR}")


# =============================================================================
# 20. MAIN
# =============================================================================

def main() -> None:
    plt.rcParams.update(PAPER_STYLE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    load_all()

    print("\nIntegrity check (empty outputs / finish_reason == length) ...")
    STATS["files"] = FILE_SUMMARY
    STATS["integrity"] = compute_integrity()
    ig = STATS["integrity"]
    print(f"  chains={ig['n_chains']} calls={ig['n_calls']} "
          f"empty_calls={ig['n_empty_calls']} length_calls={ig['n_length_calls']}")
    print(f"  thinking-OFF chains: {ig['n_chains_thinking_off']}, "
          f"empty={ig['thinking_off_empty_chains']}, "
          f"length={ig['thinking_off_length_chains']}")

    print("\nComputing statistics ...")
    STATS["baseline"] = compute_baseline()
    STATS["cross_gen"] = compute_cross_gen()
    STATS["determinism"] = compute_determinism()
    STATS["thinking"] = compute_thinking()
    STATS["format"] = compute_format()
    STATS["mtp"] = compute_mtp()
    STATS["temperature"] = compute_temperature()
    STATS["sysprompt"] = compute_sysprompt()
    STATS["locking"] = compute_locking()
    STATS["concrete"] = compute_concrete()
    STATS["breadth"] = compute_breadth()
    STATS["inflation_meta"] = compute_inflation_meta()

    print("\nFigures ...")
    plot_f1(STATS["baseline"])
    plot_f2(STATS["cross_gen"])
    plot_f3(STATS["determinism"])
    plot_f4(STATS["thinking"])
    plot_f5(STATS["format"])
    plot_f6(STATS["mtp"])
    plot_f7(STATS["temperature"])
    plot_f8(STATS["sysprompt"])
    plot_f9(STATS["locking"])
    plot_f10(STATS["concrete"])
    plot_f11(STATS["breadth"])
    plot_f12(STATS["inflation_meta"])
    plot_a1()
    plot_a2()

    print("\nTables ...")
    table1(STATS["baseline"])
    table2(STATS["cross_gen"])
    table3(STATS["determinism"])
    table4(STATS["thinking"])
    table5(STATS["format"])
    table6(STATS["sysprompt"])
    table7(STATS["concrete"])
    table8(STATS["breadth"])
    table9(STATS["breadth"])

    print("\nSummary ...")
    STATS["anomalies"] = ANOMALIES
    write_summary(STATS)
    (OUT_DIR / "all_stats.json").write_text(
        json.dumps(numpy_to_python(STATS), indent=2))
    print(f"  all_stats.json -> {OUT_DIR}")

    print(f"\nDone. {len(ANOMALIES)} anomalies recorded.")


if __name__ == "__main__":
    main()


# ---- post-processing: keep every table inside the text width ----
def _wrap_resizebox(path):
    import re
    s = open(path).read()
    if "\\resizebox" in s:
        return
    s = re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})", r"\\resizebox{\\textwidth}{!}{%\n\1}", s, flags=re.S)
    open(path, "w").write(s)

if __name__ == "__main__":
    import glob as _glob
    for _f in _glob.glob("paper3/tables/table*.tex"):
        _wrap_resizebox(_f)
