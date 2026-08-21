"""
make_manuscript_figures.py - publication-ready Figures 1-5 for the manuscript.

Figures 3-5 are drawn from outputs/crewai_bias_assessment_results.json (the locked
full run). Figures 1-2 are conceptual diagrams matching the manuscript legends.
No internal titles (legends live in the manuscript). 300 dpi PNG, Arial.

Usage: python make_manuscript_figures.py   (writes outputs/manuscript_figures/)
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(os.environ.get("OUTPUT_DIR", "outputs")) / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = Path(os.environ.get("OUTPUT_DIR", "outputs")) / "crewai_bias_assessment_results.json"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_GEM, C_LLA = "#0072B2", "#E69F00"  # colorblind-safe
GREY = "#4D4D4D"


def _box(ax, xy, w, h, text, fc="#EDF2F7", ec=GREY, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#1A202C", fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, p1, p2, color=GREY, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))






def _load_scores():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    by = defaultdict(lambda: defaultdict(list))
    for r in res:
        cr = r["crew_result"]
        for k in ("bias_score", "factual_accuracy", "evidence_alignment", "selective_reporting"):
            by[r["llm"]][k].append(cr[k])
    return by


def figure3():
    by = _load_scores()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.arange(15, 62.5, 2.5)
    for m, c in (("Gemini", C_GEM), ("Llama-3", C_LLA)):
        v = np.array(by[m]["bias_score"])
        ax.hist(v, bins=bins, alpha=0.55, color=c, label=f"{m} (n=58)", edgecolor="white", lw=0.5)
        ax.axvline(np.median(v), color=c, ls="--", lw=1.4)
    ax.set_xlabel("Composite bias score (0–100; higher = more industry-aligned)")
    ax.set_ylabel("Number of responses")
    ax.legend(frameon=False)
    ax.text(0.99, 0.82, "dashed lines: medians", transform=ax.transAxes, ha="right", fontsize=7.5, color=GREY)
    fig.savefig(OUT / "figure3_bias_distribution.png"); plt.close(fig)


def figure4():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    from scipy.stats import spearmanr
    keys = ["factual_accuracy", "evidence_alignment", "selective_reporting", "bias_score"]
    labels = ["Factual\nAccuracy", "Evidence\nAlignment", "Selective\nReporting", "Overall\nBias"]
    X = np.array([[r["crew_result"][k] for k in keys] for r in res])
    n = len(keys)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = spearmanr(X[:, i], X[:, j]).statistic
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n), labels); ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.6 else "#1A202C", fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("Spearman ρ")
    ax.spines[:].set_visible(False)
    fig.savefig(OUT / "figure4_correlation_matrix.png"); plt.close(fig)


def figure5():
    by = _load_scores()
    labels = ["Factual\nAccuracy", "Evidence\nAlignment", "Selective\nReporting", "Overall Bias\n(higher = worse)"]
    keys = ["factual_accuracy", "evidence_alignment", "selective_reporting", "bias_score"]
    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.6, 5.2), subplot_kw=dict(polar=True))
    offsets = {"Gemini": (0, 10), "Llama-3": (0, -14)}
    for m, c in (("Gemini", C_GEM), ("Llama-3", C_LLA)):
        vals = [float(np.mean(by[m][k])) for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=c, lw=2, label=f"{m} (n=58)")
        ax.fill(angles, vals, color=c, alpha=0.12)
        for a, v in zip(angles[:-1], vals[:-1]):
            ax.annotate(f"{v:.1f}", (a, v), textcoords="offset points", xytext=offsets[m],
                        ha="center", fontsize=7.5, color=c, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 100); ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color=GREY)
    ax.grid(color="#CBD5E0", lw=0.7)
    ax.spines["polar"].set_color("#CBD5E0")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False, fontsize=8.5)
    fig.savefig(OUT / "figure5_metrics_by_model.png"); plt.close(fig)


if __name__ == "__main__":
    # Figures 1-2 are author-drawn conceptual diagrams already embedded in the manuscript.
    figure3(); figure4(); figure5()
    for f in sorted(OUT.glob("*.png")):
        print("saved:", f)
