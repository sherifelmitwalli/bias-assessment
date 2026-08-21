"""
statistical_analysis.py — between-model comparisons and correlation matrix.

Design note: each query is answered by ALL evaluated models, so observations are
PAIRED by query_id. Between-model tests must account for this dependency:
  - 2 models  → Wilcoxon signed-rank test (paired)
  - 3+ models → Friedman test (nonparametric repeated-measures), then pairwise
                Wilcoxon signed-rank with Holm-Bonferroni correction
Using Kruskal-Wallis or Mann-Whitney U would incorrectly assume independence.
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


_base = os.environ.get("OUTPUT_DIR", "outputs")
RESULTS_PATH = Path(
    os.environ.get("RESULTS_PATH", f"{_base}/crewai_bias_assessment_results.json")
)
OUTPUT_DIR = Path(f"{_base}/statistical_analysis")


def _holm_bonferroni(p_values):
    """Return Holm-Bonferroni adjusted p-values for a list of raw p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    running_max = 0.0
    for rank in range(n):
        idx = order[rank]
        adjusted[idx] = max(adjusted[idx], running_max)
        running_max = adjusted[idx]
    return adjusted


def _rank_biserial_r(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Rank-biserial correlation for the Wilcoxon signed-rank test.
    r = (R+ - R-) / (R+ + R-), where R+ and R- are sums of positive and
    negative signed ranks after removing zero-differences.
    Ranges from -1 (s2 always larger) to +1 (s1 always larger).
    """
    diffs = s1 - s2
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    r_plus = float(np.sum(ranks[nonzero > 0]))
    r_minus = float(np.sum(ranks[nonzero < 0]))
    total = r_plus + r_minus
    return (r_plus - r_minus) / total if total > 0 else 0.0


def _wilcoxon_row(m1: str, m2: str, s1: np.ndarray, s2: np.ndarray) -> dict:
    """Run a paired Wilcoxon signed-rank test and return a result dict."""
    try:
        stat, p_raw = wilcoxon(s1, s2, alternative="two-sided")
        r = _rank_biserial_r(s1, s2)
    except Exception as exc:
        stat, p_raw, r = np.nan, 1.0, np.nan
        print(f"  ⚠ Wilcoxon failed for {m1} vs {m2}: {exc}")
    return {
        "model_1": m1,
        "model_2": m2,
        "n_pairs": len(s1),
        "W": stat,
        "p_raw": p_raw,
        "rank_biserial_r": round(r, 3) if not np.isnan(r) else np.nan,
        "mean_1": round(float(np.mean(s1)), 2),
        "sd_1": round(float(np.std(s1, ddof=1)), 2),
        "median_1": round(float(np.median(s1)), 2),
        "mean_2": round(float(np.mean(s2)), 2),
        "sd_2": round(float(np.std(s2, ddof=1)), 2),
        "median_2": round(float(np.median(s2)), 2),
        "median_diff_1_minus_2": round(float(np.median(s1 - s2)), 2),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results not found: {RESULTS_PATH}")

    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for rec in results:
        cr = rec.get("crew_result", {})
        rows.append(
            {
                "model": rec.get("llm"),
                "query_id": rec.get("query_id"),
                "category": rec.get("category"),
                "bias_score": cr.get("bias_score"),
                "factual_accuracy": cr.get("factual_accuracy"),
                "evidence_alignment": cr.get("evidence_alignment"),
                "selective_reporting": cr.get("selective_reporting"),
            }
        )

    df = pd.DataFrame(rows)
    # Coerce all score columns to numeric (non-parseable values become NaN)
    for col in [
        "bias_score",
        "factual_accuracy",
        "evidence_alignment",
        "selective_reporting",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(OUTPUT_DIR / "results_flat.csv", index=False)

    models = sorted(df["model"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Pivot to wide format so paired tests can align scores by query_id
    # ------------------------------------------------------------------
    wide = df.pivot_table(
        index="query_id", columns="model", values="bias_score", aggfunc="first"
    )
    # Keep only models that actually appear as columns (guard against pivot drift)
    models = [m for m in models if m in wide.columns]
    if len(models) < len(df["model"].dropna().unique()):
        dropped = set(df["model"].dropna().unique()) - set(models)
        print(f"⚠️  Models missing from pivot (no bias_score rows): {dropped}")
    # Keep only queries where ALL models provided a score
    wide = wide[models].dropna()
    n_paired = len(wide)
    n_total = df["query_id"].nunique()
    print(f"Paired queries (all models scored): {n_paired} / {n_total}")

    if n_paired < 2:
        print(
            "⚠ Fewer than 2 paired queries — skipping inferential tests. "
            "No comparison CSVs will be written."
        )
    elif len(models) == 2:
        # ---------------------------------------------------------------
        # 2 models: Wilcoxon signed-rank (paired)
        # ---------------------------------------------------------------
        m1, m2 = models
        s1, s2 = wide[m1].values, wide[m2].values
        row = _wilcoxon_row(m1, m2, s1, s2)
        row["test"] = "Wilcoxon signed-rank (paired by query_id)"
        pd.DataFrame([row]).to_csv(
            OUTPUT_DIR / "between_model_comparison.csv", index=False
        )
        print(
            f"Wilcoxon W={row['W']}, p={row['p_raw']:.4f}, r={row['rank_biserial_r']}"
        )

    elif len(models) >= 3:
        # ---------------------------------------------------------------
        # 3+ models: Friedman omnibus (nonparametric repeated-measures)
        # ---------------------------------------------------------------
        groups = [wide[m].values for m in models]
        chi2, p_friedman = friedmanchisquare(*groups)
        k = len(models)
        kendalls_W = chi2 / (n_paired * (k - 1))

        pd.DataFrame(
            [
                {
                    "test": "Friedman (paired by query_id)",
                    "k_models": k,
                    "n_pairs": n_paired,
                    "chi2": round(chi2, 3),
                    "p_value": round(p_friedman, 4),
                    "kendalls_W": round(kendalls_W, 3),
                }
            ]
        ).to_csv(OUTPUT_DIR / "friedman_omnibus.csv", index=False)
        print(
            f"Friedman chi2={chi2:.3f}, p={p_friedman:.4f}, Kendall's W={kendalls_W:.3f}"
        )

        # Post-hoc pairwise Wilcoxon with Holm-Bonferroni
        pairs = list(itertools.combinations(models, 2))
        raw_rows = [
            _wilcoxon_row(m1, m2, wide[m1].values, wide[m2].values) for m1, m2 in pairs
        ]
        comp_df = pd.DataFrame(raw_rows)
        comp_df["p_adjusted_holm"] = _holm_bonferroni(
            comp_df["p_raw"].fillna(1.0).values
        )
        comp_df["significant_adj_p05"] = comp_df["p_adjusted_holm"] < 0.05
        comp_df.to_csv(OUTPUT_DIR / "between_model_comparison.csv", index=False)
        print(f"Post-hoc pairwise Wilcoxon saved ({len(comp_df)} pairs).")

    # ------------------------------------------------------------------
    # Category-level descriptive statistics
    # ------------------------------------------------------------------
    cat_rows = []
    metric_cols_cat = [
        "bias_score",
        "factual_accuracy",
        "evidence_alignment",
        "selective_reporting",
    ]
    for cat in sorted(df["category"].dropna().unique()):
        for model in models:
            sub = df[(df["category"] == cat) & (df["model"] == model)]
            row: dict = {"category": cat, "model": model}
            for metric in metric_cols_cat:
                vals = sub[metric].dropna().values
                row[f"{metric}_n"] = len(vals)
                row[f"{metric}_mean"] = (
                    round(float(np.mean(vals)), 2) if len(vals) else np.nan
                )
                row[f"{metric}_sd"] = (
                    round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else np.nan
                )
                row[f"{metric}_median"] = (
                    round(float(np.median(vals)), 2) if len(vals) else np.nan
                )
                row[f"{metric}_min"] = (
                    round(float(np.min(vals)), 2) if len(vals) else np.nan
                )
                row[f"{metric}_max"] = (
                    round(float(np.max(vals)), 2) if len(vals) else np.nan
                )
            cat_rows.append(row)
    pd.DataFrame(cat_rows).to_csv(OUTPUT_DIR / "category_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Spearman correlation matrix across all four metrics
    # ------------------------------------------------------------------
    metric_cols = [
        "bias_score",
        "factual_accuracy",
        "evidence_alignment",
        "selective_reporting",
    ]
    corr = df[metric_cols].corr(method="spearman")
    corr.to_csv(OUTPUT_DIR / "metric_correlation_matrix.csv")

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
