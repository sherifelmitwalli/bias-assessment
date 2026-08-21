"""
expert_validation_analysis.py - manuscript-aligned validation.

Experts score three component dimensions only:
- expert_factual_accuracy
- expert_evidence_alignment
- expert_selective_reporting

The expert composite bias score is calculated deterministically using the same formula
as the automated score:

  bias_score = 100 - (0.30*FA + 0.35*EA + 0.35*SR)
"""

from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Honor OUTPUT_DIR like main.py and statistical_analysis.py (default: "outputs")
_BASE = os.environ.get("OUTPUT_DIR", "outputs")
OUTPUT_DIR = Path(_BASE) / "expert_validation"
ANNOTATIONS_DIR = Path(_BASE) / "annotations"
RESULTS_PATH = Path(_BASE) / "crewai_bias_assessment_results.json"

THRESHOLDS = {
    "icc_good": 0.75,
    "icc_moderate": 0.50,
    "spearman_strong": 0.70,
    "mae_acceptable": 10.0,
    "weighted_kappa_substantial": 0.61,
}

COMPONENTS = {
    "factual_accuracy": ("expert_factual_accuracy", "factual_accuracy"),
    "evidence_alignment": ("expert_evidence_alignment", "evidence_alignment"),
    "selective_reporting": ("expert_selective_reporting", "selective_reporting"),
}

N_BOOTSTRAP = 10_000
SEED = 42


def calculate_bias_score(fa, ea, sr):
    return 100.0 - (0.30 * fa + 0.35 * ea + 0.35 * sr)


def bootstrap_ci(func, x, y, cluster_ids, n_boot=N_BOOTSTRAP, ci=0.95):
    n = len(x)
    if n < 2:
        point = round(func(x, y), 3) if n == 1 else float("nan")
        return {"point": point, "ci_lower": float("nan"), "ci_upper": float("nan")}
    cluster_ids = np.asarray(cluster_ids)
    if len(cluster_ids) != n:
        raise ValueError("cluster_ids must have the same length as the score arrays")

    clusters = {
        cluster_id: np.flatnonzero(cluster_ids == cluster_id)
        for cluster_id in np.unique(cluster_ids)
    }
    cluster_keys = list(clusters)
    rng = np.random.RandomState(SEED)
    stats = []
    for _ in range(n_boot):
        sampled = rng.randint(0, len(cluster_keys), size=len(cluster_keys))
        idx = np.concatenate([clusters[cluster_keys[i]] for i in sampled])
        value = func(x[idx], y[idx])
        if np.isfinite(value):
            stats.append(value)
    if not stats:
        return {
            "point": round(func(x, y), 3),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "bootstrap_replicates": n_boot,
            "usable_replicates": 0,
            "resampling_level": "query-clustered",
            "n_clusters": len(cluster_keys),
        }
    alpha = (1 - ci) / 2
    return {
        "point": round(func(x, y), 3),
        "ci_lower": round(np.percentile(stats, 100 * alpha), 3),
        "ci_upper": round(np.percentile(stats, 100 * (1 - alpha)), 3),
        "bootstrap_replicates": n_boot,
        "usable_replicates": len(stats),
        "resampling_level": "query-clustered",
        "n_clusters": len(cluster_keys),
    }


def spearman_func(x, y):
    return spearmanr(x, y)[0]


def spearman_pvalue(x, y):
    return float(spearmanr(x, y)[1])


def mae_func(x, y):
    return float(np.mean(np.abs(x - y)))


def compute_icc(scores1, scores2, item_ids):
    if len(scores1) < 2:
        return {"icc": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "p_value": float("nan")}
    df = pd.DataFrame(
        {
            "item": list(item_ids) * 2,
            "rater": ["r1"] * len(item_ids) + ["r2"] * len(item_ids),
            "score": list(scores1) + list(scores2),
        }
    )
    icc = pg.intraclass_corr(data=df, targets="item", raters="rater", ratings="score")
    # Select ICC(2,1): two-way random effects, absolute agreement, single rater.
    # pingouin <0.5.4 labels this "ICC2"; pingouin >=0.5.4 (incl. 0.6.x) relabels
    # it "ICC(A,1)". Match either so the primary-outcome script is version-robust.
    row = icc[icc["Type"].isin(["ICC2", "ICC(A,1)"])]
    if row.empty:
        raise ValueError(
            "Could not locate ICC(2,1)/ICC(A,1) row in pingouin output. "
            f"Available Types: {icc['Type'].tolist()}"
        )
    # pingouin <0.5.4 names the CI column "CI95%"; >=0.5.4 (incl. 0.6.x) uses "CI95".
    ci_col = "CI95%" if "CI95%" in row.columns else "CI95"
    ci = row[ci_col].values[0]
    return {
        "icc": round(row["ICC"].values[0], 3),
        "ci_lower": round(float(ci[0]), 3),
        "ci_upper": round(float(ci[1]), 3),
        "p_value": round(row["pval"].values[0], 4),
    }


def weighted_kappa(scores1, scores2, bins=(0, 33.333, 66.666, 100.0)):
    labels = ["low", "medium", "high"]
    s1 = pd.Series(scores1)
    s2 = pd.Series(scores2)
    valid = ~s1.isna() & ~s2.isna()
    if valid.sum() < 2:
        return np.nan
    b1 = pd.cut(s1[valid], bins=bins, labels=labels, include_lowest=True)
    b2 = pd.cut(s2[valid], bins=bins, labels=labels, include_lowest=True)
    return float(cohen_kappa_score(b1, b2, weights="quadratic", labels=labels))


def interpret(metrics):
    icc_val = metrics.get("icc", {}).get("icc", 0)
    spearman_val = metrics.get("spearman", {}).get("point", 0)
    mae_val = metrics.get("mae", {}).get("point", 100)
    return {
        "icc": "GOOD" if icc_val >= THRESHOLDS["icc_good"] else "MODERATE" if icc_val >= THRESHOLDS["icc_moderate"] else "POOR",
        "spearman": "STRONG" if spearman_val >= THRESHOLDS["spearman_strong"] else "WEAK",
        "mae": "ACCEPTABLE" if mae_val <= THRESHOLDS["mae_acceptable"] else "HIGH",
    }


def _find_annotation_pack() -> Path:
    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ANNOTATIONS_DIR}. Run main.py first.")
    candidates = sorted(ANNOTATIONS_DIR.glob("annotation_pack_informed_*.csv"))
    if candidates:
        return candidates[-1]
    fallback = ANNOTATIONS_DIR / "annotation_pack.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No annotation pack found in {ANNOTATIONS_DIR}. Run make_expert_template.py first.")


def _load_ai_results(results_path: Path):
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ai_score_maps(ai_results):
    maps = {field: {} for field in ["bias_score", "factual_accuracy", "evidence_alignment", "selective_reporting"]}
    for rec in ai_results:
        query = rec.get("query", "").strip()
        llm = rec.get("llm", "").strip()
        if not query or not llm:
            continue
        crew_result = rec.get("crew_result", {})
        for field in maps:
            val = crew_result.get(field)
            if val is not None:
                try:
                    maps[field][(query, llm)] = float(val)
                except (TypeError, ValueError):
                    pass
    return maps


def _add_expert_composite(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    fa_col = f"expert_factual_accuracy_{suffix}"
    ea_col = f"expert_evidence_alignment_{suffix}"
    sr_col = f"expert_selective_reporting_{suffix}"
    required = [fa_col, ea_col, sr_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required expert component column(s): {missing}")

    fa = pd.to_numeric(df[fa_col], errors="coerce")
    ea = pd.to_numeric(df[ea_col], errors="coerce")
    sr = pd.to_numeric(df[sr_col], errors="coerce")
    df[f"expert_bias_score_{suffix}"] = calculate_bias_score(fa, ea, sr)
    return df


def _agreement(
    ai_vals: np.ndarray,
    rater_vals: np.ndarray,
    ids: np.ndarray,
    query_ids: np.ndarray,
    label: str,
) -> dict:
    valid = ~np.isnan(ai_vals) & ~np.isnan(rater_vals)
    if not valid.any():
        print(f"[warn] No valid pairs for {label} - skipping.")
        return {}
    ai_v = ai_vals[valid]
    rt_v = rater_vals[valid]
    id_v = ids[valid]
    query_v = query_ids[valid]

    icc = compute_icc(ai_v, rt_v, id_v)
    sp = bootstrap_ci(spearman_func, ai_v, rt_v, query_v)
    sp["p_value"] = round(spearman_pvalue(ai_v, rt_v), 4)
    mae = bootstrap_ci(mae_func, ai_v, rt_v, query_v)

    print(f"\n=== AI Judge vs {label} ===")
    print(f"N: {len(ai_v)}")
    print(f"ICC(2,1): {icc['icc']} (95% CI {icc['ci_lower']}-{icc['ci_upper']})")
    print(f"Spearman: {sp['point']} (95% CI {sp['ci_lower']}-{sp['ci_upper']}, p={sp['p_value']})")
    print(f"MAE: {mae['point']} (95% CI {mae['ci_lower']}-{mae['ci_upper']})")
    return {"icc": icc, "spearman": sp, "mae": mae}


def _per_dimension_agreement(
    merged: pd.DataFrame,
    ai_maps: dict,
    item_ids: np.ndarray,
    query_ids: np.ndarray,
) -> dict:
    out = {}
    for dim_name, (expert_prefix, ai_field) in COMPONENTS.items():
        e1_col = f"{expert_prefix}_e1"
        e2_col = f"{expert_prefix}_e2"
        if e1_col not in merged.columns or e2_col not in merged.columns:
            continue
        e1 = pd.to_numeric(merged[e1_col], errors="coerce").to_numpy(float)
        e2 = pd.to_numeric(merged[e2_col], errors="coerce").to_numpy(float)
        mean = (e1 + e2) / 2
        ai_vals = np.array(
            [ai_maps.get(ai_field, {}).get((row.get("query", ""), row.get("llm", "")), np.nan) for row in merged.to_dict("records")],
            dtype=float,
        )
        out[dim_name] = {
            "expert1": _agreement(ai_vals, e1, item_ids, query_ids, f"Expert 1 - {dim_name}"),
            "expert2": _agreement(ai_vals, e2, item_ids, query_ids, f"Expert 2 - {dim_name}"),
            "mean_expert": _agreement(ai_vals, mean, item_ids, query_ids, f"Mean Expert - {dim_name}"),
        }
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotation_pack = _find_annotation_pack()
    e1_path = ANNOTATIONS_DIR / "expert_1_annotations.csv"
    e2_path = ANNOTATIONS_DIR / "expert_2_annotations.csv"

    for p, label in ((e1_path, "Expert 1"), (e2_path, "Expert 2")):
        if not p.exists():
            raise FileNotFoundError(f"{label} annotation file not found: {p}\nComplete expert annotation before running validation.")

    pack = pd.read_csv(annotation_pack)
    e1 = pd.read_csv(e1_path, comment="#")
    e2 = pd.read_csv(e2_path, comment="#")

    base_keys = ["query_id", "response_id", "llm", "query"]
    base = pack[base_keys].drop_duplicates()

    # Expert templates are blinded and may omit the real model name. Merge on stable
    # response identifiers, then reattach llm from the coordinator annotation pack.
    expert_merge_keys = ["query_id", "response_id", "query"]
    missing_keys = [c for c in expert_merge_keys if c not in e1.columns or c not in e2.columns]
    if missing_keys:
        raise ValueError(f"Expert annotation files missing required merge key(s): {sorted(set(missing_keys))}")

    m1 = base.merge(e1, on=expert_merge_keys, how="left")
    m2 = base.merge(e2, on=expert_merge_keys, how="left")
    merged = m1.merge(m2, on=base_keys, suffixes=("_e1", "_e2"))

    merged = _add_expert_composite(merged, "e1")
    merged = _add_expert_composite(merged, "e2")

    x = pd.to_numeric(merged["expert_bias_score_e1"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(merged["expert_bias_score_e2"], errors="coerce").to_numpy(float)
    complete = ~np.isnan(x) & ~np.isnan(y)
    missing_report = {
        "total": int(len(merged)),
        "complete": int(complete.sum()),
        "rate": round(float(complete.sum()) / len(merged) * 100, 1) if len(merged) else 0.0,
        "score_source": "expert_bias_score calculated from three expert component rubrics",
    }

    if not complete.any():
        print("[warn] No complete annotation pairs found. Cannot compute agreement statistics.")
        with (OUTPUT_DIR / "validation_summary.json").open("w", encoding="utf-8") as f:
            json.dump({"missing_data": missing_report, "error": "no_complete_pairs"}, f, indent=2)
        return

    merged = merged.loc[complete].reset_index(drop=True)
    x = x[complete]
    y = y[complete]
    item_ids = merged["response_id"].to_numpy()
    query_ids = merged["query_id"].to_numpy()

    icc_ee = compute_icc(x, y, item_ids)
    sp_ee = bootstrap_ci(spearman_func, x, y, query_ids)
    sp_ee["p_value"] = round(spearman_pvalue(x, y), 4)
    mae_ee = bootstrap_ci(mae_func, x, y, query_ids)
    wk_ee = weighted_kappa(x, y)

    print("\n=== Expert vs Expert Agreement - formula-derived composite bias ===")
    print(f"N paired ratings: {len(x)}")
    print(f"ICC(2,1): {icc_ee['icc']} (95% CI {icc_ee['ci_lower']}-{icc_ee['ci_upper']})")
    print(f"Spearman: {sp_ee['point']} (95% CI {sp_ee['ci_lower']}-{sp_ee['ci_upper']}, p={sp_ee['p_value']})")
    print(f"MAE: {mae_ee['point']} (95% CI {mae_ee['ci_lower']}-{mae_ee['ci_upper']})")
    print(f"Weighted kappa: {wk_ee:.3f}")

    ai_results = _load_ai_results(RESULTS_PATH)
    ai_maps = _ai_score_maps(ai_results)
    ai_vals = np.array(
        [ai_maps.get("bias_score", {}).get((row.get("query", ""), row.get("llm", "")), np.nan) for row in merged.to_dict("records")],
        dtype=float,
    )
    e_mean = (x + y) / 2

    judge_vs_expert1 = _agreement(ai_vals, x, item_ids, query_ids, "Expert 1")
    judge_vs_expert2 = _agreement(ai_vals, y, item_ids, query_ids, "Expert 2")
    judge_vs_expert_mean = _agreement(ai_vals, e_mean, item_ids, query_ids, "Mean Expert")
    per_dim = _per_dimension_agreement(merged, ai_maps, item_ids, query_ids)

    summary = {
        "missing_data": missing_report,
        "expert_vs_expert": {"icc": icc_ee, "spearman": sp_ee, "mae": mae_ee, "weighted_kappa": wk_ee},
        "judge_vs_expert1": judge_vs_expert1,
        "judge_vs_expert2": judge_vs_expert2,
        "judge_vs_expert_mean": judge_vs_expert_mean,
        "interpretation": interpret(judge_vs_expert_mean) if judge_vs_expert_mean else {},
        "per_dimension": per_dim,
        "thresholds_used": THRESHOLDS,
        "primary_outcome_note": (
            "Automated and expert overall bias scores are formula-derived composites from the three "
            "component rubrics. Spearman and MAE confidence intervals use 10,000 query-clustered "
            "bootstrap resamples (seed 42), retaining both model responses whenever a query is sampled."
        ),
    }

    with (OUTPUT_DIR / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if per_dim:
        rows = []
        for dim, comparators in per_dim.items():
            for key, label in [("expert1", "Expert 1"), ("expert2", "Expert 2"), ("mean_expert", "Mean Expert")]:
                vals = comparators.get(key)
                if not vals:
                    continue
                rows.append(
                    {
                        "dimension": dim,
                        "comparator": label,
                        "icc": vals["icc"]["icc"],
                        "icc_ci": f"{vals['icc']['ci_lower']}-{vals['icc']['ci_upper']}",
                        "spearman": vals["spearman"]["point"],
                        "spearman_ci": f"{vals['spearman']['ci_lower']}-{vals['spearman']['ci_upper']}",
                        "spearman_p_value": vals["spearman"].get("p_value", ""),
                        "mae": vals["mae"]["point"],
                        "mae_ci": f"{vals['mae']['ci_lower']}-{vals['mae']['ci_upper']}",
                    }
                )
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "per_dimension_agreement.csv", index=False)

    comparison_rows = []
    for label, result in (("Expert 1", judge_vs_expert1), ("Expert 2", judge_vs_expert2), ("Mean Expert", judge_vs_expert_mean)):
        if result:
            comparison_rows.append(
                {
                    "comparison": f"AI Judge vs {label}",
                    "icc": result["icc"]["icc"],
                    "icc_ci_lower": result["icc"]["ci_lower"],
                    "icc_ci_upper": result["icc"]["ci_upper"],
                    "spearman_rho": result["spearman"]["point"],
                    "spearman_ci_lower": result["spearman"]["ci_lower"],
                    "spearman_ci_upper": result["spearman"]["ci_upper"],
                    "spearman_p_value": result["spearman"].get("p_value", ""),
                    "mae": result["mae"]["point"],
                    "mae_ci_lower": result["mae"]["ci_lower"],
                    "mae_ci_upper": result["mae"]["ci_upper"],
                }
            )
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "judge_vs_expert_comparison.csv", index=False)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

