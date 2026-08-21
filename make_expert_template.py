# make_expert_template.py
#
# Build the expert annotation template for validation.
# Experts review RESPONSES_PER_MODEL responses for EACH evaluated LLM. We sample
# whole queries (stratified by category) and include every model's response to each
# chosen query. This yields exactly RESPONSES_PER_MODEL responses per model and
# balances model source automatically.
#
# Manuscript-aligned design:
# - Experts score three component rubrics only.
# - expert_bias_score is calculated downstream from those three component scores.

import os
import random
from pathlib import Path
import pandas as pd

_base = os.environ.get("OUTPUT_DIR", "outputs")
ANNOTATIONS_DIR = Path(f"{_base}/annotations")
OUTPUT_SAMPLED_PACK = ANNOTATIONS_DIR / "sampled_annotation_pack.csv"
OUTPUT_COORDINATOR_TEMPLATE = ANNOTATIONS_DIR / "coordinator_expert_annotation_template_with_model_ids.csv"
OUTPUT_TEMPLATE = ANNOTATIONS_DIR / "expert_annotation_template.csv"

# Number of responses each expert reviews per evaluated LLM.
# Manuscript default: 18 per model = 36 responses total for two evaluated models.
RESPONSES_PER_MODEL = int(os.environ.get("RESPONSES_PER_MODEL", "18"))
SEED = 42


def _find_annotation_pack() -> Path:
    """Return the most recent annotation_pack_informed_*.csv, falling back to annotation_pack.csv."""
    candidates = sorted(ANNOTATIONS_DIR.glob("annotation_pack_informed_*.csv"))
    if candidates:
        return candidates[-1]
    fallback = ANNOTATIONS_DIR / "annotation_pack.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No annotation pack found in {ANNOTATIONS_DIR}. "
        "Run the pipeline first to generate annotation_pack_informed_*.csv"
    )


def _stratified_query_sample(pack: pd.DataFrame, n_target: int, seed: int = SEED) -> set:
    """Pick n_target query_ids, stratified by category via largest-remainder allocation.

    Sampling whole queries (not individual rows) means every evaluated model's response
    to a chosen query is included, giving exactly n_target responses per model and
    balancing model source by construction.
    """
    queries = pack[["query_id", "category"]].drop_duplicates()
    total = len(queries)
    n_target = min(n_target, total)

    by_cat = {cat: g["query_id"].tolist() for cat, g in queries.groupby("category")}
    exact = {cat: n_target * len(ids) / total for cat, ids in by_cat.items()}
    floors = {cat: int(v) for cat, v in exact.items()}
    remainder = n_target - sum(floors.values())
    for cat in sorted(exact, key=lambda c: -(exact[c] - floors[c]))[:remainder]:
        floors[cat] += 1

    rng = random.Random(seed)
    chosen: list = []
    for cat, ids in by_cat.items():
        chosen.extend(rng.sample(ids, min(floors[cat], len(ids))))
    return set(chosen)


def main():
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    input_full_pack = _find_annotation_pack()
    merged_full = pd.read_csv(input_full_pack)

    if "category" not in merged_full.columns or merged_full["category"].isna().all():
        raise ValueError("Cannot stratify: category missing or all NaN in annotation pack.")
    if "query_id" not in merged_full.columns:
        raise ValueError("Cannot sample: 'query_id' column missing from annotation pack.")

    chosen_qids = _stratified_query_sample(merged_full, RESPONSES_PER_MODEL, seed=SEED)
    sampled = (
        merged_full[merged_full["query_id"].isin(chosen_qids)]
        .sort_values(["query_id", "llm"])
        .reset_index(drop=True)
    )

    n_queries = sampled["query_id"].nunique()
    per_model = sampled.groupby("llm").size().to_dict() if "llm" in sampled.columns else {}

    sampled.to_csv(OUTPUT_SAMPLED_PACK, index=False)
    print(
        f"Stratified sample saved as: {OUTPUT_SAMPLED_PACK} "
        f"({n_queries} queries / {len(sampled)} responses)"
    )

    template = sampled.copy()
    for col in [
        "expert_factual_accuracy",
        "expert_evidence_alignment",
        "expert_selective_reporting",
        "expert_notes",
    ]:
        template[col] = ""

    # Coordinator copy keeps the real model name for audit/reconciliation.
    template.to_csv(OUTPUT_COORDINATOR_TEMPLATE, index=False)

    # Expert copy is blinded: reviewers see only llm_label. Drop every column that
    # could reveal model identity (llm AND llm_model_id, e.g. "google/gemini-2.5-pro").
    blinded_template = template.drop(columns=["llm", "llm_model_id"], errors="ignore")
    blinded_template.to_csv(OUTPUT_TEMPLATE, index=False)
    print(f"Coordinator template saved as: {OUTPUT_COORDINATOR_TEMPLATE}")
    print(f"Blinded expert annotation template saved as: {OUTPUT_TEMPLATE}")
    print(f"Responses per model: {per_model}")
    print("Experts score three component rubrics only; expert_bias_score is calculated during validation.")


if __name__ == "__main__":
    main()


