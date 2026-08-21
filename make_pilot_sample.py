"""
make_pilot_sample.py - build a balanced stratified PILOT query subset.

Purpose
-------
The conference pilot needs a small, category-balanced subset of the full
58-query instrument. main.py's --queries flag only takes the FIRST N queries,
which is NOT balanced (the dataset is not category-ordered). This script draws
an equal number of queries from each content category with a fixed seed, so the
pilot is balanced and fully reproducible.

It is non-invasive: it does not modify the master dataset, prompts, rubric,
scoring, or the judge. It only writes a curated subset file that main.py can be
pointed at via --queries-file.

Usage
-----
    python make_pilot_sample.py                 # 4 per category (12 total)
    N_PER_CATEGORY=3 python make_pilot_sample.py # 3 per category (9 total)

Outputs (written next to the master dataset, in data/)
    data/llm_bias_queries_pilot.json  - subset in the same schema as the master
    data/pilot_sample_manifest.csv    - audit trail: original index, category, query
"""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
N_PER_CATEGORY = int(os.environ.get("N_PER_CATEGORY", "4"))
CATEGORY_ORDER = ["Scientific/Evaluative", "Strategic", "Regulatory"]

DATA_DIR = Path("data")
MASTER = DATA_DIR / "llm_bias_queries.json"
PILOT = DATA_DIR / "llm_bias_queries_pilot.json"
MANIFEST = DATA_DIR / "pilot_sample_manifest.csv"


def load_master():
    with MASTER.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "tobacco_bias_queries" in payload:
        return payload["tobacco_bias_queries"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unexpected master query file format.")


def main():
    if not MASTER.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER}")

    queries = load_master()

    # Group original (1-based) indices by category, preserving original order.
    by_cat = defaultdict(list)
    for i, q in enumerate(queries, start=1):
        by_cat[q.get("category", "unknown")].append((i, q))

    rng = random.Random(SEED)
    chosen = []  # list of (original_index, query_dict)
    for cat in CATEGORY_ORDER:
        pool = by_cat.get(cat, [])
        if len(pool) < N_PER_CATEGORY:
            raise ValueError(
                f"Category '{cat}' has only {len(pool)} queries; "
                f"cannot draw {N_PER_CATEGORY}."
            )
        picks = sorted(rng.sample(pool, N_PER_CATEGORY), key=lambda t: t[0])
        chosen.extend(picks)

    # Keep a deterministic, category-blocked order (Sci, Strategic, Regulatory).
    subset = [q for (_idx, q) in chosen]

    with PILOT.open("w", encoding="utf-8") as f:
        json.dump({"tobacco_bias_queries": subset}, f, indent=2, ensure_ascii=False)

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pilot_query_id", "original_index", "category", "query"])
        for pilot_id, (orig_idx, q) in enumerate(chosen, start=1):
            w.writerow([pilot_id, orig_idx, q.get("category", "unknown"), q.get("query", "")])

    counts = defaultdict(int)
    for _idx, q in chosen:
        counts[q.get("category", "unknown")] += 1

    print(f"Seed: {SEED} | per category: {N_PER_CATEGORY}")
    print(f"Pilot sample: {len(subset)} queries -> {PILOT}")
    for cat in CATEGORY_ORDER:
        print(f"  {cat}: {counts[cat]}")
    print(f"Manifest: {MANIFEST}")
    print("Point the pipeline at the pilot file with:")
    print(f"  python main.py --llms Llama-3 Gemini --queries-file {PILOT.as_posix()}")


if __name__ == "__main__":
    main()
