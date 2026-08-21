"""Portable bootstrap sensitivity analysis for expert validation.

The script consumes an adjudication workbook and the pipeline's JSON results.
It reports both the observed Spearman correlation and the exploratory
attenuation-corrected correlation. Score standardisation is optional and must
be requested explicitly with ``--standardize``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import openpyxl
from scipy.stats import spearmanr


DIMS = {
    "FA": "factual_accuracy",
    "EA": "evidence_alignment",
    "SR": "selective_reporting",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=Path("adjudicated_reference.xlsx"),
        help="Adjudication workbook containing the Adjudicated_reference sheet.",
    )
    parser.add_argument(
        "--ai-results",
        type=Path,
        default=Path("outputs/crewai_bias_assessment_results.json"),
        help="Pipeline JSON results containing crew_result component scores.",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--standardize",
        action="store_true",
        help=(
            "Standardize each rater's component scores to a pooled per-dimension "
            "mean and SD before compositing. This is an exploratory transformation."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON summary; otherwise only stdout is used.",
    )
    return parser.parse_args()


def read_adjudication(path: Path) -> list[dict[str, object]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook["Adjudicated_reference"]
    rows = sheet.iter_rows(values_only=True)
    headers = [str(value) for value in next(rows)]
    records = [dict(zip(headers, row)) for row in rows if row and row[4]]
    workbook.close()
    return records


def read_ai_results(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    return {str(row["response_id"]): row for row in results}


def composite_bias(components: dict[str, np.ndarray]) -> np.ndarray:
    return 100.0 - (
        0.30 * components["FA"]
        + 0.35 * components["EA"]
        + 0.35 * components["SR"]
    )


def icc2k(first: np.ndarray, second: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement ICC for the mean of two raters."""
    matrix = np.column_stack([first, second])
    n, k = matrix.shape
    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_columns = n * np.sum((column_means - grand_mean) ** 2)
    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_columns
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def common_scale(values: np.ndarray, pooled: np.ndarray) -> np.ndarray:
    pooled_mean = pooled.mean()
    pooled_sd = pooled.std(ddof=1)
    value_sd = values.std(ddof=1)
    if value_sd == 0 or not np.isfinite(value_sd):
        raise ValueError("Cannot standardize a component with zero or invalid SD")
    return np.clip(
        pooled_mean + (values - values.mean()) / value_sd * pooled_sd,
        0,
        100,
    )


def adjudicate(
    first: np.ndarray,
    second: np.ndarray,
    picks: Iterable[str],
) -> np.ndarray:
    chosen = []
    for left, right, pick in zip(first, second, picks):
        label = str(pick).strip().upper()
        if label == "E1":
            chosen.append(left)
        elif label == "E2":
            chosen.append(right)
        elif label in {"TIE", "AVG", "AVERAGE"}:
            chosen.append((left + right) / 2.0)
        else:
            raise ValueError(f"Unexpected adjudication label: {pick!r}")
    return np.asarray(chosen, dtype=float)


def percentile_interval(values: list[float]) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def bootstrap(
    reference: np.ndarray,
    ai_bias: np.ndarray,
    expert_1_bias: np.ndarray,
    expert_2_bias: np.ndarray,
    query_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
    clustered: bool,
) -> dict[str, object]:
    rng = np.random.RandomState(seed)
    clusters = {
        query_id: np.flatnonzero(query_ids == query_id)
        for query_id in np.unique(query_ids)
    }
    cluster_ids = list(clusters)
    observed_values: list[float] = []
    corrected_values: list[float] = []
    discarded_observed = 0
    discarded_corrected = 0
    corrected_over_one = 0

    for _ in range(n_bootstrap):
        if clustered:
            selected = rng.randint(0, len(cluster_ids), len(cluster_ids))
            indices = np.concatenate([clusters[cluster_ids[index]] for index in selected])
        else:
            indices = rng.randint(0, len(reference), len(reference))

        observed = float(spearmanr(reference[indices], ai_bias[indices]).statistic)
        if not np.isfinite(observed):
            discarded_observed += 1
            continue
        observed_values.append(observed)

        reliability = icc2k(expert_1_bias[indices], expert_2_bias[indices])
        if not np.isfinite(reliability) or reliability <= 0.01:
            discarded_corrected += 1
            continue
        corrected = observed / np.sqrt(reliability)
        corrected_values.append(float(corrected))
        corrected_over_one += int(corrected > 1.0)

    return {
        "level": "query-clustered" if clustered else "response-level",
        "bootstrap_replicates": n_bootstrap,
        "observed_usable": len(observed_values),
        "observed_discarded": discarded_observed,
        "observed_spearman_ci": percentile_interval(observed_values),
        "corrected_usable": len(corrected_values),
        "corrected_discarded": discarded_corrected,
        "corrected_over_one": corrected_over_one,
        "corrected_spearman_ci_uncapped": percentile_interval(corrected_values),
    }


def main() -> None:
    args = parse_args()
    records = read_adjudication(args.adjudication)
    ai_results = read_ai_results(args.ai_results)
    response_ids = [str(row["response_id"]) for row in records]

    missing = [response_id for response_id in response_ids if response_id not in ai_results]
    if missing:
        raise ValueError(f"AI results are missing {len(missing)} adjudicated responses")

    expert_1 = {
        dim: np.asarray([float(row[f"{dim}_E1"]) for row in records])
        for dim in DIMS
    }
    expert_2 = {
        dim: np.asarray([float(row[f"{dim}_E2"]) for row in records])
        for dim in DIMS
    }
    ai_components = {
        dim: np.asarray(
            [
                float(ai_results[response_id]["crew_result"][result_key])
                for response_id in response_ids
            ]
        )
        for dim, result_key in DIMS.items()
    }

    if args.standardize:
        for dim in DIMS:
            pooled = np.concatenate([expert_1[dim], expert_2[dim], ai_components[dim]])
            expert_1[dim] = common_scale(expert_1[dim], pooled)
            expert_2[dim] = common_scale(expert_2[dim], pooled)
            ai_components[dim] = common_scale(ai_components[dim], pooled)

    reference_components = {
        dim: adjudicate(
            expert_1[dim],
            expert_2[dim],
            [str(row[f"{dim}_pick"]) for row in records],
        )
        for dim in DIMS
    }
    reference_bias = composite_bias(reference_components)
    ai_bias = composite_bias(ai_components)
    expert_1_bias = composite_bias(expert_1)
    expert_2_bias = composite_bias(expert_2)
    query_ids = np.asarray([str(row["query_id"]) for row in records])

    observed = float(spearmanr(reference_bias, ai_bias).statistic)
    reliability = icc2k(expert_1_bias, expert_2_bias)
    corrected = observed / np.sqrt(reliability) if reliability > 0 else float("nan")
    summary = {
        "n_responses": len(records),
        "n_query_clusters": len(np.unique(query_ids)),
        "standardized": bool(args.standardize),
        "observed_spearman": observed,
        "expert_icc_2k": reliability,
        "attenuation_corrected_spearman": corrected,
        "note": (
            "The attenuation-corrected estimate is exploratory. Corrected bootstrap "
            "values are retained uncapped; replicates with ICC <= 0.01 are discarded."
        ),
        "bootstrap": [
            bootstrap(
                reference_bias,
                ai_bias,
                expert_1_bias,
                expert_2_bias,
                query_ids,
                args.bootstrap,
                args.seed,
                clustered=False,
            ),
            bootstrap(
                reference_bias,
                ai_bias,
                expert_1_bias,
                expert_2_bias,
                query_ids,
                args.bootstrap,
                args.seed,
                clustered=True,
            ),
        ],
    }

    rendered = json.dumps(summary, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
