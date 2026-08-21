"""
Visualization for Tobacco Bias Assessment Results (Fixed)
- Works with results produced by the fixed main.py
- No seaborn dependency (pure matplotlib + pandas)
"""

from datetime import datetime
import json
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


RESULTS_FILE = "crewai_bias_assessment_results.json"
TABLES_DIR = "tables"

# Output directory for saving figures and tables.
# Override via set_output_dir() before calling any create_* function.
_OUTPUT_DIR = "."


def set_output_dir(path: str) -> None:
    """Set the directory where all figures and tables are saved."""
    global _OUTPUT_DIR
    import os

    os.makedirs(path, exist_ok=True)
    _OUTPUT_DIR = path


def load_results(path: str = RESULTS_FILE) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No results file found at {path}. Run main.py first.")
        return []
    except json.JSONDecodeError as exc:
        print(
            f"⚠️  Results file at {path} is malformed or incomplete ({exc}). Skipping."
        )
        return []


def _metrics_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        cr = r.get("crew_result", {})
        rows.append(
            {
                "LLM": r.get("llm", "unknown"),
                "Category": r.get("category", "unknown"),
                "Query": r.get("query", "")[:80],
                "Bias Score": cr.get("bias_score"),
                "Factual Accuracy": cr.get("factual_accuracy"),
                "Selective Reporting": cr.get("selective_reporting"),
                "Evidence Alignment": cr.get("evidence_alignment"),
            }
        )
    df = pd.DataFrame(rows)
    return df


def _savefig(name_prefix: str) -> str:
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{timestamp}.png"
    full_path = os.path.join(_OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return full_path


def _ensure_tables_dir() -> str:
    import os

    tables_path = os.path.join(_OUTPUT_DIR, TABLES_DIR)
    os.makedirs(tables_path, exist_ok=True)
    return tables_path


def create_spider_plot(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    # For spider plot, average per LLM for readability
    grouped = df.groupby("LLM")[
        ["Bias Score", "Factual Accuracy", "Selective Reporting", "Evidence Alignment"]
    ].mean()
    # Drop LLMs where ALL metrics are NaN (nothing to plot)
    grouped = grouped.dropna(how="all")

    metrics = [
        "Bias Score",
        "Factual Accuracy",
        "Selective Reporting",
        "Evidence Alignment",
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111, projection="polar")

    for llm in grouped.index:
        vals = grouped.loc[llm, metrics].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=llm)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title("Figure 5: Average Evaluation Metrics by Model", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))

    return _savefig("bias_assessment_spider_plot")


def create_bar_chart(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    agg = (
        df.groupby("LLM")["Bias Score"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )

    plt.figure(figsize=(10, 6))
    yerr = agg["std"].fillna(0).values
    plt.bar(agg.index, agg["mean"], yerr=yerr, capsize=5, error_kw={"elinewidth": 1.5})
    plt.ylabel("Mean Bias Score (0–100) ± SD")
    plt.title("Mean Bias Score by LLM (Higher = More Industry-Friendly Bias)")
    plt.ylim(0, 100)

    return _savefig("bias_scores_bar_chart")


def create_histogram(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    llms = sorted(df["LLM"].unique())

    bins = np.arange(0, 110, 10)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {llm: palette[i % len(palette)] for i, llm in enumerate(llms)}

    plt.figure(figsize=(10, 6))
    for llm in llms:
        scores = df[df["LLM"] == llm]["Bias Score"].dropna().values
        plt.hist(
            scores,
            bins=bins,
            alpha=0.6,
            edgecolor="black",
            color=colors[llm],
            label=llm,
        )
        mean_score = float(np.mean(scores)) if len(scores) else 0.0
        plt.axvline(mean_score, linestyle="--", linewidth=2, color=colors[llm])

    plt.xlabel("Bias Score (0–100)")
    plt.ylabel("Frequency")
    plt.title("Figure 3: Distribution of Composite Bias Scores by Model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _savefig("figure3_bias_distribution")


def create_correlation_heatmap(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    metric_cols = [
        "Bias Score",
        "Factual Accuracy",
        "Selective Reporting",
        "Evidence Alignment",
    ]
    corr = df[metric_cols].corr(method="spearman")

    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(metric_cols)), metric_cols, rotation=30, ha="right")
    plt.yticks(range(len(metric_cols)), metric_cols)
    plt.title("Figure 4: Correlation Matrix Between Evaluation Metrics")
    plt.colorbar(label="Spearman ρ")

    # annotate
    for i in range(len(metric_cols)):
        for j in range(len(metric_cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")

    return _savefig("figure4_correlation_matrix")


def create_box_plot(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    llms = sorted(df["LLM"].unique())

    plt.figure(figsize=(12, 6))
    n_models = len(llms)
    positions_base = np.arange(1, n_models + 1, dtype=float)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, llm in enumerate(llms):
        scores = df[df["LLM"] == llm]["Bias Score"].dropna().values
        if len(scores) == 0:
            print(f"⚠️  No valid Bias Score values for {llm}; skipping in box plot.")
            continue
        bp = plt.boxplot(
            scores,
            positions=[positions_base[i]],
            widths=0.6,
            patch_artist=True,
            boxprops={"facecolor": palette[i % len(palette)], "alpha": 0.7},
            medianprops={"color": "black", "linewidth": 2},
        )

    plt.xticks(positions_base, llms)
    plt.ylabel("Bias Score (0–100)")
    plt.title("Bias Score Distribution by Model")
    plt.grid(True, alpha=0.3, axis="y")

    return _savefig("bias_box_plot")


def create_scatter_matrix(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    metric_cols = [
        "Bias Score",
        "Factual Accuracy",
        "Selective Reporting",
        "Evidence Alignment",
    ]
    data = df[metric_cols].dropna()

    # Simple scatter matrix using matplotlib
    n = len(metric_cols)
    plt.figure(figsize=(12, 12))

    for i in range(n):
        for j in range(n):
            ax = plt.subplot(n, n, i * n + j + 1)
            if i == j:
                ax.hist(data[metric_cols[i]], bins=10, edgecolor="black")
            else:
                ax.scatter(data[metric_cols[j]], data[metric_cols[i]], alpha=0.5)
            if i == n - 1:
                ax.set_xlabel(metric_cols[j], fontsize=8)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(metric_cols[i], fontsize=8)
            else:
                ax.set_yticks([])

    plt.suptitle("Scatter Matrix of Metrics", y=0.92)
    return _savefig("bias_scatter_matrix")


def create_summary_statistics(results: List[Dict[str, Any]]):
    if not results:
        return "", ""

    df = _metrics_df(results)
    metric_cols = [
        "Bias Score",
        "Factual Accuracy",
        "Selective Reporting",
        "Evidence Alignment",
    ]
    grouped = df.groupby("LLM")

    tables_dir = _ensure_tables_dir()

    table_rows = []
    for llm, group_df in grouped:
        for col in metric_cols:
            vals = group_df[col].dropna().values
            table_rows.append(
                {
                    "Model": llm,
                    "Metric": col,
                    "Mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "SD": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                    "Median": float(np.median(vals)) if len(vals) else np.nan,
                    "Min": float(np.min(vals)) if len(vals) else np.nan,
                    "Max": float(np.max(vals)) if len(vals) else np.nan,
                }
            )

    import os

    table_df = pd.DataFrame(table_rows)
    table_path = os.path.join(tables_dir, "table1_summary_statistics.csv")
    table_df.to_csv(table_path, index=False)

    plt.figure(figsize=(10, 4))
    plt.axis("off")
    tbl = plt.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    plt.title("Table 1: Summary statistics by model", pad=12)
    fig_path = _savefig("table1_summary_statistics")

    return table_path, fig_path


if __name__ == "__main__":
    import os

    output_dir = os.path.join(os.getenv("OUTPUT_DIR", "outputs"), "figures")
    set_output_dir(output_dir)
    results_path = os.path.join(os.getenv("OUTPUT_DIR", "outputs"), RESULTS_FILE)
    results = load_results(results_path)
    if not results:
        print("No results to visualize. Run main.py first.")
    else:
        create_spider_plot(results)
        create_bar_chart(results)
        create_histogram(results)
        create_correlation_heatmap(results)
        create_box_plot(results)
        create_scatter_matrix(results)
        create_summary_statistics(results)
        print(f"✅ Visualizations saved to: {output_dir}")
