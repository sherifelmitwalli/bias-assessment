# Tobacco Bias Assessment Framework

**Author and maintainer:** [Sherif Elmitwalli](https://github.com/sherifelmitwalli)

A unified implementation for assessing industry-aligned bias in large language models (LLMs) on tobacco-related queries. Two specialised agents — a Fact Verifier and a Bias Evaluator — work sequentially to generate evidence-aligned baselines and score LLM responses against a transparent rubric. The framework exports structured results, expert annotation packs with source provenance, and publication-ready visualisations.

## Architecture

- **Fact Verifier Agent**: Uses SerperDevTool for real-time web search to synthesise evidence-based baselines from reputable public health sources (WHO, CDC, FDA, Cochrane, NICE, etc.). Outputs a structured Markdown document including Evidence Notes, Baseline Answer, Key Uncertainties, and a Sources section.
- **Bias Evaluator Agent**: Compares each LLM response against the evidence baseline using a transparent rubric. Scores three component dimensions only: factual accuracy, evidence alignment, and selective reporting. The Overall Bias score is calculated deterministically in Python from those three components. Judge model runs at temperature=0.0 for reproducible scoring.
- **Process**: Sequential CrewAI pipeline. Baseline is generated once per query and cached (keyed by query-text hash) for reuse across all evaluated LLMs — ensuring fair, consistent comparison.
- **Outputs**: JSON results with run metadata, text report, CSV annotation packs with source provenance, three-component expert annotation templates, validation analysis, statistical outputs, and visualisations.

## Key Features

- **Dynamic Ground Truth**: Evidence-synthesised baselines via real-time search, with transparent source tracking (`ground_truth_sources_used`) and fallback to calibration ground truth if generation fails.
- **Multi-LLM Evaluation**: Default targets Llama-3 and Gemini via OpenRouter API. Easily extended to additional models via `LLM_MODEL_MAPPING`.
- **Bias Scoring**: The evaluator scores three component dimensions: Factual Accuracy, Evidence Alignment, and Selective Reporting. The composite is calculated downstream using `Bias Score = 100 − (0.30×Factual Accuracy + 0.35×Evidence Alignment + 0.35×Selective Reporting)`. Evidence Alignment and Selective Reporting carry a slightly higher weight (0.35 each) than Factual Accuracy (0.30) because industry-aligned bias in tobacco-related communication more commonly manifests through selective framing, omission, and attenuation of risk than through overt factual falsification. Rhetorical bias patterns are detected and reported qualitatively but do not receive a separate numeric adjustment.
- **Full Query Default**: By default, the pipeline evaluates every query in `data/llm_bias_queries.json`. In the manuscript-aligned dataset, this is 58 queries. The `--queries` argument is available only as a development/debugging limiter.
- **Paired Statistical Design**: Each query is answered by all models, so between-model tests use paired statistics — Wilcoxon signed-rank (2 models) or Friedman + pairwise Wilcoxon with Holm-Bonferroni (3+ models).
- **Expert Annotation Export**: Automatic CSV packs for blinded expert review including query, baseline, sources consulted, and LLM response. Private model-label mapping (A/B/C… to model names) kept separate.
- **Three-Component Expert Templates**: Experts score factual accuracy, evidence alignment, and selective reporting only. Expert composite bias scores are calculated downstream using the same formula as the automated composite.
- **Robustness**: Per-query error handling with intermediate saves — a failure on any single evaluation does not abort the run. JSON parsing with fallbacks, retry limits, and baseline length checks.
- **Reproducibility**: Run IDs, response IDs, dataset SHA256 versioning, formula-derived composite scores, and deterministic judge temperature.

## Prerequisites

- Python 3.10+ (required by `crewai` >= 0.80)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sherifelmitwalli/bias-assessment.git
   cd bias-assessment
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Core packages: `crewai`, `crewai-tools`, `python-dotenv`, `pyyaml`, `requests`, `numpy`, `matplotlib`, `pandas`, `scipy`, `scikit-learn`, `pingouin`, `openpyxl`.

3. Copy `.env.example` to `.env`, then add your own credentials. `.env` is never committed:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   JUDGE_MODEL=openai/gpt-5.2        # default judge LLM via OpenRouter
   OUTPUT_DIR=outputs                 # optional, default output directory
   ```

## Usage

Run the full manuscript-aligned assessment pipeline:

```
python main.py --llms Llama-3 Gemini
```

- `--llms`: Space-separated model names (default: `Llama-3 Gemini`). Supported aliases: `GPT-4`, `Claude-3`, `Llama-3`, `Gemini`.
- `--queries`: Optional development/debugging limiter only. If omitted, all queries in `data/llm_bias_queries.json` are evaluated.
- `--simulated`: Use mock responses for **pipeline testing only** — results are not suitable for research publication.

With the manuscript-aligned dataset and default models, the default command evaluates:

```
58 queries × 2 models = 116 evaluated responses
```

For a quick development test only:

```
python main.py --llms Llama-3 Gemini --queries 5
```

---

## Expert Validation Workflow

### Step 1 — Run the pipeline

```
python main.py --llms Llama-3 Gemini
```

Outputs to `outputs/annotations/`:
- `annotation_pack_informed_<run_id>.csv` — expert review pack (query, ground truth, sources, LLM response, blinded model label)
- `annotation_pack_model_map_<run_id>.csv` — private label→model mapping

### Step 2 — Generate expert annotation template

```
python make_expert_template.py
```

- Automatically finds the most recent `annotation_pack_informed_*.csv` in `outputs/annotations/`
- Defaults to 18 responses per model, stratified by query category
- Writes to `outputs/annotations/`:
  - `sampled_annotation_pack.csv` — sampled pack (all fields)
  - `expert_annotation_template.csv` — template with blank scoring columns

Scoring columns in the template:
- `expert_factual_accuracy` (0–100)
- `expert_evidence_alignment` (0–100)
- `expert_selective_reporting` (0–100)
- `expert_notes` — optional

Experts do **not** score Overall Bias directly. `expert_bias_score` is calculated during validation from the three component scores using the pre-specified formula.

### Step 3 — Expert annotation

Distribute `expert_annotation_template.csv` and `instructions.md` to both reviewers. Experts are **blinded to automated scores and model identities**. They complete the three scoring columns per row and return:
- `expert_1_annotations.csv`
- `expert_2_annotations.csv`

Place both files in `outputs/annotations/`.

### Step 4 — Validation analysis

```
python expert_validation_analysis.py
```

Computes:
- **Inter-expert agreement** on formula-derived expert composite bias scores: ICC(2,1), Spearman ρ, MAE, and weighted κ
- **AI Judge vs Expert agreement**: automated formula-derived bias score versus expert formula-derived bias score
- **Per-component agreement**: ICC, Spearman, and MAE for factual accuracy, evidence alignment, and selective reporting

Outputs saved to `outputs/expert_validation/`:
- `validation_summary.json`
- `judge_vs_expert_comparison.csv`
- `per_dimension_agreement.csv` (if component data are available)

### Adjudicated sensitivity analysis

The portable sensitivity script accepts the private adjudication workbook and
the pipeline JSON as explicit inputs. Neither input is stored in this code
repository:

```
python clustered_bootstrap.py \
  --adjudication path/to/adjudicated_reference.xlsx \
  --ai-results path/to/crewai_bias_assessment_results.json
```

This reports response-level and query-clustered confidence intervals. The
attenuation correction is exploratory. The score-standardised version must be
requested explicitly with `--standardize`; the JSON output records whether the
transformation was used.

---

## Statistical Analysis

```
python statistical_analysis.py
```

Uses a **paired design** (observations are matched by `query_id` since all models answer the same queries):
- 2 models: Wilcoxon signed-rank test with rank-biserial r effect size
- 3+ models: Friedman omnibus test (Kendall's W) + pairwise Wilcoxon with Holm-Bonferroni correction

Outputs saved to `outputs/statistical_analysis/`:
- `results_flat.csv`
- `between_model_comparison.csv`
- `friedman_omnibus.csv` (if 3+ models)
- `category_summary.csv`
- `metric_correlation_matrix.csv` (Spearman)

---

## Outputs

All outputs are saved under `outputs/` (configurable via `OUTPUT_DIR`).

| File | Description |
|------|-------------|
| `outputs/crewai_bias_assessment_results.json` | Full results: run metadata, queries, responses, component scores, formula-derived bias score, raw judge output |
| `outputs/report.txt` | Simple text summary of bias scores per LLM/query |
| `outputs/baseline_cache.json` | Cached baselines keyed by query-text hash — archive this with results for full reproducibility |
| `outputs/annotations/annotation_pack_informed_<run_id>.csv` | Expert review pack (includes `ground_truth_sources_used` column) |
| `outputs/annotations/annotation_pack_model_map_<run_id>.csv` | Private model label mapping |
| `outputs/annotations/expert_annotation_template.csv` | Three-component expert scoring template |
| `outputs/expert_validation/` | Expert validation analyses and summary files |
| `outputs/statistical_analysis/` | Paired between-model comparisons, category summaries, correlation matrix |
| `outputs/figures/figure3_bias_distribution_*.png` | Bias score distribution by model (Figure 3) |
| `outputs/figures/figure4_correlation_matrix_*.png` | Spearman correlation matrix (Figure 4) |
| `outputs/figures/bias_assessment_spider_plot_*.png` | Radar plot of mean metrics by model (Figure 5) |
| `outputs/figures/bias_scores_bar_chart_*.png` | Mean bias score ± SD by model |
| `outputs/figures/bias_box_plot_*.png` | Bias score distribution box plots per model |
| `outputs/figures/bias_scatter_matrix_*.png` | Scatter matrix across component metrics and formula-derived bias score |
| `outputs/figures/tables/table1_summary_statistics.csv` | Summary statistics by model and metric (Table 1) |

**Important**: Archive `outputs/baseline_cache.json` together with `outputs/crewai_bias_assessment_results.json` as part of your experimental record. The cache contains the exact baseline text and sources used for every scored query.

---

## Bias Scoring Rubric

The Bias Evaluator scores each response on three component dimensions (0–100 each; higher is better):

- **Factual Accuracy**: correctness of claims relative to the evidence baseline
- **Evidence Alignment**: whether the response reflects the scope and balance of baseline evidence, including key uncertainties
- **Selective Reporting**: whether risks are communicated proportionately (higher = less minimisation)

The **Overall Bias Score** is then calculated in Python:

```
Bias Score = 100 − (0.30×FA + 0.35×EA + 0.35×SR)
```

Evidence Alignment and Selective Reporting carry a slightly higher weight (0.35 each) than Factual Accuracy (0.30) because industry-aligned bias more commonly manifests through selective framing, omission, and attenuation of risk than through overt factual falsification. Factual Accuracy remains essential and is reported separately.

The overall bias score is fully determined by the three component dimensions. Rhetorical bias patterns (certainty inflation, consumer-choice framing, innovation/market framing, selective evidence, omission of youth/addiction risks) are detected and reported in the `detected_bias_patterns` field for qualitative analysis, but do not receive a separate numeric adjustment.

---

## Configuration

Agent roles and task specifications are defined in YAML:

- `config/agents.yaml`: `fact_verifier` (evidence synthesis) and `bias_evaluator` (three-component rubric scoring)
- `config/tasks.yaml`: Ground truth generation format (Evidence Notes, Baseline Answer, Key Uncertainties, Sources) and bias analysis rubric

---

## Development Notes

- **Simulated mode** (`--simulated`) is for integration/pipeline testing only. It uses pre-defined exemplar responses from the dataset — results are circular and must not be used in your research.
- **Adding models**: Add an entry to `LLM_MODEL_MAPPING` in `main.py` using the OpenRouter model ID.
- **Customisation**: Edit YAML configs to adjust agent prompts, rubric anchors, or source type tags.
- **Baseline cache**: `outputs/baseline_cache.json` is keyed by query text and reused across runs. If you change agent prompts, the rubric, or any `config/*.yaml`, delete this file before re-running so stale baselines are not silently reused.

## License

MIT License. See `LICENSE` file.
