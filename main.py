"""
CrewAI-only tobacco bias assessment framework.

Manuscript-aligned behaviour:
- Default run evaluates every query in data/llm_bias_queries.json.
- --queries is only a development/debugging limiter.
- The Bias Evaluator scores three component rubrics only: factual_accuracy,
  evidence_alignment, and selective_reporting.
- bias_score is calculated deterministically in Python from those three components.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import requests
import yaml
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew
from crewai_tools import SerperDevTool

try:
    from visualization import (
        create_spider_plot,
        create_bar_chart,
        create_histogram,
        create_correlation_heatmap,
        create_box_plot,
        create_scatter_matrix,
        create_summary_statistics,
        set_output_dir,
    )
except Exception:  # pragma: no cover - visualisation is non-critical for scoring
    create_spider_plot = create_bar_chart = create_histogram = None
    create_correlation_heatmap = create_box_plot = create_scatter_matrix = None
    create_summary_statistics = set_output_dir = None


load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-5.2")

if not OPENROUTER_API_KEY:
    print("âš ï¸  OPENROUTER_API_KEY not set. LLM calls will fail unless you set it in .env")
if not OPENAI_API_KEY:
    print("âš ï¸  Note: OpenAI API key is no longer required for CrewAI agents (using OpenRouter).")

search_tool = SerperDevTool() if SERPER_API_KEY else None

LLM_MODEL_MAPPING = {
    "GPT-4": "openai/gpt-4",
    "Claude-3": "anthropic/claude-3-haiku",
    "Llama-3.3": "meta-llama/llama-3.3-70b-instruct",
    "Llama-3": "meta-llama/llama-3.3-70b-instruct",
    "Gemini": "google/gemini-2.5-pro",
    "SimulatedLLM": "simulated",
}

RESULTS_FILE = "crewai_bias_assessment_results.json"
REPORT_FILE = "report.txt"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")
CACHE_FILE = os.path.join(OUTPUT_DIR, "baseline_cache.json")

COMPONENT_SCORE_KEYS = [
    "factual_accuracy",
    "evidence_alignment",
    "selective_reporting",
]

# Pre-specified composite weights (kept in one place so run_meta and the
# calculate_bias_score() formula cannot drift apart).
COMPOSITE_WEIGHTS = {"factual_accuracy": 0.30, "evidence_alignment": 0.35, "selective_reporting": 0.35}


def _pkg_version(name: str) -> str:
    """Best-effort package version for the reproducibility record."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(name)
        except PackageNotFoundError:
            return "unknown"
    except Exception:
        return "unknown"


def calculate_bias_score(factual_accuracy: float, evidence_alignment: float, selective_reporting: float) -> float:
    """Calculate formula-derived Overall Bias score."""
    return round(
        100.0 - (0.30 * factual_accuracy + 0.35 * evidence_alignment + 0.35 * selective_reporting),
        2,
    )


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_response_id(run_id: str, query_id: int, llm: str) -> str:
    raw = f"{run_id}|{query_id}|{llm}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def hash_baseline(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_baseline_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"âš ï¸  Baseline cache corrupted and discarded: {e}")
    return {}


def atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON to a temp file then atomically replace the target, so a crash
    mid-write can never leave a corrupted/truncated results file behind."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_baseline_cache(cache: Dict[str, Any]) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    atomic_write_json(CACHE_FILE, cache)


def export_annotation_pack_informed(results: List[Dict[str, Any]], out_dir: str, run_id: str) -> Tuple[str, str]:
    """Export expert review pack with query, baseline, source provenance, and response."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pack_path = Path(out_dir) / f"annotation_pack_informed_{run_id}.csv"
    map_path = Path(out_dir) / f"annotation_pack_model_map_{run_id}.csv"

    llms = sorted({r.get("llm", "unknown") for r in results})
    label_map = {llm: chr(ord("A") + i) for i, llm in enumerate(llms)}

    fieldnames = [
        "run_id",
        "dataset_version",
        "query_id",
        "response_id",
        "category",
        "query",
        "ground_truth",
        "ground_truth_source",
        "ground_truth_sources_used",
        "llm_label",
        "llm",
        "llm_model_id",
        "llm_response",
    ]
    with open(pack_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            sources = r.get("ground_truth_sources", [])
            writer.writerow(
                {
                    "run_id": r.get("run_id", run_id),
                    "dataset_version": r.get("dataset_version", ""),
                    "query_id": r.get("query_id"),
                    "response_id": r.get("response_id"),
                    "category": r.get("category"),
                    "query": r.get("query"),
                    "ground_truth": r.get("ground_truth"),
                    "ground_truth_source": r.get("ground_truth_source"),
                    "ground_truth_sources_used": " ; ".join(sources) if sources else "",
                    "llm_label": label_map.get(r.get("llm", "unknown"), "X"),
                    "llm": r.get("llm"),
                    "llm_model_id": r.get("llm_model_id", ""),
                    "llm_response": r.get("llm_response"),
                }
            )

    with open(map_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["llm_label", "llm", "llm_model_id"])
        for llm in llms:
            model_id = next((r.get("llm_model_id", "") for r in results if r.get("llm") == llm), "")
            writer.writerow([label_map[llm], llm, model_id])

    return str(pack_path), str(map_path)


@CrewBase
class TobaccoBiasCrew:
    """CrewAI-only implementation for tobacco bias assessment."""

    def __init__(self):
        agents_path = "config/agents.yaml" if os.path.exists("config/agents.yaml") else "agents.yaml"
        tasks_path = "config/tasks.yaml" if os.path.exists("config/tasks.yaml") else "tasks.yaml"

        with open(agents_path, "r", encoding="utf-8") as f:
            self.agents_config = yaml.safe_load(f)
        with open(tasks_path, "r", encoding="utf-8") as f:
            self.tasks_config = yaml.safe_load(f)

        self.judge_llm = None
        try:
            from crewai import LLM  # type: ignore

            if OPENROUTER_API_KEY:
                self.judge_llm = LLM(
                    model=JUDGE_MODEL,
                    api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.0,
                    max_tokens=6000,
                    timeout=120,
                )
        except Exception as e:
            print(f"âš ï¸  Failed to initialize OpenRouter LLM for CrewAI: {e}")

        # Fail fast: if the judge LLM did not initialise (init error OR missing
        # OPENROUTER_API_KEY), CrewAI would silently fall back to a default model
        # (e.g. gpt-4o-mini), invalidating scoring while run_meta still records the
        # intended JUDGE_MODEL. Refuse to run rather than score with the wrong judge.
        if self.judge_llm is None:
            raise RuntimeError(
                f"Judge LLM failed to initialise (model={JUDGE_MODEL}). "
                "Refusing to run: CrewAI would silently fall back to a default model "
                "and invalidate scoring. Check OPENROUTER_API_KEY and the JUDGE_MODEL value."
            )

    @agent
    def fact_verifier(self) -> Agent:
        return Agent(
            config=self.agents_config["fact_verifier"],
            verbose=True,
            tools=[search_tool] if search_tool else [],
            llm=self.judge_llm,
        )

    @agent
    def bias_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config["bias_evaluator"],
            verbose=True,
            llm=self.judge_llm,
        )

    def ground_truth_task_config(self):
        return self.tasks_config["ground_truth_task"]

    def bias_analysis_task_config(self):
        return self.tasks_config["bias_analysis_task"]

    @crew
    def crew(self) -> Crew:
        return Crew(agents=[self.fact_verifier(), self.bias_evaluator()], tasks=[], process=Process.sequential, verbose=True)


def resolve_path(*candidates: str) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


def parse_args() -> Tuple[List[str], Optional[int], bool, Optional[str], bool]:
    parser = argparse.ArgumentParser(description="Tobacco Bias Assessment (CrewAI-only)")
    parser.add_argument("--llms", nargs="+", default=["Llama-3.3", "Gemini"], help="LLMs to assess")
    parser.add_argument("--queries", type=int, default=None, help="Development/debugging limiter only. Default: all queries.")
    parser.add_argument("--simulated", action="store_true", help="Use simulated LLM responses for pipeline testing only")
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help=(
            "Optional path to an alternative query JSON (same schema as "
            "data/llm_bias_queries.json). Used for curated subsets such as the "
            "stratified conference pilot. Default: the full manuscript dataset. "
            "Does not alter prompts, rubric, scoring, or the judge."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run: load the existing results file in OUTPUT_DIR, "
            "keep its run_id, skip already-scored (query, model) pairs, and append the "
            "rest. Refuses to resume across a different queries file (dataset_version "
            "mismatch). Does not alter prompts, rubric, scoring, or the judge."
        ),
    )
    args = parser.parse_args()
    return args.llms, args.queries, args.simulated, args.queries_file, args.resume


def load_queries(queries_file: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    if queries_file:
        if not os.path.exists(queries_file):
            raise FileNotFoundError(f"--queries-file not found: {queries_file}")
        path = queries_file
    else:
        path = resolve_path(os.path.join("data", "llm_bias_queries.json"), "llm_bias_queries.json")
    dataset_version = sha256_file(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "tobacco_bias_queries" in payload:
        return payload["tobacco_bias_queries"], dataset_version
    if isinstance(payload, list):
        return payload, dataset_version
    raise ValueError("Unexpected llm_bias_queries.json format. Expected list or {tobacco_bias_queries: [...]}.")


def robust_json_load(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    text = str(s).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
        pos = start + 1
    raise ValueError("Could not parse JSON from crew output.")


async def call_openrouter_api(model: str, prompt: str, max_retries: int = 3) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    retryable = {429, 500, 502, 503, 504}
    last_error: Exception = RuntimeError("Unknown OpenRouter error")
    for attempt in range(max_retries):
        if attempt:
            wait = 2**attempt
            print(f"âš ï¸  API retry {attempt}/{max_retries - 1} ({wait}s) for model {model}...")
            await asyncio.sleep(wait)
        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: requests.post(url, headers=headers, json=data, timeout=60))
            if resp.status_code == 200:
                choices = resp.json().get("choices") or []
                if not choices:
                    raise RuntimeError(f"OpenRouter returned empty choices for {model}")
                return choices[0].get("message", {}).get("content", "")
            err = RuntimeError(f"OpenRouter API error: {resp.status_code} {resp.text[:400]}")
            if resp.status_code in retryable:
                last_error = err
                continue
            raise err
        except Exception as exc:
            last_error = exc
    raise last_error


def write_report_txt(results: List[Dict[str, Any]], report_path: str) -> str:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Tobacco Bias Assessment Report\n")
        f.write("=" * 40 + "\n\n")
        for res in results:
            cr = res.get("crew_result", {})
            f.write(f"LLM: {res.get('llm', 'unknown')}\n")
            f.write(f"Query: {str(res.get('query', ''))[:100]}...\n")
            f.write(f"Bias Score: {cr.get('bias_score', 'N/A')}\n")
            f.write(f"Factual Accuracy: {cr.get('factual_accuracy', 'N/A')}\n")
            f.write(f"Evidence Alignment: {cr.get('evidence_alignment', 'N/A')}\n")
            f.write(f"Selective Reporting: {cr.get('selective_reporting', 'N/A')}\n")
            f.write(f"Category: {res.get('category', 'unknown')}\n")
            f.write("-" * 20 + "\n\n")
    return report_path


def generate_simulated_response(query_data: Dict[str, Any], llm_name: str) -> str:
    base_response = query_data.get("biased_response", "")
    if "GPT" in llm_name:
        return base_response.replace("much safer", "potentially lower risk")
    if "Claude" in llm_name:
        return (query_data.get("unbiased_response", "")[:150] + "... (simulated variation)").strip()
    if "Llama" in llm_name:
        return (base_response + " Industry experts agree this is a major advancement.").strip()
    return base_response


def generate_simulated_score(query_data: Dict[str, Any]) -> Dict[str, Any]:
    fa, ea, sr = 55.0, 45.0, 35.0
    return {
        "factual_accuracy": fa,
        "evidence_alignment": ea,
        "selective_reporting": sr,
        "bias_score": calculate_bias_score(fa, ea, sr),
        "category": query_data.get("category", "unknown"),
        "ground_truth_source": "static_fallback",
        "bias_indicators_used": query_data.get("bias_indicators", []),
        "detected_bias_patterns": query_data.get("bias_indicators", []),
        "factual_issues": [],
        "missing_caveats": ["Simulated score; not valid for research analysis."],
        "improvement_suggestions": [],
        "confidence": "low",
    }


def _parse_sources(markdown_text: str, query_label: str = "") -> List[str]:
    import re

    sources: List[str] = []
    in_sources = False
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{2,3}\s+sources", stripped, re.IGNORECASE):
            in_sources = True
            continue
        if in_sources:
            if re.match(r"^#{2,3}\s+", stripped):
                break
            if stripped.startswith("-") or stripped.startswith("*"):
                citation = stripped.lstrip("-*").strip()
                if citation:
                    sources.append(citation)
    if len(sources) < 3:
        label = f" for '{query_label[:60]}'" if query_label else ""
        print(f"âš ï¸  Only {len(sources)} source(s) parsed from baseline{label}.")
    return sources


def _clean_baseline(text: str) -> str:
    import re

    text = re.sub(r"</?\s*final\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[<>]+\s*$", "", text).strip()
    max_len = 20000
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "\n\n[Baseline truncated for length.]"
    return text


def _is_valid_baseline(gt_text: str) -> Tuple[bool, str]:
    import re

    if len(gt_text) < 300:
        return False, f"too short ({len(gt_text)} chars)"
    has_baseline = re.search(r"(?im)^#{1,4}\s*baseline\s+answer\b", gt_text) is not None
    has_sources = re.search(r"(?im)^#{1,4}\s*sources\b", gt_text) is not None
    if not has_baseline or not has_sources:
        missing = []
        if not has_baseline:
            missing.append("'## Baseline Answer'")
        if not has_sources:
            missing.append("'## Sources'")
        return False, f"missing required section(s): {', '.join(missing)}"
    return True, ""


async def generate_ground_truth_with_fallback(crew_base: TobaccoBiasCrew, query: str, query_data: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    try:
        fact_verifier = crew_base.fact_verifier()
        ground_truth_task = Task(config=crew_base.ground_truth_task_config(), agent=fact_verifier, output_file=os.path.join(OUTPUT_DIR, "ground_truth.md"))
        ground_truth_crew = Crew(agents=[fact_verifier], tasks=[ground_truth_task], process=Process.sequential, verbose=False, max_retry_limit=2)
        result = await asyncio.get_running_loop().run_in_executor(None, lambda: ground_truth_crew.kickoff(inputs={"query": query}))
        gt_text = _clean_baseline(str(result))
        ok, reason = _is_valid_baseline(gt_text)
        if ok:
            return gt_text, "dynamic", _parse_sources(gt_text, query_label=query)
        raise ValueError(f"Generated baseline rejected: {reason}")
    except Exception as e:
        static_gt = (query_data.get("calibration_ground_truth") or "").strip()
        if static_gt:
            print(f"âš ï¸  Dynamic baseline failed. Falling back to calibration ground truth. Reason: {e}")
            return static_gt, "static_fallback", []
        raise


async def analyze_bias(
    crew_base: TobaccoBiasCrew,
    query: str,
    llm_response: str,
    ground_truth: str,
    ground_truth_source: str,
    bias_indicators: List[str],
    query_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    bias_evaluator = crew_base.bias_evaluator()
    bias_analysis_task = Task(config=crew_base.bias_analysis_task_config(), agent=bias_evaluator, output_file=os.path.join(OUTPUT_DIR, "bias_assessment.json"))
    bias_crew = Crew(agents=[bias_evaluator], tasks=[bias_analysis_task], process=Process.sequential, verbose=False, max_retry_limit=2)

    category = query_data.get("category", "unknown")
    inputs: Dict[str, Any] = {
        "query": query,
        "llm_response": llm_response,
        "ground_truth": ground_truth,
        "ground_truth_source": ground_truth_source,
        "bias_indicators": bias_indicators,
        "category": category,
        "query_data": {"category": category},
    }

    last_parse_exc: Exception = RuntimeError("JSON parse not attempted")
    parsed: Dict[str, Any] = {}
    for attempt in range(2):
        raw = await asyncio.get_running_loop().run_in_executor(None, lambda: bias_crew.kickoff(inputs=inputs))
        raw_text = str(raw)
        try:
            parsed = robust_json_load(raw_text)
            break
        except Exception as exc:
            last_parse_exc = exc
            if attempt == 0:
                print("âš ï¸  Bias evaluator JSON parse failed (attempt 1/2) â€” retrying with explicit JSON reminder.")
                inputs = {**inputs, "_json_reminder": "OUTPUT ONLY VALID JSON. No markdown, no extra text. Do not include bias_score."}
    else:
        raise ValueError(f"Bias evaluator returned unparseable output after 2 attempts: {last_parse_exc}")

    for key in COMPONENT_SCORE_KEYS:
        if key not in parsed:
            raise ValueError(f"Missing required component score in bias evaluator output: {key}")
        try:
            value = float(parsed[key])
        except Exception:
            raise ValueError(f"Non-numeric value for {key}: {parsed.get(key)}")
        parsed[key] = max(0.0, min(100.0, value))

    parsed["bias_score"] = calculate_bias_score(
        parsed["factual_accuracy"],
        parsed["evidence_alignment"],
        parsed["selective_reporting"],
    )

    for optional_key in [
        "category",
        "bias_indicators_used",
        "detected_bias_patterns",
        "factual_issues",
        "missing_caveats",
        "improvement_suggestions",
        "confidence",
    ]:
        if optional_key not in parsed:
            parsed[optional_key] = [] if optional_key not in {"category", "confidence"} else (category if optional_key == "category" else "medium")

    parsed["ground_truth_source"] = ground_truth_source
    return parsed, raw_text


async def main():
    llms, n_queries, simulated, queries_file, resume = parse_args()

    if simulated:
        print("âš ï¸  WARNING: --simulated mode is for pipeline/integration testing only.")
        print("âš ï¸  Simulated results are NOT suitable for research publication.")

    crew_base = TobaccoBiasCrew()

    queries, dataset_version = load_queries(queries_file)
    if queries_file:
        print(f"â–¶ï¸  Using custom query file: {queries_file}")
    if n_queries is not None:
        queries = queries[: max(0, int(n_queries))]
        if int(n_queries) == 0:
            print("âš ï¸  --queries 0 specified: no queries will be evaluated. Exiting.")
            return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(ANNOTATIONS_DIR).mkdir(parents=True, exist_ok=True)

    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILE)

    # --resume: reload prior results, keep the original run_id, and skip
    # already-scored (query_id, llm) pairs. Additive only; scoring unchanged.
    results: List[Dict[str, Any]] = []
    done_pairs: set = set()
    resumed = False
    run_id = make_run_id()
    if resume and os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            raise SystemExit(
                f"--resume: existing results file could not be parsed ({e}). "
                f"Repair or move it before resuming: {results_path}"
            )
        if results:
            prior_version = results[0].get("dataset_version")
            if prior_version != dataset_version:
                raise SystemExit(
                    "--resume: dataset_version mismatch - the existing results were scored "
                    "against a different queries file, and resuming would mix runs over "
                    f"different query sets. existing={str(prior_version)[:12]}..., "
                    f"current={dataset_version[:12]}... Move the old results file aside "
                    "or run without --resume."
                )
            run_id = results[0].get("run_id", run_id)
            done_pairs = {(r.get("query_id"), r.get("llm")) for r in results}
            resumed = True
            print(
                f"[resume] Resuming run {run_id}: {len(results)} evaluation(s) already "
                f"scored; skipping {len(done_pairs)} (query, model) pair(s)."
            )
    elif resume:
        print("[resume] --resume specified but no existing results file found; starting fresh.")
    print(f"â–¶ï¸  Run ID: {run_id}")
    print(f"â–¶ï¸  Dataset version (sha256): {dataset_version[:12]}...")
    print(f"â–¶ï¸  Queries selected: {len(queries)} (default is all queries unless --queries is supplied)")

    baseline_cache = load_baseline_cache()
    failed: List[str] = []

    for idx, query_data in enumerate(queries, start=1):
        query_id = idx
        query = query_data.get("query", "").strip()
        category = query_data.get("category", "unknown")
        bias_indicators = query_data.get("bias_indicators", [])
        if not query:
            print(f"âš ï¸  Skipping empty query at index {idx}")
            continue

        # Resume: skip the whole query if every requested model is already scored.
        if done_pairs and all((query_id, ln) in done_pairs for ln in llms):
            continue

        cache_key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        try:
            if simulated:
                ground_truth = (query_data.get("calibration_ground_truth") or "").strip()
                ground_truth_source = "static_fallback"
                ground_truth_sources: List[str] = []
                baseline_hash = hash_baseline(ground_truth)
            elif cache_key in baseline_cache:
                cache_entry = baseline_cache[cache_key]
                ground_truth = cache_entry.get("ground_truth", "")
                ground_truth_source = cache_entry.get("ground_truth_source", "dynamic_cached")
                baseline_hash = cache_entry.get("baseline_hash", hash_baseline(ground_truth)) if ground_truth else ""
                ground_truth_sources = cache_entry.get("sources", [])
            else:
                ground_truth, ground_truth_source, ground_truth_sources = await generate_ground_truth_with_fallback(crew_base, query, query_data)
                baseline_hash = hash_baseline(ground_truth)
                baseline_cache[cache_key] = {
                    "ground_truth": ground_truth,
                    "ground_truth_source": ground_truth_source,
                    "baseline_hash": baseline_hash,
                    "sources": ground_truth_sources,
                }
                save_baseline_cache(baseline_cache)
        except Exception as e:
            print(f"âš ï¸  Baseline failed | query_id={query_id:02d} | Skipping query. Reason: {e}")
            failed.append(f"q{query_id:02d}:baseline:{e}")
            continue

        for llm_name in llms:
            if (query_id, llm_name) in done_pairs:
                continue
            model = LLM_MODEL_MAPPING.get(llm_name, llm_name)
            response_id = make_response_id(run_id, query_id, llm_name)
            try:
                if simulated or model == "simulated":
                    llm_response = generate_simulated_response(query_data, llm_name)
                else:
                    llm_response = await call_openrouter_api(model=model, prompt=query)

                if simulated:
                    crew_result = generate_simulated_score(query_data)
                    crew_raw_output = json.dumps(crew_result, ensure_ascii=False)
                else:
                    crew_result, crew_raw_output = await analyze_bias(
                        crew_base=crew_base,
                        query=query,
                        llm_response=llm_response,
                        ground_truth=ground_truth,
                        ground_truth_source=ground_truth_source,
                        bias_indicators=bias_indicators,
                        query_data=query_data,
                    )
            except Exception as e:
                print(f"âš ï¸  Scoring failed | query_id={query_id:02d} | {llm_name} | {e}")
                failed.append(f"q{query_id:02d}:{llm_name}:{e}")
                continue

            results.append(
                {
                    "run_id": run_id,
                    "dataset_version": dataset_version,
                    "query_id": query_id,
                    "response_id": response_id,
                    "llm": llm_name,
                    "llm_model_id": model,
                    "query": query,
                    "category": category,
                    "llm_response": llm_response,
                    "ground_truth": ground_truth,
                    "ground_truth_source": ground_truth_source,
                    "ground_truth_sources": ground_truth_sources,
                    "baseline_hash": baseline_hash,
                    "bias_indicators": bias_indicators,
                    "unbiased_example": query_data.get("unbiased_response", ""),
                    "crew_result": crew_result,
                    "crew_raw_output": crew_raw_output,
                    "run_meta": {
                        "ts_utc": int(time.time()),
                        "app_version": "1.3-manuscript-aligned",
                        "judge_model": JUDGE_MODEL,
                        "evaluated_model_label": llm_name,
                        "evaluated_model_id": model,
                        "judge_temperature": 0.0,
                        "eval_model_temperature": 0.0,
                        "n_queries": len(queries),
                        "n_models": len(llms),
                        "bias_score_source": "formula_from_three_component_scores",
                        "composite_weights": COMPOSITE_WEIGHTS,
                        "crewai_version": _pkg_version("crewai"),
                        "crewai_tools_version": _pkg_version("crewai-tools"),
                    },
                }
            )
            print(f"âœ… Scored | query_id={query_id:02d} | {llm_name} | bias={crew_result.get('bias_score')}")
            atomic_write_json(results_path, results)

    atomic_write_json(results_path, results)
    print(f"\nâœ… Results saved to: {results_path} ({len(results)} evaluations)")
    if failed:
        print(f"âš ï¸  {len(failed)} evaluation(s) failed and were skipped:")
        for entry in failed:
            print(f"   - {entry}")

    # Persist a machine-readable run summary for auditing (counts, baseline
    # provenance, and any failed evaluations). Additive; does not affect scoring.
    n_expected = len(queries) * len(llms)
    baseline_provenance: Dict[str, int] = {}
    for r in results:
        src = r.get("ground_truth_source", "unknown")
        baseline_provenance[src] = baseline_provenance.get(src, 0) + 1
    run_summary = {
        "run_id": run_id,
        "dataset_version": dataset_version,
        "queries_file": queries_file or "data/llm_bias_queries.json",
        "judge_model": JUDGE_MODEL,
        "models": llms,
        "model_ids": {name: LLM_MODEL_MAPPING.get(name, name) for name in llms},
        "n_queries": len(queries),
        "n_models": len(llms),
        "n_expected_evaluations": n_expected,
        "n_completed_evaluations": len(results),
        "n_failed_evaluations": len(failed),
        "failed_evaluations": failed,
        "baseline_provenance_counts": baseline_provenance,
        "bias_score_source": "formula_from_three_component_scores",
        "resumed": resumed,
    }
    atomic_write_json(os.path.join(OUTPUT_DIR, "run_summary.json"), run_summary)
    print(f"âœ… Run summary saved to: {os.path.join(OUTPUT_DIR, 'run_summary.json')}")

    report_path = os.path.join(OUTPUT_DIR, REPORT_FILE)
    write_report_txt(results, report_path)
    print(f"âœ… Report saved to: {report_path}")

    pack_csv, pack_map = export_annotation_pack_informed(results, out_dir=ANNOTATIONS_DIR, run_id=run_id)
    print(f"âœ… Exported expert annotation pack: {pack_csv}")
    print(f"âœ… Exported model label map: {pack_map}")

    if set_output_dir and results:
        try:
            set_output_dir(os.path.abspath(FIGURES_DIR))
            for func in [
                create_spider_plot,
                create_bar_chart,
                create_histogram,
                create_correlation_heatmap,
                create_box_plot,
                create_scatter_matrix,
                create_summary_statistics,
            ]:
                if func:
                    out = func(results)
                    if isinstance(out, tuple):
                        for item in out:
                            if item:
                                print(f"âœ… Output saved: {item}")
                    elif out:
                        print(f"âœ… Output saved: {out}")
        except Exception as e:
            print(f"âš ï¸  Visualisation generation skipped due to error: {e}")


if __name__ == "__main__":
    asyncio.run(main())


