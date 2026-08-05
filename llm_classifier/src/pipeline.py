from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .client import ClassificationResult, VLLMSurveyClassifier
from .data_io import (
    TEXT_COL_DEFAULT,
    UNKNOWN_CODE,
    assignable_codes,
    clean_text,
    combine_text,
    parse_codebook,
    read_table,
    render_codebook,
    write_table,
)
from .metrics import calculate_metrics, metrics_paths


OUTPUT_COLUMNS = [
    "predicted_codes",
    "predicted_names",
    "predicted_parent_codes",
    "predicted_parent_names",
    "confidence",
    "margin",
    "top_candidates",
    "needs_review",
    "invalid_codes",
    "llm_error",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "raw_response",
]


def _result_columns(
    result: ClassificationResult,
    codebook_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    codes = [code for code in result.codes if code != UNKNOWN_CODE]
    names = [
        str(codebook_lookup.get(code, {}).get("name", code))
        for code in codes
    ]
    parent_codes: list[str] = []
    parent_names: list[str] = []
    for code in codes:
        info = codebook_lookup.get(code, {})
        parent = str(info.get("parent_code", "") or "")
        if parent and parent not in parent_codes:
            parent_codes.append(parent)
            parent_names.append(
                str(
                    info.get("parent_name", "")
                    or codebook_lookup.get(parent, {}).get("name", parent)
                )
            )
    return {
        "predicted_codes": ", ".join(result.codes),
        "predicted_names": "; ".join(names) or UNKNOWN_CODE,
        "predicted_parent_codes": ", ".join(parent_codes),
        "predicted_parent_names": "; ".join(parent_names),
        "confidence": None,
        "margin": None,
        "top_candidates": ", ".join(codes),
        "needs_review": result.needs_review,
        "invalid_codes": ", ".join(result.invalid_codes),
        "llm_error": result.error,
        "latency_seconds": round(result.latency_seconds, 4),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "raw_response": result.raw_response,
    }


def _empty_result(reason: str) -> ClassificationResult:
    return ClassificationResult(
        codes=[UNKNOWN_CODE],
        needs_review=True,
        invalid_codes=[],
        error=reason,
        raw_response="",
        latency_seconds=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )


def _classify_safely(classifier: Any, answer: str) -> ClassificationResult:
    started_at = time.perf_counter()
    try:
        return classifier.classify(answer)
    except Exception as exc:
        result = _empty_result(f"{type(exc).__name__}: {exc}")
        result.latency_seconds = time.perf_counter() - started_at
        return result


def process_dataframe(
    frame: pd.DataFrame,
    classifier: Any,
    text_col: str = TEXT_COL_DEFAULT,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 250,
    concurrency: int = 8,
    context_col: str | None = None,
    codebook_lookup: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if text_col not in frame.columns:
        existing = ", ".join(map(str, frame.columns))
        raise ValueError(f"Missing text column {text_col!r}. Existing columns: {existing}")
    if concurrency < 1:
        raise ValueError("concurrency must be positive.")
    if context_col and context_col not in frame.columns:
        raise ValueError(f"Missing context column {context_col!r}.")
    lookup = codebook_lookup or {}

    result = frame.copy()
    for column in OUTPUT_COLUMNS:
        result[column] = pd.Series([None] * len(result), dtype=object)

    completed = 0

    def record_prediction(
        index: Any,
        prediction: ClassificationResult,
        progress: tqdm[Any],
    ) -> None:
        nonlocal completed
        for column, value in _result_columns(prediction, lookup).items():
            result.at[index, column] = value
        completed += 1
        progress.update(1)
        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and completed % checkpoint_every == 0
        ):
            write_table(result, checkpoint_path)

    with tqdm(total=len(result), desc="LLM classification") as progress:
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="vllm-request",
        ) as executor:
            futures: dict[Future[ClassificationResult], Any] = {}
            for index, row in result.iterrows():
                raw_answer = clean_text(row[text_col])
                answer = combine_text(
                    raw_answer,
                    row[context_col] if context_col else None,
                )
                if not raw_answer:
                    record_prediction(index, _empty_result("empty_text"), progress)
                    continue
                future = executor.submit(_classify_safely, classifier, answer)
                futures[future] = index

            for future in as_completed(futures):
                record_prediction(futures[future], future.result(), progress)
    return result


def _run_statistics(
    result: pd.DataFrame,
    wall_time_seconds: float,
    concurrency: int,
) -> dict[str, Any]:
    latencies = pd.to_numeric(result["latency_seconds"], errors="coerce").fillna(0.0)
    errors = result["llm_error"].fillna("").astype(str)
    review_values = result["needs_review"].map(
        lambda value: True if pd.isna(value) else bool(value)
    )
    return {
        "input_rows": int(len(result)),
        "successful_rows": int(errors.eq("").sum()),
        "failed_rows": int(errors.ne("").sum()),
        "needs_review_rows": int(review_values.sum()),
        "needs_review_rate": float(review_values.mean()) if len(result) else 0.0,
        "latency_seconds_total": float(latencies.sum()),
        "latency_seconds_mean": float(latencies.mean()) if len(latencies) else 0.0,
        "latency_seconds_p50": float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
        "latency_seconds_p95": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        "prompt_tokens": int(
            pd.to_numeric(result["prompt_tokens"], errors="coerce").fillna(0).sum()
        ),
        "completion_tokens": int(
            pd.to_numeric(result["completion_tokens"], errors="coerce").fillna(0).sum()
        ),
        "wall_time_seconds": float(wall_time_seconds),
        "throughput_rows_per_second": float(len(result) / wall_time_seconds)
        if wall_time_seconds > 0
        else 0.0,
        "concurrency": concurrency,
    }


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    codebook_path: str | Path,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    model: str | None = None,
    text_col: str = TEXT_COL_DEFAULT,
    context_col: str | None = None,
    gold_codes_col: str | None = None,
    csv_sep: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 2,
    max_tokens: int = 64,
    thinking_max_tokens: int = 1024,
    temperature: float = 0.0,
    thinking_temperature: float = 0.6,
    thinking_top_p: float = 0.95,
    thinking_top_k: int = 20,
    seed: int = 42,
    structured_output: bool = True,
    enable_thinking: bool = False,
    checkpoint_every: int = 250,
    concurrency: int = 8,
    max_labels: int = 6,
) -> tuple[Path, dict[str, Any]]:
    source = read_table(input_path, csv_sep=csv_sep)
    required = [text_col]
    if context_col:
        required.append(context_col)
    if gold_codes_col:
        required.append(gold_codes_col)
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Existing: {list(source.columns)}")
    codebook = parse_codebook(codebook_path)
    allowed_codes = assignable_codes(codebook)
    classifier = VLLMSurveyClassifier(
        base_url=base_url,
        api_key=api_key,
        model=model,
        codebook_text=render_codebook(codebook),
        allowed_codes=allowed_codes,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        thinking_max_tokens=thinking_max_tokens,
        temperature=temperature,
        thinking_temperature=thinking_temperature,
        thinking_top_p=thinking_top_p,
        thinking_top_k=thinking_top_k,
        seed=seed,
        structured_output=structured_output,
        enable_thinking=enable_thinking,
        max_labels=max_labels,
    )
    resolved_model = classifier.resolve_model()

    started_at = time.perf_counter()
    result = process_dataframe(
        frame=source,
        classifier=classifier,
        text_col=text_col,
        checkpoint_path=output_path,
        checkpoint_every=checkpoint_every,
        concurrency=concurrency,
        context_col=context_col,
        codebook_lookup=codebook.set_index("code").to_dict(orient="index"),
    )
    wall_time_seconds = time.perf_counter() - started_at
    saved_output = write_table(result, output_path)

    stats = _run_statistics(
        result,
        wall_time_seconds=wall_time_seconds,
        concurrency=concurrency,
    )
    stats["model"] = resolved_model
    stats["base_url"] = base_url
    stats["structured_output"] = structured_output
    stats["enable_thinking"] = enable_thinking
    stats["max_tokens"] = max_tokens
    stats["thinking_max_tokens"] = thinking_max_tokens
    stats["temperature"] = temperature
    stats["thinking_temperature"] = thinking_temperature
    stats["thinking_top_p"] = thinking_top_p
    stats["thinking_top_k"] = thinking_top_k

    stats_path, per_class_path, errors_path = metrics_paths(output_path)
    if gold_codes_col:
        quality, per_class, errors = calculate_metrics(
            result,
            gold_codes_col=gold_codes_col,
            known_codes=set(allowed_codes),
        )
        stats.update(quality)
        per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")
        errors.to_csv(errors_path, index=False, encoding="utf-8-sig")

    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return saved_output, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify survey answers through concurrent independent requests "
            "to a vLLM OpenAI-compatible API."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path, help="Codebook CSV.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--text-col", default=TEXT_COL_DEFAULT)
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--gold-codes-col", default=None)
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--thinking-max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking-temperature", type=float, default=0.6)
    parser.add_argument("--thinking-top-p", type=float, default=0.95)
    parser.add_argument("--thinking-top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-labels", type=int, default=6)
    parser.add_argument("--no-structured-output", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args(argv)

    output_path, stats = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        codebook_path=args.codebook,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        text_col=args.text_col,
        context_col=args.context_col,
        gold_codes_col=args.gold_codes_col,
        csv_sep=args.csv_sep,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        thinking_max_tokens=args.thinking_max_tokens,
        temperature=args.temperature,
        thinking_temperature=args.thinking_temperature,
        thinking_top_p=args.thinking_top_p,
        thinking_top_k=args.thinking_top_k,
        seed=args.seed,
        structured_output=not args.no_structured_output,
        enable_thinking=args.enable_thinking,
        checkpoint_every=args.checkpoint_every,
        concurrency=args.concurrency,
        max_labels=args.max_labels,
    )
    print(f"Saved predictions to {output_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
