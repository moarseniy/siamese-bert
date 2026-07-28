from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .client import ClassificationResult, VLLMSurveyClassifier
from .data_io import (
    CODES_COL_DEFAULT,
    TEXT_COL_DEFAULT,
    UNKNOWN_CODE,
    assignable_codes,
    clean_text,
    parse_codebook,
    read_table,
    render_codebook,
    write_table,
)
from .metrics import calculate_metrics, metrics_paths


OUTPUT_COLUMNS = [
    "predicted_codes",
    "confidence",
    "needs_review",
    "explanation",
    "invalid_codes",
    "llm_error",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "raw_response",
]


def _result_columns(result: ClassificationResult) -> dict[str, Any]:
    return {
        "predicted_codes": ", ".join(result.codes),
        "confidence": result.confidence,
        "needs_review": result.needs_review,
        "explanation": result.explanation,
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
        confidence=0.0,
        needs_review=True,
        explanation="",
        invalid_codes=[],
        error=reason,
        raw_response="",
        latency_seconds=0.0,
        prompt_tokens=0,
        completion_tokens=0,
    )


def process_dataframe(
    frame: pd.DataFrame,
    classifier: Any,
    text_col: str = TEXT_COL_DEFAULT,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 20,
) -> pd.DataFrame:
    if text_col not in frame.columns:
        existing = ", ".join(map(str, frame.columns))
        raise ValueError(f"Missing text column {text_col!r}. Existing columns: {existing}")

    result = frame.copy()
    for column in OUTPUT_COLUMNS:
        result[column] = pd.Series([None] * len(result), dtype=object)

    for position, (index, row) in enumerate(
        tqdm(result.iterrows(), total=len(result), desc="LLM classification"),
        start=1,
    ):
        answer = clean_text(row[text_col])
        prediction = classifier.classify(answer) if answer else _empty_result("empty_text")
        for column, value in _result_columns(prediction).items():
            result.at[index, column] = value

        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and position % checkpoint_every == 0
        ):
            write_table(result, checkpoint_path)
    return result


def _run_statistics(result: pd.DataFrame) -> dict[str, Any]:
    latencies = pd.to_numeric(result["latency_seconds"], errors="coerce").fillna(0.0)
    errors = result["llm_error"].fillna("").astype(str)
    review_values = result["needs_review"].map(
        lambda value: True if pd.isna(value) else bool(value)
    )
    return {
        "n_rows": int(len(result)),
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
    }


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    codebook_path: str | Path,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    model: str | None = None,
    text_col: str = TEXT_COL_DEFAULT,
    gold_col: str = CODES_COL_DEFAULT,
    csv_sep: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 2,
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
    review_threshold: float = 0.6,
    structured_output: bool = True,
    enable_thinking: bool = False,
    checkpoint_every: int = 20,
) -> tuple[Path, dict[str, Any]]:
    source = read_table(input_path, csv_sep=csv_sep)
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
        temperature=temperature,
        seed=seed,
        review_threshold=review_threshold,
        structured_output=structured_output,
        enable_thinking=enable_thinking,
    )

    result = process_dataframe(
        frame=source,
        classifier=classifier,
        text_col=text_col,
        checkpoint_path=output_path,
        checkpoint_every=checkpoint_every,
    )
    saved_output = write_table(result, output_path)

    stats = _run_statistics(result)
    stats["model"] = classifier.resolve_model()
    stats["base_url"] = base_url
    stats["structured_output"] = structured_output
    stats["enable_thinking"] = enable_thinking
    stats["review_threshold"] = review_threshold

    stats_path, per_class_path, errors_path = metrics_paths(output_path)
    if gold_col in result.columns:
        quality, per_class, errors = calculate_metrics(result, gold_col=gold_col)
        stats["quality"] = quality
        per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")
        errors.to_csv(errors_path, index=False, encoding="utf-8-sig")

    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return saved_output, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify survey answers sequentially through a vLLM OpenAI-compatible API."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--codebook-txt", required=True, type=Path)
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--text-col", default=TEXT_COL_DEFAULT)
    parser.add_argument("--gold-col", default=CODES_COL_DEFAULT)
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-threshold", type=float, default=0.6)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--no-structured-output", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args(argv)

    output_path, stats = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        codebook_path=args.codebook_txt,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        text_col=args.text_col,
        gold_col=args.gold_col,
        csv_sep=args.csv_sep,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        review_threshold=args.review_threshold,
        structured_output=not args.no_structured_output,
        enable_thinking=args.enable_thinking,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Saved predictions to {output_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
