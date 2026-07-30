from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .classifier import SurveyClassifier
from .data_io import (
    TEXT_COL_DEFAULT,
    combine_text,
    read_table,
    write_table,
)
from .metrics import calculate_metrics, metrics_paths


def _join_values(values: Any) -> str:
    if not isinstance(values, list):
        return "" if values is None else str(values)
    return ", ".join(str(value) for value in values if str(value))


def _shorten(text: str, limit: int = 120) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def format_top_candidates(candidates: Any) -> str:
    if not isinstance(candidates, list):
        return ""
    parts = []
    for candidate in candidates:
        code = candidate.get("code", "")
        score = candidate.get("similarity", "")
        if code:
            parts.append(f"{code}:{score:.4f}" if isinstance(score, float) else f"{code}:{score}")
    return "; ".join(parts)


def format_nearest_examples(nearest_examples: Any) -> str:
    if not isinstance(nearest_examples, list):
        return ""
    groups: list[str] = []
    for group in nearest_examples:
        code = group.get("code", "")
        examples = group.get("examples", [])
        texts = [f'"{_shorten(example.get("text", ""))}"' for example in examples if example.get("text")]
        if code and texts:
            groups.append(f"{code} -> " + " | ".join(texts))
    return "; ".join(groups)


def _require_text_column(frame: pd.DataFrame, text_col: str, source: Path) -> None:
    if text_col not in frame.columns:
        existing = ", ".join(map(str, frame.columns.tolist()))
        raise ValueError(f"Missing text column {text_col!r} in {source}. Existing columns: {existing}")


def classify_file(
    model_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    text_col: str = TEXT_COL_DEFAULT,
    context_col: str | None = None,
    gold_codes_col: str | None = None,
    csv_sep: str | None = None,
    top_k: int = 5,
    threshold: float = 0.65,
    max_labels: int = 6,
    margin_threshold: float = 0.05,
    nearest_k: int = 2,
    batch_size: int = 64,
) -> Path:
    input_path = Path(input_path)
    source = read_table(input_path, csv_sep=csv_sep)
    _require_text_column(source, text_col, input_path)
    if context_col and context_col not in source.columns:
        raise ValueError(f"Missing context column {context_col!r} in {input_path}.")
    if gold_codes_col and gold_codes_col not in source.columns:
        raise ValueError(f"Missing gold column {gold_codes_col!r} in {input_path}.")

    classifier = SurveyClassifier.load(model_dir)
    texts = [
        combine_text(
            row[text_col],
            row[context_col] if context_col else None,
        )
        for _, row in source.iterrows()
    ]
    predictions = classifier.predict_batch(
        texts,
        top_k=top_k,
        threshold=threshold,
        max_labels=max_labels,
        margin_threshold=margin_threshold,
        nearest_k=nearest_k,
        batch_size=batch_size,
    )

    result = source.copy()
    result["predicted_codes"] = predictions["predicted_codes"].apply(_join_values)
    result["predicted_names"] = predictions["predicted_names"].apply(_join_values)
    result["predicted_parent_codes"] = predictions[
        "predicted_parent_codes"
    ].apply(_join_values)
    result["predicted_parent_names"] = predictions[
        "predicted_parent_names"
    ].apply(_join_values)
    result["confidence"] = predictions["confidence"]
    result["margin"] = predictions["margin"]
    result["top_candidates"] = predictions["top_candidates"].apply(format_top_candidates)
    result["needs_review"] = predictions["needs_review"]
    result["nearest_examples"] = predictions["nearest_examples"].apply(format_nearest_examples)

    saved_path = write_table(result, output_path)
    if gold_codes_col:
        known_codes = set(
            classifier.subcategory_metadata["code"].astype(str).tolist()
        )
        metrics, per_class, errors = calculate_metrics(
            result,
            gold_codes_col=gold_codes_col,
            known_codes=known_codes,
        )
        stats_path, per_class_path, errors_path = metrics_paths(saved_path)
        stats_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")
        errors.to_csv(errors_path, index=False, encoding="utf-8-sig")
    return saved_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify CSV/XLSX survey answers with a sentence-transformer index."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-col", default=TEXT_COL_DEFAULT)
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--gold-codes-col", default=None)
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--max-labels", type=int, default=6)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--nearest-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    output_path = classify_file(
        model_dir=args.model_dir,
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        context_col=args.context_col,
        gold_codes_col=args.gold_codes_col,
        csv_sep=args.csv_sep,
        top_k=args.top_k,
        threshold=args.threshold,
        max_labels=args.max_labels,
        margin_threshold=args.margin_threshold,
        nearest_k=args.nearest_k,
        batch_size=args.batch_size,
    )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
