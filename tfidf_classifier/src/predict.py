from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import combine_text, read_table, split_codes, write_table
from .metrics import calculate_metrics, encode_labels
from .model import TfidfSurveyClassifier
from .utils import write_json


def _join_values(values: Any) -> str:
    if isinstance(values, (list, tuple, set)):
        return ", ".join(map(str, values))
    return "" if values is None else str(values)


def _format_candidates(candidates: Any) -> str:
    if not isinstance(candidates, list):
        return ""
    return "; ".join(
        f"{candidate['code']}:{float(candidate['probability']):.4f}"
        for candidate in candidates
    )


def classify_file(
    input_path: str | Path,
    output_path: str | Path,
    model_dir: str | Path,
    text_col: str = "Ответ",
    context_col: str | None = None,
    gold_codes_col: str | None = None,
    csv_sep: str | None = None,
    top_k: int = 5,
    threshold: float | None = None,
    max_labels: int | None = None,
    margin_threshold: float = 0.05,
) -> Path:
    source = read_table(input_path, csv_sep=csv_sep)
    required = [text_col]
    if context_col:
        required.append(context_col)
    if gold_codes_col:
        required.append(gold_codes_col)
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Existing: {list(source.columns)}")

    classifier = TfidfSurveyClassifier.load(model_dir)
    texts = [
        combine_text(
            row[text_col],
            row[context_col] if context_col else None,
        )
        for _, row in source.iterrows()
    ]
    probabilities = classifier.predict_probabilities(texts)
    predictions = classifier.predict_batch(
        texts,
        top_k=top_k,
        threshold=threshold,
        max_labels=max_labels,
        margin_threshold=margin_threshold,
        probabilities=probabilities,
    )
    result = source.reset_index(drop=True).copy()
    result["predicted_codes"] = predictions["predicted_codes"].apply(_join_values)
    result["predicted_names"] = predictions["predicted_names"].apply(_join_values)
    result["predicted_parent_codes"] = predictions["parent_codes"].apply(_join_values)
    result["predicted_parent_names"] = predictions["parent_names"].apply(_join_values)
    result["confidence"] = predictions["confidence"]
    result["margin"] = predictions["margin"]
    result["top_candidates"] = predictions["top_candidates"].apply(_format_candidates)
    result["needs_review"] = predictions["needs_review"]
    output_path = write_table(result, output_path)

    if gold_codes_col:
        known_codes = set(classifier.classes)
        gold_lists = [
            [code for code in split_codes(value) if code in known_codes]
            for value in source[gold_codes_col]
        ]
        labeled_mask = np.array([bool(codes) for codes in gold_lists])
        y_true = encode_labels(gold_lists, classifier.classes)
        selected_threshold = (
            classifier.threshold if threshold is None else float(threshold)
        )
        selected_max_labels = (
            classifier.max_labels if max_labels is None else int(max_labels)
        )
        metrics, per_class = calculate_metrics(
            y_true[labeled_mask],
            probabilities[labeled_mask],
            classifier.classes,
            threshold=selected_threshold,
            max_labels=selected_max_labels,
        )
        metrics["input_rows"] = int(len(source))
        metrics["evaluated_rows"] = int(labeled_mask.sum())
        metrics["rows_without_known_gold_codes"] = int((~labeled_mask).sum())

        report_prefix = output_path.with_suffix("")
        stats_path = report_prefix.parent / f"{report_prefix.name}_stats.json"
        per_class_path = report_prefix.parent / f"{report_prefix.name}_per_class.csv"
        errors_path = report_prefix.parent / f"{report_prefix.name}_errors.csv"
        write_json(metrics, stats_path)
        per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")

        predicted_sets = result["predicted_codes"].apply(
            lambda value: {
                code for code in split_codes(value) if code in known_codes
            }
        )
        gold_sets = pd.Series([set(codes) for codes in gold_lists])
        error_mask = pd.Series(labeled_mask) & (predicted_sets != gold_sets)
        result[error_mask].to_csv(errors_path, index=False, encoding="utf-8-sig")
        print(
            f"Metrics: micro_f1={metrics.get('micro_f1', 0.0):.4f}; "
            f"macro_f1={metrics.get('macro_f1', 0.0):.4f}; "
            f"evaluated={metrics['evaluated_rows']}"
        )
        print(f"Reports: {stats_path}, {per_class_path}, {errors_path}")

    print(f"Predictions saved to {output_path.resolve()}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify CSV/XLSX survey responses with a TF-IDF artifact."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--text-col", default="Ответ")
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--gold-codes-col", default=None)
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    classify_file(
        input_path=args.input,
        output_path=args.output,
        model_dir=args.model_dir,
        text_col=args.text_col,
        context_col=args.context_col,
        gold_codes_col=args.gold_codes_col,
        csv_sep=args.csv_sep,
        top_k=args.top_k,
        threshold=args.threshold,
        max_labels=args.max_labels,
        margin_threshold=args.margin_threshold,
    )


if __name__ == "__main__":
    main()
