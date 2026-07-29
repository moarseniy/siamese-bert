from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classifier import BertSurveyClassifier
from .data_io import read_table, split_codes, write_table
from .metrics import calculate_metrics, encode_labels


def _write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def classify_file(
    input_path: str | Path,
    output_path: str | Path,
    model_dir: str | Path,
    text_col: str = "Ответ",
    context_col: str | None = None,
    gold_codes_col: str | None = None,
    csv_sep: str | None = None,
    batch_size: int = 64,
    threshold: float | None = None,
    max_labels: int | None = None,
    top_k: int = 5,
    device: str | None = None,
    trust_remote_code: bool | None = None,
) -> Path:
    source = read_table(input_path, csv_sep=csv_sep)
    required = [text_col]
    if context_col:
        required.append(context_col)
    if gold_codes_col:
        required.append(gold_codes_col)
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    classifier = BertSurveyClassifier(
        model_dir,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    contexts = source[context_col].tolist() if context_col else None
    predictions, probabilities = classifier.predict_batch(
        source[text_col].tolist(),
        contexts=contexts,
        batch_size=batch_size,
        threshold=threshold,
        max_labels=max_labels,
        top_k=top_k,
    )
    result = pd.concat(
        [source.reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    )
    output_path = write_table(result, output_path)

    if gold_codes_col:
        known_codes = set(classifier.classes)
        gold_lists = [
            [code for code in split_codes(value) if code in known_codes]
            for value in source[gold_codes_col]
        ]
        labeled_mask = np.array([bool(codes) for codes in gold_lists])
        chosen_threshold = (
            classifier.threshold if threshold is None else float(threshold)
        )
        chosen_max_labels = (
            classifier.max_labels if max_labels is None else int(max_labels)
        )
        y_true = encode_labels(gold_lists, classifier.classes)
        metrics, per_class = calculate_metrics(
            y_true[labeled_mask],
            probabilities[labeled_mask],
            classifier.classes,
            threshold=chosen_threshold,
            max_labels=chosen_max_labels,
        )
        metrics["input_rows"] = int(len(source))
        metrics["evaluated_rows"] = int(labeled_mask.sum())
        metrics["rows_without_known_gold_codes"] = int((~labeled_mask).sum())

        report_prefix = output_path.with_suffix("")
        stats_path = report_prefix.parent / f"{report_prefix.name}_stats.json"
        per_class_path = report_prefix.parent / f"{report_prefix.name}_per_class.csv"
        errors_path = report_prefix.parent / f"{report_prefix.name}_errors.csv"
        _write_json(metrics, stats_path)
        per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")

        predicted_sets = result["predicted_codes"].apply(
            lambda value: {
                code
                for code in split_codes(value)
                if code in known_codes
            }
        )
        gold_sets = pd.Series([set(codes) for codes in gold_lists])
        errors = result[labeled_mask & (predicted_sets != gold_sets)]
        errors.to_csv(errors_path, index=False, encoding="utf-8-sig")
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
        description="Classify CSV/XLSX survey responses with a trained BERT artifact."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--text-col", default="Ответ")
    parser.add_argument("--context-col", default=None)
    parser.add_argument(
        "--gold-codes-col",
        default=None,
        help="Optional label column: calculate metrics and error reports.",
    )
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    remote_group = parser.add_mutually_exclusive_group()
    remote_group.add_argument(
        "--trust-remote-code",
        action="store_const",
        const=True,
        dest="trust_remote_code",
    )
    remote_group.add_argument(
        "--no-trust-remote-code",
        action="store_const",
        const=False,
        dest="trust_remote_code",
    )
    parser.set_defaults(trust_remote_code=None)
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
        batch_size=args.batch_size,
        threshold=args.threshold,
        max_labels=args.max_labels,
        top_k=args.top_k,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
