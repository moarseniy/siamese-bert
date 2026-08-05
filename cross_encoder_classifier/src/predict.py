from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .classifier import CrossEncoderSurveyClassifier
from .data_io import (
    ConflictingSentimentsError,
    parse_annotations,
    read_table,
    write_table,
)
from .metrics import (
    calculate_pair_metrics,
    calculate_response_metrics,
    decode_response_predictions,
)


def _write_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _gold_targets(
    source: pd.DataFrame,
    codes_col: str,
    sentiments_col: str | None,
    classifier: CrossEncoderSurveyClassifier,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    sentiment_column_exists = bool(sentiments_col and sentiments_col in source.columns)
    code_index = {code: index for index, code in enumerate(classifier.codes)}
    targets = np.zeros((len(source), len(classifier.codes)), dtype=np.int8)
    valid_mask = np.ones(len(source), dtype=bool)
    conflicting_rows: list[int] = []
    unknown_codes: set[str] = set()
    for row_position, (_, row) in enumerate(source.iterrows()):
        sentiments_value = row[sentiments_col] if sentiment_column_exists else None
        try:
            annotations = parse_annotations(row[codes_col], sentiments_value)
        except ConflictingSentimentsError:
            valid_mask[row_position] = False
            conflicting_rows.append(row_position + 2)
            continue
        except ValueError as exc:
            raise ValueError(
                f"Invalid gold annotations at source row {row_position + 2}: {exc}"
            ) from exc
        for code, raw_sentiment in annotations:
            if code not in code_index:
                unknown_codes.add(code)
                continue
            targets[row_position, code_index[code]] = raw_sentiment + 1
    if unknown_codes:
        raise ValueError(
            "Gold codes are absent from the active leaf codebook: "
            + ", ".join(sorted(unknown_codes)[:20])
        )
    return targets, valid_mask, conflicting_rows


def classify_file(
    input_path: str | Path,
    output_path: str | Path,
    model_dir: str | Path,
    codebook_path: str | Path | None = None,
    text_col: str = "Ответ",
    context_col: str | None = None,
    gold_codes_col: str | None = None,
    gold_sentiments_col: str | None = None,
    csv_sep: str | None = None,
    batch_size: int = 64,
    threshold: float | None = None,
    max_labels: int | None = None,
    top_k: int = 5,
    margin_threshold: float = 0.05,
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
        raise ValueError(
            f"Missing columns: {missing}. Existing columns: {list(source.columns)}"
        )

    classifier = CrossEncoderSurveyClassifier(
        model_dir,
        codebook_path=codebook_path,
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
        margin_threshold=margin_threshold,
    )
    result = pd.concat(
        [source.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1
    )
    output_path = write_table(result, output_path)

    if gold_codes_col:
        y_true, evaluation_mask, conflicting_rows = _gold_targets(
            source,
            gold_codes_col,
            gold_sentiments_col,
            classifier,
        )
        evaluated_true = y_true[evaluation_mask]
        evaluated_probabilities = probabilities[evaluation_mask]
        chosen_threshold = (
            classifier.threshold if threshold is None else float(threshold)
        )
        chosen_max_labels = (
            classifier.max_labels if max_labels is None else int(max_labels)
        )
        response_metrics, per_code = calculate_response_metrics(
            evaluated_true,
            evaluated_probabilities,
            classifier.codes,
            chosen_threshold,
            chosen_max_labels,
        )
        pair_metrics, pair_per_class = calculate_pair_metrics(
            evaluated_true.reshape(-1),
            evaluated_probabilities.reshape(-1, 4),
        )
        metrics = {
            **response_metrics,
            "input_rows": int(len(source)),
            "evaluated_rows": int(evaluation_mask.sum()),
            "skipped_conflicting_sentiment_rows": int(len(conflicting_rows)),
            "skipped_conflicting_sentiment_source_rows": conflicting_rows,
            "leaf_codes": int(len(classifier.codes)),
            "pair": pair_metrics,
        }
        report_prefix = output_path.with_suffix("")
        stats_path = report_prefix.parent / f"{report_prefix.name}_stats.json"
        per_code_path = report_prefix.parent / f"{report_prefix.name}_per_class.csv"
        pair_class_path = (
            report_prefix.parent / f"{report_prefix.name}_pair_per_class.csv"
        )
        errors_path = report_prefix.parent / f"{report_prefix.name}_errors.csv"
        _write_json(metrics, stats_path)
        per_code.to_csv(per_code_path, index=False, encoding="utf-8-sig")
        pair_per_class.to_csv(pair_class_path, index=False, encoding="utf-8-sig")

        evaluated_predictions = decode_response_predictions(
            evaluated_probabilities,
            chosen_threshold,
            chosen_max_labels,
        )
        error_mask = np.zeros(len(source), dtype=bool)
        error_mask[np.flatnonzero(evaluation_mask)] = (
            evaluated_predictions != evaluated_true
        ).any(axis=1)
        result[error_mask].to_csv(errors_path, index=False, encoding="utf-8-sig")
        print(
            "Skipped gold rows with conflicting code sentiments: "
            f"{len(conflicting_rows)}"
        )
        if evaluation_mask.any():
            print(
                f"Metrics: micro_f1={response_metrics['micro_f1']:.4f}; "
                "sentiment_accuracy="
                f"{response_metrics['gold_code_sentiment_accuracy']}; "
                f"evaluated={int(evaluation_mask.sum())}"
            )
        else:
            print("Metrics: no rows remain after filtering conflicting sentiments.")
        print(
            f"Reports: {stats_path}, {per_code_path}, {pair_class_path}, {errors_path}"
        )

    print(f"Predictions saved to {output_path.resolve()}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify survey codes and code-level sentiment with a cross-encoder."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--codebook",
        type=Path,
        default=None,
        help="Optional replacement codebook for zero-shot scoring.",
    )
    parser.add_argument("--text-col", default="Ответ")
    parser.add_argument("--context-col", default=None)
    parser.add_argument(
        "--gold-codes-col",
        default=None,
        help="Optional code column: calculate metrics and error reports.",
    )
    parser.add_argument(
        "--gold-sentiments-col",
        default=None,
        help="Sentiments aligned with --gold-codes-col (0 neutral, 1 positive, 2 negative).",
    )
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
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
        codebook_path=args.codebook,
        text_col=args.text_col,
        context_col=args.context_col,
        gold_codes_col=args.gold_codes_col,
        gold_sentiments_col=args.gold_sentiments_col,
        csv_sep=args.csv_sep,
        batch_size=args.batch_size,
        threshold=args.threshold,
        max_labels=args.max_labels,
        top_k=args.top_k,
        margin_threshold=args.margin_threshold,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
