from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data_io import assignable_codes, parse_codebook, read_table
from .metrics import calculate_metrics, metrics_paths


def evaluate_predictions(
    predictions_path: str | Path,
    codebook_path: str | Path,
    gold_codes_col: str = "Коды_новые",
    prediction_col: str = "predicted_code_sentiments",
    csv_sep: str | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    predictions = read_table(predictions_path, csv_sep=csv_sep)
    codebook = parse_codebook(codebook_path)
    quality, per_class, errors = calculate_metrics(
        predictions,
        gold_codes_col=gold_codes_col,
        known_codes=set(assignable_codes(codebook)),
        prediction_col=prediction_col,
    )

    stats_path, per_class_path, errors_path = metrics_paths(predictions_path)
    stats: dict[str, Any] = {}
    if stats_path.exists():
        existing = json.loads(stats_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            stats.update(existing)
    stats.update(quality)
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")
    errors.to_csv(errors_path, index=False, encoding="utf-8-sig")
    return stats, stats_path, per_class_path, errors_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calculate code and sentiment metrics from saved predictions."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--gold-codes-col", default="Коды_новые")
    parser.add_argument(
        "--prediction-col",
        default="predicted_code_sentiments",
    )
    parser.add_argument("--csv-sep", default=None)
    args = parser.parse_args(argv)

    stats, stats_path, per_class_path, errors_path = evaluate_predictions(
        predictions_path=args.input,
        codebook_path=args.codebook,
        gold_codes_col=args.gold_codes_col,
        prediction_col=args.prediction_col,
        csv_sep=args.csv_sep,
    )
    print(f"Saved statistics to {stats_path}")
    print(f"Saved per-class metrics to {per_class_path}")
    print(f"Saved errors to {errors_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
