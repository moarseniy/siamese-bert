from __future__ import annotations

import argparse
from pathlib import Path

from .model import train_tfidf
from .utils import read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a standalone multi-label TF-IDF survey classifier."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path, help="Codebook CSV.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--text-col", default="Ответ")
    parser.add_argument("--codes-col", default="Коды_новые")
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-max-features", type=int, default=100_000)
    parser.add_argument("--char-max-features", type=int, default=150_000)
    parser.add_argument("--classifier-c", type=float, default=4.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-labels", type=int, default=6)
    parser.add_argument(
        "--threshold-metric",
        choices=["micro_f1", "macro_f1"],
        default="micro_f1",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifact_path = train_tfidf(
        train_path=args.train,
        codebook_path=args.codebook,
        output_dir=args.out_dir,
        text_col=args.text_col,
        codes_col=args.codes_col,
        context_col=args.context_col,
        csv_sep=args.csv_sep,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        min_df=args.min_df,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        classifier_c=args.classifier_c,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        threshold_metric=args.threshold_metric,
        max_labels=args.max_labels,
    )
    config = read_json(args.out_dir / "tfidf_config.json")
    test_metrics = config["metrics"]["test"]
    print(f"Saved TF-IDF model to {artifact_path.resolve()}")
    print(
        f"Test: micro_f1={test_metrics.get('micro_f1', 0.0):.4f}; "
        f"macro_f1={test_metrics.get('macro_f1', 0.0):.4f}; "
        f"threshold={config['threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
