from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_index import build_index
from src.data_io import load_train_data, parse_codebook
from src.train import DEFAULT_BASE_MODEL, train_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train model and build production index.")
    parser.add_argument("--train-xlsx", required=True, type=Path)
    parser.add_argument("--codebook-txt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--min-class-size", type=int, default=10)
    parser.add_argument("--max-pairs-per-code", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    print("Loading training data...")
    train_df = load_train_data(args.train_xlsx, args.codebook_txt)
    codebook_df = parse_codebook(args.codebook_txt)
    print(f"Loaded {len(train_df)} training rows across {train_df['code'].nunique()} codes.")

    print("Training sentence-transformer...")
    model_dir = train_model(
        train_df=train_df,
        out_dir=args.out_dir,
        base_model=args.base_model,
        min_class_size=args.min_class_size,
        max_pairs_per_code=args.max_pairs_per_code,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    print("Building production index...")
    index_dir = build_index(
        train_df=train_df,
        codebook_df=codebook_df,
        out_dir=args.out_dir,
        model_dir=model_dir,
        batch_size=args.encode_batch_size,
    )
    print(f"Done. Model: {model_dir}. Index: {index_dir}.")


if __name__ == "__main__":
    main()
