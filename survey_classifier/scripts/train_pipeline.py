from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_index import build_index
from src.data_io import load_train_data, parse_codebook
from src.split import save_splits, split_train_val_test
from src.train import DEFAULT_BASE_MODEL, train_model
from src.utils import read_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train model and build production index.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--text-col", default="Ответ")
    parser.add_argument("--codes-col", default="Коды_новые")
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--training-mode", choices=["mnrl", "contrastive", "triplet"], default="contrastive")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--min-class-size", type=int, default=10)
    parser.add_argument("--max-pairs-per-code", type=int, default=5000)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-negatives-per-code", type=int, default=None)
    parser.add_argument("--max-triplets-per-code", type=int, default=None)
    parser.add_argument("--triplet-margin", type=float, default=0.5)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--index-split", choices=["train", "all"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-name", default=None)
    prompt_group.add_argument("--input-prefix", default=None)
    args = parser.parse_args(argv)

    print("Loading training data...")
    train_df = load_train_data(
        args.train,
        args.codebook,
        text_col=args.text_col,
        codes_col=args.codes_col,
        context_col=args.context_col,
        csv_sep=args.csv_sep,
    )
    codebook_df = parse_codebook(args.codebook)
    print(f"Loaded {len(train_df)} training rows across {train_df['code'].nunique()} codes.")

    print("Splitting train/val/test...")
    splits = split_train_val_test(
        train_df,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    split_dir = save_splits(splits, args.out_dir / "splits", seed=args.seed)
    print(
        "Split rows: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}. "
        f"Saved to {split_dir}."
    )
    if splits["train"].empty:
        raise ValueError("Train split is empty. Reduce val_size/test_size or provide more data.")

    print("Training sentence-transformer...")
    model_dir = train_model(
        train_df=splits["train"],
        out_dir=args.out_dir,
        base_model=args.base_model,
        training_mode=args.training_mode,
        min_class_size=args.min_class_size,
        max_pairs_per_code=args.max_pairs_per_code,
        negative_ratio=args.negative_ratio,
        max_negatives_per_code=args.max_negatives_per_code,
        max_triplets_per_code=args.max_triplets_per_code,
        triplet_margin=args.triplet_margin,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        prompt_name=args.prompt_name,
        input_prefix=args.input_prefix,
    )

    print("Building production index...")
    train_config = read_json(args.out_dir / "train_config.json")
    index_df = train_df if args.index_split == "all" else splits["train"]
    index_dir = build_index(
        train_df=index_df,
        codebook_df=codebook_df,
        out_dir=args.out_dir,
        model_dir=model_dir,
        batch_size=args.encode_batch_size,
        input_prefix=str(train_config.get("input_prefix", "")),
        prompt_name_for_config=train_config.get("prompt_name"),
    )
    print(f"Done. Model: {model_dir}. Index: {index_dir}.")


if __name__ == "__main__":
    main()
