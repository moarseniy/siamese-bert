from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, set_global_seed, unique_preserve_order, utc_now_iso, write_json

DEFAULT_BASE_MODEL = "BAAI/bge-m3"


def build_training_pairs(
    train_df: pd.DataFrame,
    min_class_size: int = 10,
    max_pairs_per_code: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"text", "code"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Training dataframe is missing columns: {sorted(missing)}")
    if min_class_size < 2:
        raise ValueError("min_class_size must be at least 2 for siamese pairs.")
    if max_pairs_per_code < 1:
        raise ValueError("max_pairs_per_code must be positive.")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for code, group in train_df.groupby("code", sort=True):
        texts = unique_preserve_order(
            text.strip() for text in group["text"].astype(str).tolist() if text.strip()
        )
        if len(texts) < min_class_size:
            continue

        total_pairs = len(texts) * (len(texts) - 1) // 2
        if total_pairs <= max_pairs_per_code:
            pair_indices = list(itertools.combinations(range(len(texts)), 2))
            rng.shuffle(pair_indices)
        else:
            pair_indices_set: set[tuple[int, int]] = set()
            attempts = 0
            max_attempts = max_pairs_per_code * 50
            while len(pair_indices_set) < max_pairs_per_code and attempts < max_attempts:
                first, second = rng.sample(range(len(texts)), 2)
                if first > second:
                    first, second = second, first
                pair_indices_set.add((first, second))
                attempts += 1
            pair_indices = list(pair_indices_set)

        for first, second in pair_indices[:max_pairs_per_code]:
            records.append({"code": code, "text_a": texts[first], "text_b": texts[second]})

    return pd.DataFrame(records, columns=["code", "text_a", "text_b"])


def train_model(
    train_df: pd.DataFrame,
    out_dir: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    min_class_size: int = 10,
    max_pairs_per_code: int = 5000,
    epochs: int = 1,
    batch_size: int = 16,
    seed: int = 42,
    device: str | None = None,
    show_progress_bar: bool = True,
) -> Path:
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    output_dir = ensure_dir(out_dir)
    model_dir = output_dir / "model"
    set_global_seed(seed)

    pairs_df = build_training_pairs(
        train_df=train_df,
        min_class_size=min_class_size,
        max_pairs_per_code=max_pairs_per_code,
        seed=seed,
    )
    pairs_path = output_dir / "training_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    if pairs_df.empty:
        counts = train_df.groupby("code")["text"].nunique().sort_values(ascending=False)
        top_counts = counts.head(20).to_dict()
        raise ValueError(
            "No positive pairs were generated. Lower min_class_size or check label distribution. "
            f"Top class sizes: {top_counts}"
        )

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model_kwargs: dict[str, Any] = {}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(base_model, **model_kwargs)

    train_examples = [
        InputExample(texts=[row.text_a, row.text_b])
        for row in pairs_df.itertuples(index=False)
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, int(len(train_dataloader) * epochs * 0.1))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=show_progress_bar,
        output_path=str(model_dir),
    )

    config = {
        "created_at": utc_now_iso(),
        "base_model": base_model,
        "model_dir": "model",
        "min_class_size": min_class_size,
        "max_pairs_per_code": max_pairs_per_code,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "n_training_rows": int(len(train_df)),
        "n_training_pairs": int(len(pairs_df)),
    }
    write_json(config, output_dir / "train_config.json")
    return model_dir
