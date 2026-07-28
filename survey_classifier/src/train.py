from __future__ import annotations

import itertools
import math
import random
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .model_input import configure_model_input, prepare_model_text
from .utils import ensure_dir, set_global_seed, unique_preserve_order, utc_now_iso, write_json

DEFAULT_BASE_MODEL = "BAAI/bge-m3"
TrainingMode = Literal["mnrl", "contrastive", "triplet"]


def _sample_pair_indices(n_items: int, max_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    total_pairs = n_items * (n_items - 1) // 2
    if total_pairs <= max_pairs:
        pair_indices = list(itertools.combinations(range(n_items), 2))
        rng.shuffle(pair_indices)
        return pair_indices

    pair_indices_set: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max_pairs * 50
    while len(pair_indices_set) < max_pairs and attempts < max_attempts:
        first, second = rng.sample(range(n_items), 2)
        if first > second:
            first, second = second, first
        pair_indices_set.add((first, second))
        attempts += 1
    return list(pair_indices_set)


def _texts_by_code(train_df: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for code, group in train_df.groupby("code", sort=True):
        result[str(code)] = unique_preserve_order(
            text.strip() for text in group["text"].astype(str).tolist() if text.strip()
        )
    return result


def _response_code_sets(train_df: pd.DataFrame) -> list[tuple[str, set[str]]]:
    group_columns = ["row_id", "text"] if "row_id" in train_df.columns else ["text"]
    grouped = (
        train_df.groupby(group_columns, sort=False)["code"]
        .agg(lambda values: set(map(str, values)))
        .reset_index()
    )
    return [
        (str(row.text).strip(), set(row.code))
        for row in grouped.itertuples(index=False)
        if str(row.text).strip()
    ]


def _negative_candidates_by_code(train_df: pd.DataFrame, codes: list[str]) -> dict[str, list[str]]:
    response_sets = _response_code_sets(train_df)
    candidates: dict[str, list[str]] = {}
    for code in codes:
        texts = [text for text, row_codes in response_sets if code not in row_codes]
        candidates[code] = unique_preserve_order(texts)
    return candidates


def build_training_pairs(
    train_df: pd.DataFrame,
    min_class_size: int = 10,
    max_pairs_per_code: int = 5000,
    include_negatives: bool = True,
    negative_ratio: float = 1.0,
    max_negatives_per_code: int | None = None,
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
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative.")
    if max_negatives_per_code is not None and max_negatives_per_code < 1:
        raise ValueError("max_negatives_per_code must be positive when provided.")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    positive_records_by_code: dict[str, list[dict[str, Any]]] = {}
    texts_by_code = _texts_by_code(train_df)

    for code, texts in texts_by_code.items():
        if len(texts) < min_class_size:
            continue

        positive_records: list[dict[str, Any]] = []
        for first, second in _sample_pair_indices(len(texts), max_pairs_per_code, rng):
            record = {
                "code": code,
                "text_a": texts[first],
                "text_b": texts[second],
                "label": 1.0,
                "sample_type": "positive",
            }
            records.append(record)
            positive_records.append(record)
        positive_records_by_code[code] = positive_records

    if include_negatives and negative_ratio > 0 and positive_records_by_code:
        codes = list(positive_records_by_code.keys())
        negative_candidates = _negative_candidates_by_code(train_df, codes)
        for code in codes:
            positives = positive_records_by_code[code]
            anchors = texts_by_code.get(code, [])
            candidates = negative_candidates.get(code, [])
            if not positives or not anchors or not candidates:
                continue

            target_count = math.ceil(len(positives) * negative_ratio)
            if max_negatives_per_code is not None:
                target_count = min(target_count, max_negatives_per_code)

            seen: set[tuple[str, str]] = set()
            attempts = 0
            max_attempts = max(target_count * 50, 100)
            while len(seen) < target_count and attempts < max_attempts:
                anchor = rng.choice(anchors)
                negative = rng.choice(candidates)
                attempts += 1
                if anchor == negative:
                    continue
                key = (anchor, negative)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "code": code,
                        "text_a": anchor,
                        "text_b": negative,
                        "label": 0.0,
                        "sample_type": "negative",
                    }
                )

    return pd.DataFrame(records, columns=["code", "text_a", "text_b", "label", "sample_type"])


def build_training_triplets(
    train_df: pd.DataFrame,
    min_class_size: int = 10,
    max_triplets_per_code: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"row_id", "text", "code"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Training dataframe is missing columns: {sorted(missing)}")
    if min_class_size < 2:
        raise ValueError("min_class_size must be at least 2 for triplets.")
    if max_triplets_per_code < 1:
        raise ValueError("max_triplets_per_code must be positive.")

    rng = random.Random(seed)
    texts_by_code = _texts_by_code(train_df)
    negative_candidates = _negative_candidates_by_code(train_df, list(texts_by_code.keys()))
    records: list[dict[str, Any]] = []

    for code, texts in texts_by_code.items():
        if len(texts) < min_class_size:
            continue
        candidates = negative_candidates.get(code, [])
        if not candidates:
            continue

        pair_indices = _sample_pair_indices(len(texts), max_triplets_per_code, rng)
        for first, second in pair_indices[:max_triplets_per_code]:
            records.append(
                {
                    "code": code,
                    "anchor": texts[first],
                    "positive": texts[second],
                    "negative": rng.choice(candidates),
                }
            )

    return pd.DataFrame(records, columns=["code", "anchor", "positive", "negative"])


def train_model(
    train_df: pd.DataFrame,
    out_dir: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    training_mode: TrainingMode = "contrastive",
    min_class_size: int = 10,
    max_pairs_per_code: int = 5000,
    negative_ratio: float = 1.0,
    max_negatives_per_code: int | None = None,
    max_triplets_per_code: int | None = None,
    triplet_margin: float = 0.5,
    epochs: int = 1,
    batch_size: int = 16,
    seed: int = 42,
    device: str | None = None,
    prompt_name: str | None = None,
    input_prefix: str | None = None,
    show_progress_bar: bool = True,
) -> Path:
    if training_mode not in ("mnrl", "contrastive", "triplet"):
        raise ValueError("training_mode must be one of: mnrl, contrastive, triplet.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if triplet_margin <= 0:
        raise ValueError("triplet_margin must be positive.")

    output_dir = ensure_dir(out_dir)
    model_dir = output_dir / "model"
    set_global_seed(seed)

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.losses import TripletDistanceMetric
    from torch.utils.data import DataLoader

    model_kwargs: dict[str, Any] = {}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(base_model, **model_kwargs)
    resolved_input_prefix = configure_model_input(
        model=model,
        prompt_name=prompt_name,
        input_prefix=input_prefix,
    )

    pairs_df = pd.DataFrame()
    triplets_df = pd.DataFrame()
    n_positive_pairs = 0
    n_negative_pairs = 0
    n_triplets = 0

    if training_mode == "triplet":
        triplets_df = build_training_triplets(
            train_df=train_df,
            min_class_size=min_class_size,
            max_triplets_per_code=max_triplets_per_code or max_pairs_per_code,
            seed=seed,
        )
        triplets_df.to_csv(output_dir / "training_triplets.csv", index=False, encoding="utf-8-sig")
        if triplets_df.empty:
            counts = train_df.groupby("code")["text"].nunique().sort_values(ascending=False)
            raise ValueError(
                "No triplets were generated. Lower min_class_size or check label distribution. "
                f"Top class sizes: {counts.head(20).to_dict()}"
            )
        train_examples = [
            InputExample(
                texts=[
                    prepare_model_text(row.anchor, resolved_input_prefix),
                    prepare_model_text(row.positive, resolved_input_prefix),
                    prepare_model_text(row.negative, resolved_input_prefix),
                ]
            )
            for row in triplets_df.itertuples(index=False)
        ]
        train_loss = losses.TripletLoss(
            model,
            distance_metric=TripletDistanceMetric.COSINE,
            triplet_margin=triplet_margin,
        )
        n_triplets = int(len(triplets_df))
    else:
        pairs_df = build_training_pairs(
            train_df=train_df,
            min_class_size=min_class_size,
            max_pairs_per_code=max_pairs_per_code,
            include_negatives=training_mode == "contrastive",
            negative_ratio=negative_ratio,
            max_negatives_per_code=max_negatives_per_code,
            seed=seed,
        )
        pairs_df.to_csv(output_dir / "training_pairs.csv", index=False, encoding="utf-8-sig")
        if pairs_df.empty:
            counts = train_df.groupby("code")["text"].nunique().sort_values(ascending=False)
            raise ValueError(
                "No training pairs were generated. Lower min_class_size or check label distribution. "
                f"Top class sizes: {counts.head(20).to_dict()}"
            )

        n_positive_pairs = int((pairs_df["sample_type"] == "positive").sum())
        n_negative_pairs = int((pairs_df["sample_type"] == "negative").sum())
        if training_mode == "mnrl":
            positive_pairs = pairs_df[pairs_df["sample_type"] == "positive"]
            train_examples = [
                InputExample(
                    texts=[
                        prepare_model_text(row.text_a, resolved_input_prefix),
                        prepare_model_text(row.text_b, resolved_input_prefix),
                    ]
                )
                for row in positive_pairs.itertuples(index=False)
            ]
            train_loss = losses.MultipleNegativesRankingLoss(model)
        else:
            train_examples = [
                InputExample(
                    texts=[
                        prepare_model_text(row.text_a, resolved_input_prefix),
                        prepare_model_text(row.text_b, resolved_input_prefix),
                    ],
                    label=float(row.label),
                )
                for row in pairs_df.itertuples(index=False)
            ]
            train_loss = losses.CosineSimilarityLoss(model)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
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
        "training_mode": training_mode,
        "min_class_size": min_class_size,
        "max_pairs_per_code": max_pairs_per_code,
        "negative_ratio": negative_ratio,
        "max_negatives_per_code": max_negatives_per_code,
        "max_triplets_per_code": max_triplets_per_code or max_pairs_per_code,
        "triplet_margin": triplet_margin,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "prompt_name": prompt_name,
        "input_prefix": resolved_input_prefix,
        "n_training_rows": int(len(train_df)),
        "n_training_pairs": int(len(pairs_df)),
        "n_positive_pairs": n_positive_pairs,
        "n_negative_pairs": n_negative_pairs,
        "n_training_triplets": n_triplets,
    }
    write_json(config, output_dir / "train_config.json")
    return model_dir
