from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data_io import (
    CODE_DESCRIPTION_FORMAT_TEXT_ONLY,
    MODEL_CLASS_NAMES,
    build_pairs,
    leaf_codebook,
    load_labeled_data,
    save_splits,
    split_train_val_test,
)
from .metrics import (
    calculate_pair_metrics,
    calculate_response_metrics,
    decode_response_predictions,
    select_presence_threshold,
)

DEFAULT_BASE_MODEL = "DeepPavlov/rubert-base-cased"


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str | None, torch: Any) -> Any:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _PairDataset:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.answers = frame["text"].astype(str).tolist()
        self.descriptions = frame["code_description"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, str, int]:
        return self.answers[index], self.descriptions[index], self.labels[index]


def _make_loader(
    frame: pd.DataFrame,
    tokenizer: Any,
    torch: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    device: Any,
) -> Any:
    dataset = _PairDataset(frame)

    def collate(batch: list[tuple[str, str, int]]) -> dict[str, Any]:
        answers, descriptions, labels = zip(*batch)
        encoded = tokenizer(
            list(answers),
            text_pair=list(descriptions),
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and len(dataset) > 0,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )


def _class_weights(
    labels: np.ndarray,
    max_class_weight: float,
    torch: Any,
    device: Any,
) -> Any:
    counts = np.bincount(labels, minlength=4).astype(np.float32)
    weights = np.divide(
        len(labels),
        4.0 * counts,
        out=np.ones(4, dtype=np.float32),
        where=counts > 0,
    )
    weights = np.clip(weights, 0.25, max_class_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _autocast_settings(
    torch: Any, device: Any, mixed_precision: bool
) -> tuple[Any, bool]:
    if not mixed_precision or device.type != "cuda":
        return torch.float32, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True
    return torch.float16, True


def _evaluate(
    model: Any,
    loader: Any,
    criterion: Any,
    torch: Any,
    device: Any,
    amp_dtype: Any,
    use_autocast: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    true_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_autocast,
            ):
                logits = model(**inputs).logits
                loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            true_batches.append(labels.detach().cpu().numpy())
            probability_batches.append(
                torch.softmax(logits, dim=1).float().cpu().numpy()
            )
    if not true_batches:
        return 0.0, np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32)
    return (
        float(np.mean(losses)),
        np.concatenate(true_batches),
        np.concatenate(probability_batches),
    )


def _response_arrays(
    pair_frame: pd.DataFrame,
    probabilities: np.ndarray,
    n_responses: int,
    codes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.zeros((n_responses, len(codes)), dtype=np.int8)
    response_probabilities = np.zeros((n_responses, len(codes), 4), dtype=np.float32)
    response_probabilities[:, :, 0] = 1.0
    code_index = {code: index for index, code in enumerate(codes)}
    for pair_index, pair in pair_frame.reset_index(drop=True).iterrows():
        row_index = int(pair["response_position"])
        column_index = code_index[str(pair["code"])]
        y_true[row_index, column_index] = int(pair["label"])
        response_probabilities[row_index, column_index] = probabilities[pair_index]
    return y_true, response_probabilities


def _linear_schedule(
    optimizer: Any, total_steps: int, warmup_ratio: float, torch: Any
) -> Any:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return float(remaining) / float(decay_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _response_prediction_frame(
    responses: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    codebook: pd.DataFrame,
    threshold: float,
    max_labels: int,
) -> pd.DataFrame:
    leaves = leaf_codebook(codebook)
    codes = leaves["code"].astype(str).tolist()
    names = leaves.set_index("code")["name"].astype(str).to_dict()
    predicted = decode_response_predictions(probabilities, threshold, max_labels)
    rows: list[dict[str, Any]] = []
    for row_index, response in responses.reset_index(drop=True).iterrows():
        true_items = [
            f"{codes[index]}:{int(y_true[row_index, index]) - 1}"
            for index in np.flatnonzero(y_true[row_index])
        ]
        predicted_items = [
            f"{codes[index]}:{int(predicted[row_index, index]) - 1}"
            for index in np.flatnonzero(predicted[row_index])
        ]
        presence = 1.0 - probabilities[row_index, :, 0]
        ranked = np.argsort(-presence)
        rows.append(
            {
                "row_id": int(response["row_id"]),
                "answer": response["answer"],
                "context": response["context"],
                "true_code_sentiments": ", ".join(true_items) or "UNKNOWN",
                "predicted_code_sentiments": ", ".join(predicted_items) or "UNKNOWN",
                "top_candidates": "; ".join(
                    f"{codes[index]}:{presence[index]:.4f}:"
                    f"{int(probabilities[row_index, index, 1:].argmax())}"
                    for index in ranked[: min(5, len(ranked))]
                ),
                "correct": bool(
                    np.array_equal(y_true[row_index], predicted[row_index])
                ),
                "predicted_names": "; ".join(
                    names[codes[index]]
                    for index in np.flatnonzero(predicted[row_index])
                ),
            }
        )
    return pd.DataFrame(rows)


def _save_pair_distribution(splits: dict[str, pd.DataFrame], output_path: Path) -> None:
    rows = []
    for split_name, pairs in splits.items():
        counts = pairs["label"].value_counts().to_dict()
        for model_class in range(4):
            rows.append(
                {
                    "split": split_name,
                    "model_class": model_class,
                    "class_name": MODEL_CLASS_NAMES[model_class],
                    "pairs": int(counts.get(model_class, 0)),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def train_model(
    train_path: str | Path,
    codebook_path: str | Path,
    output_dir: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    text_col: str = "Ответ",
    codes_col: str = "Коды_новые",
    sentiments_col: str | None = "Тональности",
    context_col: str | None = None,
    after_semicolon_prefix: str | None = None,
    csv_sep: str | None = None,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 16,
    eval_batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 256,
    max_labels: int = 6,
    negative_ratio: float | None = 3.0,
    max_class_weight: float = 10.0,
    use_class_weights: bool = True,
    label_smoothing: float = 0.0,
    threshold_metric: str = "joint_micro_f1",
    early_stopping_patience: int = 2,
    num_workers: int = 0,
    device_name: str | None = None,
    mixed_precision: bool = True,
    trust_remote_code: bool = False,
) -> Path:
    if epochs < 1 or batch_size < 1 or eval_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive.")
    if gradient_accumulation_steps < 1 or max_length < 1 or max_labels < 1:
        raise ValueError(
            "accumulation steps, max_length and max_labels must be positive."
        )
    if learning_rate <= 0 or max_class_weight < 1:
        raise ValueError(
            "learning_rate must be positive and max_class_weight at least 1."
        )
    if not 0 <= warmup_ratio < 1 or not 0 <= label_smoothing < 1:
        raise ValueError("warmup_ratio and label_smoothing must be in [0, 1).")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install training dependencies first: pip install -r requirements.txt"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed, torch)
    device = _resolve_device(device_name, torch)
    data, codebook = load_labeled_data(
        train_path,
        codebook_path,
        text_col=text_col,
        codes_col=codes_col,
        sentiments_col=sentiments_col,
        context_col=context_col,
        after_semicolon_prefix=after_semicolon_prefix,
        csv_sep=csv_sep,
    )
    data_report = dict(data.attrs.get("load_report", {}))
    skipped_conflicts = int(
        data_report.get("skipped_conflicting_sentiment_rows", 0)
    )
    skipped_source_rows = list(
        data_report.get("skipped_conflicting_sentiment_source_rows", [])
    )
    print(f"Skipped rows with conflicting code sentiments: {skipped_conflicts}")
    if skipped_source_rows:
        preview = ", ".join(map(str, skipped_source_rows[:20]))
        suffix = " ..." if len(skipped_source_rows) > 20 else ""
        print(f"Conflicting source rows: {preview}{suffix}")
    _write_json(data_report, output_dir / "data_report.json")
    leaves = leaf_codebook(codebook)
    codes = leaves["code"].astype(str).tolist()
    response_splits = split_train_val_test(data, val_size, test_size, seed)
    save_splits(response_splits, output_dir / "splits")
    codebook.to_csv(output_dir / "codebook.csv", index=False, encoding="utf-8-sig")

    pair_splits = {
        "train": build_pairs(
            response_splits["train"], codebook, negative_ratio=negative_ratio, seed=seed
        ),
        "val": build_pairs(
            response_splits["val"], codebook, negative_ratio=None, seed=seed
        ),
        "test": build_pairs(
            response_splits["test"], codebook, negative_ratio=None, seed=seed
        ),
    }
    _save_pair_distribution(pair_splits, output_dir / "pair_distribution.csv")
    evaluation_split = "val" if len(pair_splits["val"]) else "train"
    if evaluation_split == "train":
        pair_splits["train_eval"] = build_pairs(
            response_splits["train"], codebook, negative_ratio=None, seed=seed
        )
        evaluation_pairs = pair_splits["train_eval"]
    else:
        evaluation_pairs = pair_splits[evaluation_split]

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=4,
        id2label=MODEL_CLASS_NAMES,
        label2id={name: index for index, name in MODEL_CLASS_NAMES.items()},
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
        trust_remote_code=trust_remote_code,
    ).to(device)
    loaders = {
        name: _make_loader(
            frame,
            tokenizer,
            torch,
            batch_size if name == "train" else eval_batch_size,
            max_length,
            name == "train",
            num_workers,
            device,
        )
        for name, frame in pair_splits.items()
    }
    if evaluation_split == "train":
        evaluation_loader = loaders["train_eval"]
        evaluation_responses = response_splits["train"]
    else:
        evaluation_loader = loaders[evaluation_split]
        evaluation_responses = response_splits[evaluation_split]

    weights = (
        _class_weights(
            pair_splits["train"]["label"].to_numpy(),
            max_class_weight,
            torch,
            device,
        )
        if use_class_weights
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, label_smoothing=label_smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    updates_per_epoch = math.ceil(
        max(len(loaders["train"]), 1) / gradient_accumulation_steps
    )
    scheduler = _linear_schedule(
        optimizer, updates_per_epoch * epochs, warmup_ratio, torch
    )
    amp_dtype, use_autocast = _autocast_settings(torch, device, mixed_precision)
    use_scaler = use_autocast and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(
        f"Responses: train={len(response_splits['train'])}, "
        f"val={len(response_splits['val'])}, test={len(response_splits['test'])}; "
        f"leaf_codes={len(codes)}"
    )
    print(
        f"Pairs: train={len(pair_splits['train'])}, val={len(pair_splits['val'])}, "
        f"test={len(pair_splits['test'])}; negative_ratio={negative_ratio}"
    )
    print(
        f"Model: {base_model}; device={device}; max_length={max_length}; "
        f"effective_batch={batch_size * gradient_accumulation_steps}"
    )

    model_dir = output_dir / "model"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_threshold = 0.5
    epochs_without_improvement = 0
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_index, batch in enumerate(progress, start=1):
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_autocast
            ):
                logits = model(**inputs).logits
                loss = criterion(logits, labels)
                scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())
            should_update = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == len(loaders["train"])
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(loss=f"{running_loss / batch_index:.4f}")

        eval_loss, _, eval_probabilities = _evaluate(
            model,
            evaluation_loader,
            criterion,
            torch,
            device,
            amp_dtype,
            use_autocast,
        )
        eval_true, eval_response_probabilities = _response_arrays(
            evaluation_pairs,
            eval_probabilities,
            len(evaluation_responses),
            codes,
        )
        threshold, threshold_scores = select_presence_threshold(
            eval_true,
            eval_response_probabilities,
            codes,
            metric=threshold_metric,
            max_labels=max_labels,
        )
        eval_metrics, _ = calculate_response_metrics(
            eval_true,
            eval_response_probabilities,
            codes,
            threshold,
            max_labels,
        )
        score = float(eval_metrics[threshold_metric])
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(loaders["train"]), 1),
            "validation_split": evaluation_split,
            "validation_loss": eval_loss,
            "threshold": threshold,
            **eval_metrics,
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch}: train_loss={epoch_result['train_loss']:.4f}; "
            f"{evaluation_split}_loss={eval_loss:.4f}; "
            f"{threshold_metric}={score:.4f}; threshold={threshold:.2f}"
        )
        if epoch == 1 or score > best_score:
            best_score = score
            best_threshold = threshold
            epochs_without_improvement = 0
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
            threshold_scores.to_csv(
                output_dir / "threshold_search.csv", index=False, encoding="utf-8-sig"
            )
        else:
            epochs_without_improvement += 1
            if (
                early_stopping_patience >= 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(f"Early stopping after epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    del optimizer, scheduler, scaler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    ).to(device)

    final_metrics: dict[str, Any] = {}
    for split_name in ("val", "test"):
        pairs = pair_splits[split_name]
        if pairs.empty:
            final_metrics[split_name] = {"n_rows": 0, "n_pairs": 0}
            _write_json(
                final_metrics[split_name], output_dir / f"{split_name}_metrics.json"
            )
            continue
        loss, pair_true, pair_probabilities = _evaluate(
            model,
            loaders[split_name],
            criterion,
            torch,
            device,
            amp_dtype,
            use_autocast,
        )
        pair_metrics, pair_per_class = calculate_pair_metrics(
            pair_true, pair_probabilities
        )
        y_true, response_probabilities = _response_arrays(
            pairs, pair_probabilities, len(response_splits[split_name]), codes
        )
        response_metrics, per_code = calculate_response_metrics(
            y_true, response_probabilities, codes, best_threshold, max_labels
        )
        metrics = {"loss": loss, **response_metrics, "pair": pair_metrics}
        final_metrics[split_name] = metrics
        _write_json(metrics, output_dir / f"{split_name}_metrics.json")
        pair_per_class.to_csv(
            output_dir / f"{split_name}_pair_per_class.csv",
            index=False,
            encoding="utf-8-sig",
        )
        per_code.to_csv(
            output_dir / f"{split_name}_per_class.csv",
            index=False,
            encoding="utf-8-sig",
        )
        predictions = _response_prediction_frame(
            response_splits[split_name],
            y_true,
            response_probabilities,
            codebook,
            best_threshold,
            max_labels,
        )
        predictions.to_csv(
            output_dir / f"{split_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        predictions[~predictions["correct"]].to_csv(
            output_dir / f"{split_name}_errors.csv", index=False, encoding="utf-8-sig"
        )

    config = {
        "pipeline": "cross_encoder_code_sentiment_classifier",
        "base_model": base_model,
        "model_subdir": "model",
        "code_description_format": CODE_DESCRIPTION_FORMAT_TEXT_ONLY,
        "num_model_classes": 4,
        "model_classes": MODEL_CLASS_NAMES,
        "sentiment_mapping": {"0": "neutral", "1": "positive", "2": "negative"},
        "threshold": best_threshold,
        "threshold_metric": threshold_metric,
        "max_labels": max_labels,
        "max_length": max_length,
        "text_col": text_col,
        "codes_col": codes_col,
        "sentiments_col": sentiments_col,
        "context_col": context_col,
        "after_semicolon_prefix": after_semicolon_prefix,
        "seed": seed,
        "val_size": val_size,
        "test_size": test_size,
        "negative_ratio": negative_ratio,
        "class_weights": use_class_weights,
        "trust_remote_code": trust_remote_code,
        "data_report": data_report,
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "label_smoothing": label_smoothing,
            "mixed_precision": mixed_precision,
            "global_steps": global_step,
        },
        "split_rows": {name: len(frame) for name, frame in response_splits.items()},
        "split_pairs": {
            name: len(pair_splits[name]) for name in ("train", "val", "test")
        },
        "best_validation_score": best_score,
    }
    _write_json(config, output_dir / "classifier_config.json")
    _write_json(final_metrics, output_dir / "metrics.json")
    print(f"Done. Model and reports saved to {output_dir.resolve()}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pairwise transformer for code presence and sentiment."
    )
    parser.add_argument(
        "--train", required=True, type=Path, help="CSV/XLSX with labels."
    )
    parser.add_argument("--codebook", required=True, type=Path, help="Codebook CSV.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--text-col", default="Ответ")
    parser.add_argument("--codes-col", default="Коды_новые")
    parser.add_argument("--sentiments-col", default="Тональности")
    parser.add_argument("--context-col", default=None)
    parser.add_argument(
        "--after-semicolon-prefix",
        default=None,
        help=(
            "Insert this text immediately after the first ';' in every answer. "
            "Answers without ';' are unchanged."
        ),
    )
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-labels", type=int, default=6)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=3.0,
        help="Sampled absent pairs per positive pair (default: 3).",
    )
    parser.add_argument(
        "--all-negatives",
        action="store_true",
        help="Use every absent code pair in train instead of sampling.",
    )
    parser.add_argument("--max-class-weight", type=float, default=10.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--threshold-metric",
        choices=["micro_f1", "macro_f1", "joint_micro_f1", "joint_macro_f1"],
        default="joint_micro_f1",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default=None, help="For example: cuda, cuda:1, cpu."
    )
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_model(
        train_path=args.train,
        codebook_path=args.codebook,
        output_dir=args.out_dir,
        base_model=args.base_model,
        text_col=args.text_col,
        codes_col=args.codes_col,
        sentiments_col=args.sentiments_col,
        context_col=args.context_col,
        after_semicolon_prefix=args.after_semicolon_prefix,
        csv_sep=args.csv_sep,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        max_labels=args.max_labels,
        negative_ratio=None if args.all_negatives else args.negative_ratio,
        max_class_weight=args.max_class_weight,
        use_class_weights=not args.no_class_weights,
        label_smoothing=args.label_smoothing,
        threshold_metric=args.threshold_metric,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        device_name=args.device,
        mixed_precision=not args.no_mixed_precision,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
