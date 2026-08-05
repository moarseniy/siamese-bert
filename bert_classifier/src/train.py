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

from .data_io import load_labeled_data, save_splits, split_train_val_test
from .metrics import (
    calculate_metrics,
    encode_labels,
    select_threshold,
    threshold_predictions,
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


class _TextDataset:
    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray]:
        return self.texts[index], self.labels[index]


def _make_loader(
    frame: pd.DataFrame,
    labels: np.ndarray,
    tokenizer: Any,
    torch: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    device: Any,
) -> Any:
    dataset = _TextDataset(frame["text"].astype(str).tolist(), labels)

    def collate(batch: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
        texts, targets = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(np.stack(targets), dtype=torch.float32)
        return encoded

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )


def _positive_weights(
    labels: np.ndarray,
    max_positive_weight: float,
    torch: Any,
    device: Any,
) -> Any:
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    weights = np.divide(
        negatives,
        positives,
        out=np.ones_like(positives, dtype=np.float32),
        where=positives > 0,
    )
    weights = np.clip(weights, 1.0, max_positive_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _save_label_distribution(
    splits: dict[str, pd.DataFrame],
    classes: list[str],
    codebook: pd.DataFrame,
    output_path: Path,
) -> None:
    names = codebook.set_index("code")["name"].astype(str).to_dict()
    rows = []
    for code in classes:
        row: dict[str, Any] = {"code": code, "name": names.get(code, code)}
        for split_name, frame in splits.items():
            row[split_name] = int(
                frame["codes"].apply(lambda values: code in values).sum()
            )
        row["total"] = sum(row[name] for name in splits)
        rows.append(row)
    pd.DataFrame(rows).sort_values(
        ["train", "code"],
        ascending=[True, True],
    ).to_csv(output_path, index=False, encoding="utf-8-sig")


def _autocast_settings(torch: Any, device: Any, mixed_precision: bool) -> tuple[Any, bool]:
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
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
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
            probability_batches.append(torch.sigmoid(logits).float().cpu().numpy())

    if not true_batches:
        n_labels = int(model.config.num_labels)
        empty = np.empty((0, n_labels), dtype=np.float32)
        return 0.0, empty, empty
    return (
        float(np.mean(losses)),
        np.concatenate(true_batches),
        np.concatenate(probability_batches),
    )


def _prediction_frame(
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
    max_labels: int,
) -> pd.DataFrame:
    result = frame[["row_id", "answer", "context"]].reset_index(drop=True).copy()
    predicted = threshold_predictions(probabilities, threshold, max_labels)
    result["true_codes"] = [
        ", ".join(classes[index] for index in np.flatnonzero(row))
        for row in labels
    ]
    result["predicted_codes"] = [
        ", ".join(classes[index] for index in np.flatnonzero(row)) or "UNKNOWN"
        for row in predicted
    ]
    result["top_candidates"] = [
        "; ".join(
            f"{classes[index]}:{scores[index]:.4f}"
            for index in np.argsort(-scores)[: min(5, len(classes))]
        )
        for scores in probabilities
    ]
    result["correct"] = result["true_codes"] == result["predicted_codes"]
    return result


def _linear_schedule(
    optimizer: Any,
    total_steps: int,
    warmup_ratio: float,
    torch: Any,
) -> Any:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return float(remaining) / float(decay_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_model(
    train_path: str | Path,
    codebook_path: str | Path,
    output_dir: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    text_col: str = "Ответ",
    codes_col: str = "Коды_новые",
    context_col: str | None = None,
    csv_sep: str | None = None,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 128,
    max_labels: int = 6,
    max_positive_weight: float = 20.0,
    use_class_weights: bool = True,
    threshold_metric: str = "micro_f1",
    early_stopping_patience: int = 2,
    num_workers: int = 0,
    device_name: str | None = None,
    mixed_precision: bool = True,
    trust_remote_code: bool = False,
) -> Path:
    if epochs < 1 or batch_size < 1 or eval_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive.")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if learning_rate <= 0 or max_length < 1 or max_labels < 1:
        raise ValueError("learning_rate, max_length and max_labels must be positive.")
    if max_positive_weight < 1:
        raise ValueError("max_positive_weight must be at least 1.")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1).")

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
        context_col=context_col,
        csv_sep=csv_sep,
    )
    classes = sorted({code for values in data["codes"] for code in values})
    splits = split_train_val_test(
        data,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    save_splits(splits, output_dir / "splits")
    codebook.to_csv(output_dir / "codebook.csv", index=False, encoding="utf-8-sig")
    _save_label_distribution(
        splits,
        classes,
        codebook,
        output_dir / "label_distribution.csv",
    )

    labels = {
        name: encode_labels(frame["codes"].tolist(), classes)
        for name, frame in splits.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(classes),
        id2label={index: code for index, code in enumerate(classes)},
        label2id={code: index for index, code in enumerate(classes)},
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        trust_remote_code=trust_remote_code,
    ).to(device)

    loaders = {
        "train": _make_loader(
            splits["train"],
            labels["train"],
            tokenizer,
            torch,
            batch_size,
            max_length,
            True,
            num_workers,
            device,
        ),
        "val": _make_loader(
            splits["val"],
            labels["val"],
            tokenizer,
            torch,
            eval_batch_size,
            max_length,
            False,
            num_workers,
            device,
        ),
        "test": _make_loader(
            splits["test"],
            labels["test"],
            tokenizer,
            torch,
            eval_batch_size,
            max_length,
            False,
            num_workers,
            device,
        ),
    }
    positive_weights = (
        _positive_weights(
            labels["train"],
            max_positive_weight=max_positive_weight,
            torch=torch,
            device=device,
        )
        if use_class_weights
        else None
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    updates_per_epoch = math.ceil(
        max(len(loaders["train"]), 1) / gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * epochs
    scheduler = _linear_schedule(optimizer, total_steps, warmup_ratio, torch)
    amp_dtype, use_autocast = _autocast_settings(torch, device, mixed_precision)
    use_scaler = use_autocast and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(
        f"Data: train={len(splits['train'])}, val={len(splits['val'])}, "
        f"test={len(splits['test'])}, labels={len(classes)}"
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
            batch_labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_autocast,
            ):
                logits = model(**inputs).logits
                loss = criterion(logits, batch_labels)
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

        val_loss, val_true, val_probabilities = _evaluate(
            model,
            loaders["val"],
            criterion,
            torch,
            device,
            amp_dtype,
            use_autocast,
        )
        threshold, threshold_scores = select_threshold(
            val_true,
            val_probabilities,
            classes,
            metric=threshold_metric,
            max_labels=max_labels,
        )
        val_metrics, _ = calculate_metrics(
            val_true,
            val_probabilities,
            classes,
            threshold=threshold,
            max_labels=max_labels,
        )
        score = float(val_metrics.get(threshold_metric, 0.0))
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(loaders["train"]), 1),
            "val_loss": val_loss,
            "threshold": threshold,
            **val_metrics,
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch}: train_loss={epoch_result['train_loss']:.4f}; "
            f"val_loss={val_loss:.4f}; {threshold_metric}={score:.4f}; "
            f"threshold={threshold:.2f}"
        )

        if epoch == 1 or score > best_score:
            best_score = score
            best_threshold = threshold
            epochs_without_improvement = 0
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
            threshold_scores.to_csv(
                output_dir / "threshold_search.csv",
                index=False,
                encoding="utf-8-sig",
            )
        else:
            epochs_without_improvement += 1
            if (
                len(splits["val"]) > 0
                and early_stopping_patience >= 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(f"Early stopping after epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    del optimizer, scheduler, scaler
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
    ).to(device)

    final_metrics: dict[str, Any] = {}
    for split_name in ("val", "test"):
        split_loss, split_true, split_probabilities = _evaluate(
            model,
            loaders[split_name],
            criterion,
            torch,
            device,
            amp_dtype,
            use_autocast,
        )
        split_metrics, per_class = calculate_metrics(
            split_true,
            split_probabilities,
            classes,
            threshold=best_threshold,
            max_labels=max_labels,
        )
        split_metrics["loss"] = split_loss
        final_metrics[split_name] = split_metrics
        _write_json(split_metrics, output_dir / f"{split_name}_metrics.json")
        per_class.to_csv(
            output_dir / f"{split_name}_per_class.csv",
            index=False,
            encoding="utf-8-sig",
        )
        prediction_frame = _prediction_frame(
            splits[split_name],
            split_true,
            split_probabilities,
            classes,
            best_threshold,
            max_labels,
        )
        prediction_frame.to_csv(
            output_dir / f"{split_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        prediction_frame[~prediction_frame["correct"]].to_csv(
            output_dir / f"{split_name}_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

    config = {
        "pipeline": "bert_multi_label_classifier",
        "base_model": base_model,
        "model_subdir": "model",
        "classes": classes,
        "threshold": best_threshold,
        "threshold_metric": threshold_metric,
        "max_labels": max_labels,
        "max_length": max_length,
        "text_col": text_col,
        "codes_col": codes_col,
        "context_col": context_col,
        "seed": seed,
        "val_size": val_size,
        "test_size": test_size,
        "class_weights": use_class_weights,
        "max_positive_weight": max_positive_weight,
        "trust_remote_code": trust_remote_code,
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "mixed_precision": mixed_precision,
            "global_steps": global_step,
        },
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "best_validation_score": best_score,
    }
    _write_json(config, output_dir / "classifier_config.json")
    _write_json(final_metrics, output_dir / "metrics.json")
    print(f"Done. Model and reports saved to {output_dir.resolve()}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune RuBERT for multi-label survey response classification."
    )
    parser.add_argument("--train", required=True, type=Path, help="CSV/XLSX with labels.")
    parser.add_argument("--codebook", required=True, type=Path, help="Codebook CSV.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--text-col", default="Ответ")
    parser.add_argument("--codes-col", default="Коды_новые")
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-labels", type=int, default=6)
    parser.add_argument("--max-positive-weight", type=float, default=20.0)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable positive weights for rare labels.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["micro_f1", "macro_f1"],
        default="micro_f1",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:1, cpu.")
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
        context_col=args.context_col,
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
        max_positive_weight=args.max_positive_weight,
        use_class_weights=not args.no_class_weights,
        threshold_metric=args.threshold_metric,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        device_name=args.device,
        mixed_precision=not args.no_mixed_precision,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
