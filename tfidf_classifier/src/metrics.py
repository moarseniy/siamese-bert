from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def encode_labels(code_lists: list[list[str]], classes: list[str]) -> np.ndarray:
    index_by_code = {code: index for index, code in enumerate(classes)}
    targets = np.zeros((len(code_lists), len(classes)), dtype=np.int8)
    for row_index, codes in enumerate(code_lists):
        for code in codes:
            if code in index_by_code:
                targets[row_index, index_by_code[code]] = 1
    return targets


def threshold_predictions(
    probabilities: np.ndarray,
    threshold: float,
    max_labels: int = 6,
) -> np.ndarray:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    if max_labels < 1:
        raise ValueError("max_labels must be positive.")
    predictions = np.zeros_like(probabilities, dtype=np.int8)
    for row_index, scores in enumerate(probabilities):
        accepted = np.flatnonzero(scores >= threshold)
        if len(accepted) > max_labels:
            accepted = accepted[np.argsort(-scores[accepted])[:max_labels]]
        predictions[row_index, accepted] = 1
    return predictions


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
    max_labels: int = 6,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(y_true) == 0:
        return (
            {"n_rows": 0, "threshold": threshold},
            pd.DataFrame(
                columns=["code", "support", "precision", "recall", "f1"]
            ),
        )

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        label_ranking_average_precision_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    predictions = threshold_predictions(probabilities, threshold, max_labels)
    single_mask = y_true.sum(axis=1) == 1
    top1_accuracy: float | None = None
    if single_mask.any():
        top1_accuracy = float(
            (
                y_true[single_mask].argmax(axis=1)
                == probabilities[single_mask].argmax(axis=1)
            ).mean()
        )
    metrics = {
        "n_rows": int(len(y_true)),
        "threshold": float(threshold),
        "micro_precision": float(
            precision_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, predictions)),
        "hamming_loss": float(hamming_loss(y_true, predictions)),
        "lrap": float(label_ranking_average_precision_score(y_true, probabilities)),
        "single_label_top1_accuracy": top1_accuracy,
        "unknown_prediction_rate": float((predictions.sum(axis=1) == 0).mean()),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "code": classes,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ).sort_values(["support", "code"], ascending=[False, True])
    return metrics, per_class


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    metric: str = "micro_f1",
    max_labels: int = 6,
) -> tuple[float, pd.DataFrame]:
    if metric not in {"micro_f1", "macro_f1"}:
        raise ValueError("metric must be micro_f1 or macro_f1.")
    if len(y_true) == 0:
        return 0.5, pd.DataFrame()

    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        metrics, _ = calculate_metrics(
            y_true,
            probabilities,
            classes,
            float(threshold),
            max_labels,
        )
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    scores = pd.DataFrame(rows)
    best = scores.sort_values([metric, "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), scores
