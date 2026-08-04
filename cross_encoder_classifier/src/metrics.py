from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def decode_response_predictions(
    probabilities: np.ndarray,
    threshold: float,
    max_labels: int = 6,
) -> np.ndarray:
    if probabilities.ndim != 3 or probabilities.shape[-1] != 4:
        raise ValueError("probabilities must have shape [responses, codes, 4].")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    if max_labels < 1:
        raise ValueError("max_labels must be positive.")
    presence = 1.0 - probabilities[:, :, 0]
    sentiment_classes = probabilities[:, :, 1:].argmax(axis=2) + 1
    result = np.zeros(presence.shape, dtype=np.int8)
    for row_index, scores in enumerate(presence):
        accepted = np.flatnonzero(scores >= threshold)
        if len(accepted) > max_labels:
            accepted = accepted[np.argsort(-scores[accepted])[:max_labels]]
        result[row_index, accepted] = sentiment_classes[row_index, accepted]
    return result


def calculate_pair_metrics(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(y_true) == 0:
        return {"n_pairs": 0}, pd.DataFrame()
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    predicted = probabilities.argmax(axis=1)
    true_present = y_true > 0
    predicted_present = predicted > 0
    metrics = {
        "n_pairs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
        "presence_precision": float(
            precision_score(true_present, predicted_present, zero_division=0)
        ),
        "presence_recall": float(
            recall_score(true_present, predicted_present, zero_division=0)
        ),
        "presence_f1": float(
            f1_score(true_present, predicted_present, zero_division=0)
        ),
    }
    positive_mask = true_present
    metrics["sentiment_accuracy_on_gold_pairs"] = (
        float((predicted[positive_mask] == y_true[positive_mask]).mean())
        if positive_mask.any()
        else None
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=[0, 1, 2, 3], zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "model_class": [0, 1, 2, 3],
            "class_name": ["absent", "neutral", "positive", "negative"],
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return metrics, per_class


def calculate_response_metrics(
    y_true_classes: np.ndarray,
    probabilities: np.ndarray,
    codes: list[str],
    threshold: float,
    max_labels: int = 6,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(y_true_classes) == 0:
        return {"n_rows": 0, "threshold": threshold}, pd.DataFrame()
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    predicted_classes = decode_response_predictions(
        probabilities, threshold, max_labels
    )
    y_true = y_true_classes > 0
    predicted = predicted_classes > 0
    gold_mask = y_true
    detected_gold_mask = gold_mask & predicted
    true_joint = np.zeros((*y_true_classes.shape, 3), dtype=np.int8)
    predicted_joint = np.zeros((*predicted_classes.shape, 3), dtype=np.int8)
    for sentiment_class in (1, 2, 3):
        true_joint[:, :, sentiment_class - 1] = y_true_classes == sentiment_class
        predicted_joint[:, :, sentiment_class - 1] = (
            predicted_classes == sentiment_class
        )
    true_joint_flat = true_joint.reshape(len(y_true), -1)
    predicted_joint_flat = predicted_joint.reshape(len(y_true), -1)
    metrics = {
        "n_rows": int(len(y_true)),
        "threshold": float(threshold),
        "micro_precision": float(
            precision_score(y_true, predicted, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, predicted, average="micro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, predicted, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "subset_accuracy": float(accuracy_score(y_true, predicted)),
        "hamming_loss": float(hamming_loss(y_true, predicted)),
        "unknown_prediction_rate": float((predicted.sum(axis=1) == 0).mean()),
        "joint_micro_precision": float(
            precision_score(
                true_joint_flat,
                predicted_joint_flat,
                average="micro",
                zero_division=0,
            )
        ),
        "joint_micro_recall": float(
            recall_score(
                true_joint_flat,
                predicted_joint_flat,
                average="micro",
                zero_division=0,
            )
        ),
        "joint_micro_f1": float(
            f1_score(
                true_joint_flat,
                predicted_joint_flat,
                average="micro",
                zero_division=0,
            )
        ),
        "joint_macro_f1": float(
            f1_score(
                true_joint_flat,
                predicted_joint_flat,
                average="macro",
                zero_division=0,
            )
        ),
        "joint_pair_accuracy": float((predicted_classes == y_true_classes).mean()),
        "gold_code_sentiment_accuracy": (
            float((predicted_classes[gold_mask] == y_true_classes[gold_mask]).mean())
            if gold_mask.any()
            else None
        ),
        "detected_gold_sentiment_accuracy": (
            float(
                (
                    predicted_classes[detected_gold_mask]
                    == y_true_classes[detected_gold_mask]
                ).mean()
            )
            if detected_gold_mask.any()
            else None
        ),
        "gold_codes": int(gold_mask.sum()),
        "detected_gold_codes": int(detected_gold_mask.sum()),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, average=None, zero_division=0
    )
    sentiment_accuracy = []
    for code_index in range(len(codes)):
        mask = y_true[:, code_index]
        sentiment_accuracy.append(
            float(
                (
                    predicted_classes[mask, code_index]
                    == y_true_classes[mask, code_index]
                ).mean()
            )
            if mask.any()
            else None
        )
    per_code = pd.DataFrame(
        {
            "code": codes,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gold_code_sentiment_accuracy": sentiment_accuracy,
        }
    ).sort_values(["support", "code"], ascending=[False, True])
    return metrics, per_code


def select_presence_threshold(
    y_true_classes: np.ndarray,
    probabilities: np.ndarray,
    codes: list[str],
    metric: str = "micro_f1",
    max_labels: int = 6,
) -> tuple[float, pd.DataFrame]:
    allowed_metrics = {
        "micro_f1",
        "macro_f1",
        "joint_micro_f1",
        "joint_macro_f1",
    }
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of {sorted(allowed_metrics)}.")
    if len(y_true_classes) == 0:
        return 0.5, pd.DataFrame()
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        metrics, _ = calculate_response_metrics(
            y_true_classes,
            probabilities,
            codes,
            threshold=float(threshold),
            max_labels=max_labels,
        )
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "joint_micro_f1": metrics["joint_micro_f1"],
                "joint_macro_f1": metrics["joint_macro_f1"],
            }
        )
    scores = pd.DataFrame(rows)
    best = scores.sort_values([metric, "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), scores
