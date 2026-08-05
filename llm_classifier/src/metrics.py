from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import parse_code_sentiments


def calculate_metrics(
    frame: pd.DataFrame,
    gold_codes_col: str,
    known_codes: set[str],
    prediction_col: str = "predicted_code_sentiments",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if gold_codes_col not in frame.columns:
        raise ValueError(f"Gold column {gold_codes_col!r} is absent.")
    if prediction_col not in frame.columns:
        raise ValueError(f"Prediction column {prediction_col!r} is absent.")

    def parse_values(
        values: pd.Series,
        column: str,
    ) -> list[list[tuple[str, int]]]:
        parsed: list[list[tuple[str, int]]] = []
        for row_position, value in enumerate(values):
            try:
                labels = parse_code_sentiments(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid code/sentiment labels in column {column!r} at "
                    f"source row {row_position + 2}: {exc}"
                ) from exc
            parsed.append(
                [
                    (code, sentiment)
                    for code, sentiment in labels
                    if code in known_codes
                ]
            )
        return parsed

    gold_labels = parse_values(frame[gold_codes_col], gold_codes_col)
    prediction_labels = parse_values(frame[prediction_col], prediction_col)
    valid_mask = np.asarray([bool(values) for values in gold_labels], dtype=bool)
    evaluated = frame.loc[valid_mask].copy()
    gold_labels = [
        values for values, keep in zip(gold_labels, valid_mask, strict=True) if keep
    ]
    prediction_labels = [
        values
        for values, keep in zip(prediction_labels, valid_mask, strict=True)
        if keep
    ]
    base_metrics = {
        "input_rows": int(len(frame)),
        "evaluated_rows": int(valid_mask.sum()),
        "rows_without_known_gold_codes": int((~valid_mask).sum()),
    }
    if not gold_labels:
        return (
            base_metrics,
            pd.DataFrame(
                columns=[
                    "code",
                    "support",
                    "precision",
                    "recall",
                    "f1",
                    "gold_code_sentiment_accuracy",
                    "detected_gold_sentiment_accuracy",
                ]
            ),
            pd.DataFrame(),
        )

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import MultiLabelBinarizer

    def unique_codes(labels: list[tuple[str, int]]) -> list[str]:
        return list(dict.fromkeys(code for code, _ in labels))

    classes = sorted(known_codes)
    gold_lists = [unique_codes(labels) for labels in gold_labels]
    prediction_lists = [unique_codes(labels) for labels in prediction_labels]
    binarizer = MultiLabelBinarizer(classes=classes)
    binarizer.fit([classes])
    y_true = binarizer.transform(gold_lists)
    y_pred = binarizer.transform(prediction_lists)

    single_mask = y_true.sum(axis=1) == 1
    single_top1_accuracy: float | None = None
    if single_mask.any():
        true_top1 = y_true[single_mask].argmax(axis=1)
        predicted_first = [
            classes.index(prediction_lists[index][0])
            if prediction_lists[index] and prediction_lists[index][0] in classes
            else -1
            for index in np.flatnonzero(single_mask)
        ]
        single_top1_accuracy = float(
            (true_top1 == np.asarray(predicted_first, dtype=int)).mean()
        )

    joint_classes = [
        f"{code}:{sentiment}" for code in classes for sentiment in (0, 1, 2)
    ]
    joint_binarizer = MultiLabelBinarizer(classes=joint_classes)
    joint_binarizer.fit([joint_classes])
    joint_true = joint_binarizer.transform(
        [
            [f"{code}:{sentiment}" for code, sentiment in labels]
            for labels in gold_labels
        ]
    )
    joint_pred = joint_binarizer.transform(
        [
            [f"{code}:{sentiment}" for code, sentiment in labels]
            for labels in prediction_labels
        ]
    )
    gold_pair_sets = [set(labels) for labels in gold_labels]
    prediction_pair_sets = [set(labels) for labels in prediction_labels]
    gold_code_sets = [{code for code, _ in labels} for labels in gold_labels]
    prediction_code_sets = [
        {code for code, _ in labels} for labels in prediction_labels
    ]
    gold_pairs = sum(len(labels) for labels in gold_pair_sets)
    detected_gold_pairs = sum(
        code in predicted_codes
        for gold, predicted_codes in zip(
            gold_pair_sets,
            prediction_code_sets,
            strict=True,
        )
        for code, _ in gold
    )
    correct_sentiments = sum(
        len(gold & predicted)
        for gold, predicted in zip(
            gold_pair_sets,
            prediction_pair_sets,
            strict=True,
        )
    )
    gold_codes = sum(len(codes) for codes in gold_code_sets)
    detected_gold_codes = sum(
        len(gold & predicted)
        for gold, predicted in zip(
            gold_code_sets,
            prediction_code_sets,
            strict=True,
        )
    )

    metrics = {
        **base_metrics,
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "single_label_top1_accuracy": single_top1_accuracy,
        "unknown_prediction_rate": float(
            np.mean([not values for values in prediction_lists])
        ),
        "joint_micro_precision": float(
            precision_score(joint_true, joint_pred, average="micro", zero_division=0)
        ),
        "joint_micro_recall": float(
            recall_score(joint_true, joint_pred, average="micro", zero_division=0)
        ),
        "joint_micro_f1": float(
            f1_score(joint_true, joint_pred, average="micro", zero_division=0)
        ),
        "joint_macro_f1": float(
            f1_score(joint_true, joint_pred, average="macro", zero_division=0)
        ),
        "joint_subset_accuracy": float(accuracy_score(joint_true, joint_pred)),
        "gold_code_sentiment_accuracy": float(correct_sentiments / gold_pairs),
        "detected_gold_sentiment_accuracy": (
            float(correct_sentiments / detected_gold_pairs)
            if detected_gold_pairs
            else None
        ),
        "gold_codes": int(gold_codes),
        "detected_gold_codes": int(detected_gold_codes),
        "gold_code_sentiment_pairs": int(gold_pairs),
        "gold_pairs_with_detected_code": int(detected_gold_pairs),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    gold_sentiment_accuracy: list[float | None] = []
    detected_sentiment_accuracy: list[float | None] = []
    for code in classes:
        gold_values: list[tuple[int, set[int]]] = []
        for gold, predicted in zip(
            gold_pair_sets,
            prediction_pair_sets,
            strict=True,
        ):
            gold_sentiments = {sentiment for label, sentiment in gold if label == code}
            predicted_sentiments = {
                sentiment for label, sentiment in predicted if label == code
            }
            gold_values.extend(
                (sentiment, predicted_sentiments) for sentiment in gold_sentiments
            )
        detected_values = [values for values in gold_values if values[1]]
        gold_sentiment_accuracy.append(
            float(
                np.mean(
                    [sentiment in predicted for sentiment, predicted in gold_values]
                )
            )
            if gold_values
            else None
        )
        detected_sentiment_accuracy.append(
            float(
                np.mean(
                    [sentiment in predicted for sentiment, predicted in detected_values]
                )
            )
            if detected_values
            else None
        )

    per_class = pd.DataFrame(
        {
            "code": classes,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gold_code_sentiment_accuracy": gold_sentiment_accuracy,
            "detected_gold_sentiment_accuracy": detected_sentiment_accuracy,
        }
    ).sort_values(["support", "code"], ascending=[False, True])

    mismatches = [
        gold != predicted
        for gold, predicted in zip(
            gold_pair_sets,
            prediction_pair_sets,
            strict=True,
        )
    ]
    errors = evaluated.loc[mismatches].copy()
    return metrics, per_class, errors


def metrics_paths(output_path: str | Path) -> tuple[Path, Path, Path]:
    output = Path(output_path)
    prefix = output.with_suffix("")
    return (
        prefix.parent / f"{prefix.name}_stats.json",
        prefix.parent / f"{prefix.name}_per_class.csv",
        prefix.parent / f"{prefix.name}_errors.csv",
    )
