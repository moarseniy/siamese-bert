from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import UNKNOWN_CODE, split_codes


def calculate_metrics(
    frame: pd.DataFrame,
    gold_col: str,
    prediction_col: str = "predicted_codes",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if gold_col not in frame.columns:
        raise ValueError(f"Gold column {gold_col!r} is absent.")

    gold_lists = [split_codes(value) for value in frame[gold_col]]
    prediction_lists = [split_codes(value) for value in frame[prediction_col]]
    valid_mask = np.asarray([bool(values) for values in gold_lists], dtype=bool)
    evaluated = frame.loc[valid_mask].copy()
    gold_lists = [values for values, keep in zip(gold_lists, valid_mask, strict=True) if keep]
    prediction_lists = [
        values for values, keep in zip(prediction_lists, valid_mask, strict=True) if keep
    ]
    if not gold_lists:
        return {"n_evaluated": 0}, pd.DataFrame(), pd.DataFrame()

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import MultiLabelBinarizer

    classes = sorted(
        {
            code
            for values in [*gold_lists, *prediction_lists]
            for code in values
            if code
        }
    )
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

    metrics = {
        "n_evaluated": int(len(y_true)),
        "n_classes_in_evaluation": len(classes),
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
            np.mean([values == [UNKNOWN_CODE] for values in prediction_lists])
        ),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
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

    mismatches = [
        set(gold) != set(predicted)
        for gold, predicted in zip(gold_lists, prediction_lists, strict=True)
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
