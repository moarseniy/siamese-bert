from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import split_codes


def calculate_metrics(
    frame: pd.DataFrame,
    gold_codes_col: str,
    known_codes: set[str],
    prediction_col: str = "predicted_codes",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    gold_lists = [
        [code for code in split_codes(value) if code in known_codes]
        for value in frame[gold_codes_col]
    ]
    prediction_lists = [
        [code for code in split_codes(value) if code in known_codes]
        for value in frame[prediction_col]
    ]
    labeled_mask = np.array([bool(codes) for codes in gold_lists])
    evaluated = frame.loc[labeled_mask].copy()
    evaluated_gold = [
        codes for codes, keep in zip(gold_lists, labeled_mask, strict=True) if keep
    ]
    evaluated_predictions = [
        codes
        for codes, keep in zip(prediction_lists, labeled_mask, strict=True)
        if keep
    ]
    base_metrics = {
        "input_rows": int(len(frame)),
        "evaluated_rows": int(labeled_mask.sum()),
        "rows_without_known_gold_codes": int((~labeled_mask).sum()),
    }
    columns = ["code", "support", "precision", "recall", "f1"]
    if not evaluated_gold:
        return base_metrics, pd.DataFrame(columns=columns), pd.DataFrame()

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import MultiLabelBinarizer

    classes = sorted(known_codes)
    binarizer = MultiLabelBinarizer(classes=classes)
    binarizer.fit([classes])
    y_true = binarizer.transform(evaluated_gold)
    y_pred = binarizer.transform(evaluated_predictions)
    single_mask = y_true.sum(axis=1) == 1
    single_top1: float | None = None
    if single_mask.any():
        true_top1 = y_true[single_mask].argmax(axis=1)
        predicted_first = [
            classes.index(evaluated_predictions[index][0])
            if evaluated_predictions[index]
            else -1
            for index in np.flatnonzero(single_mask)
        ]
        single_top1 = float(
            (true_top1 == np.asarray(predicted_first, dtype=int)).mean()
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
        "single_label_top1_accuracy": single_top1,
        "unknown_prediction_rate": float(
            np.mean([not values for values in evaluated_predictions])
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
        for gold, predicted in zip(
            evaluated_gold,
            evaluated_predictions,
            strict=True,
        )
    ]
    return metrics, per_class, evaluated.loc[mismatches].copy()


def metrics_paths(output_path: str | Path) -> tuple[Path, Path, Path]:
    prefix = Path(output_path).with_suffix("")
    return (
        prefix.parent / f"{prefix.name}_stats.json",
        prefix.parent / f"{prefix.name}_per_class.csv",
        prefix.parent / f"{prefix.name}_errors.csv",
    )
