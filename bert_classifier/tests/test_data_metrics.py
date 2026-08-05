from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bert_classifier.src.data_io import (
    load_labeled_data,
    parse_codebook,
    split_codes,
    split_train_val_test,
)
from bert_classifier.src.metrics import calculate_metrics, encode_labels, select_threshold


def _write_codebook(path: Path) -> None:
    pd.DataFrame(
        {
            "Код": ["A1", "A2", "B1"],
            "Категория": ["Сервис", "Сервис", "Продукт"],
            "Подкатегория": [
                "Скорость обслуживания",
                "Вежливость",
                "Качество продукта",
            ],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_codebook_and_code_normalization(tmp_path: Path) -> None:
    codebook_path = tmp_path / "codes.csv"
    _write_codebook(codebook_path)

    codebook = parse_codebook(codebook_path)

    assert split_codes("А1; B1, A1") == ["A1", "B1"]
    assert codebook.set_index("code").loc["A1", "parent_code"] == "A"
    assert codebook.set_index("code").loc["A1", "parent_name"] == "Сервис"


def test_load_labeled_data_preserves_multilabel_and_context(tmp_path: Path) -> None:
    codebook_path = tmp_path / "codes.csv"
    source_path = tmp_path / "answers.csv"
    _write_codebook(codebook_path)
    pd.DataFrame(
        {
            "Ответ": ["Быстро и вежливо", "Не знаю", "", "Хороший товар"],
            "Вопрос": ["Как прошел визит?", "", "Пусто", "Что понравилось?"],
            "Коды_новые": ["A1, A2", "UNKNOWN", "A1", "B1"],
        }
    ).to_csv(source_path, index=False, encoding="utf-8-sig")

    data, _ = load_labeled_data(
        source_path,
        codebook_path,
        context_col="Вопрос",
    )

    assert len(data) == 2
    assert data.iloc[0]["codes"] == ["A1", "A2"]
    assert data.iloc[0]["text"] == (
        "Контекст: Как прошел визит?\nОтвет: Быстро и вежливо"
    )
    assert data.iloc[1]["text"] == (
        "Контекст: Что понравилось?\nОтвет: Хороший товар"
    )


def test_split_is_deterministic_disjoint_and_covers_labels() -> None:
    rows = []
    code_sets = [["A1"], ["A2"], ["B1"], ["A1", "A2"]]
    for index in range(40):
        rows.append(
            {
                "row_id": index,
                "text": f"Ответ {index}",
                "answer": f"Ответ {index}",
                "context": "",
                "codes": code_sets[index % len(code_sets)],
            }
        )
    frame = pd.DataFrame(rows)

    first = split_train_val_test(frame, val_size=0.2, test_size=0.2, seed=42)
    second = split_train_val_test(frame, val_size=0.2, test_size=0.2, seed=42)

    first_ids = {name: set(part["row_id"]) for name, part in first.items()}
    second_ids = {name: set(part["row_id"]) for name, part in second.items()}
    assert first_ids == second_ids
    assert first_ids["train"].isdisjoint(first_ids["val"])
    assert first_ids["train"].isdisjoint(first_ids["test"])
    assert first_ids["val"].isdisjoint(first_ids["test"])
    assert set.union(*first_ids.values()) == set(frame["row_id"])
    train_codes = {code for codes in first["train"]["codes"] for code in codes}
    assert train_codes == {"A1", "A2", "B1"}


def test_metrics_and_threshold_selection() -> None:
    classes = ["A1", "A2", "B1"]
    y_true = encode_labels([["A1"], ["A2", "B1"], ["B1"]], classes)
    probabilities = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.1, 0.8, 0.7],
            [0.2, 0.1, 0.9],
        ],
        dtype=np.float32,
    )

    metrics, per_class = calculate_metrics(
        y_true,
        probabilities,
        classes,
        threshold=0.5,
        max_labels=6,
    )
    threshold, search = select_threshold(
        y_true,
        probabilities,
        classes,
        metric="micro_f1",
    )

    assert metrics["micro_f1"] == 1.0
    assert metrics["subset_accuracy"] == 1.0
    assert set(per_class["code"]) == set(classes)
    assert 0.1 <= threshold <= 0.9
    assert not search.empty
