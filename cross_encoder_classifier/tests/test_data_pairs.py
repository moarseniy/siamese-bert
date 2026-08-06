from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cross_encoder_classifier.src.data_io import (
    CODE_DESCRIPTION_FORMAT_LEGACY,
    ConflictingSentimentsError,
    add_after_semicolon_prefix,
    build_pairs,
    load_labeled_data,
    parse_annotations,
    parse_codebook,
    split_train_val_test,
)


def _write_codebook(path: Path) -> None:
    pd.DataFrame(
        {
            "Код": ["A1", "A2", "B1"],
            "Категория": ["Условия труда", "Условия труда", "Команда"],
            "Подкатегория": [
                "Уровень заработной платы и соответствие рынку",
                "Рабочее место и оборудование",
                "Атмосфера и отношения в коллективе",
            ],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_annotations_support_aligned_and_inline_formats() -> None:
    assert parse_annotations("А1, B1", "2, 0") == [("A1", 2), ("B1", 0)]
    assert parse_annotations("A1:2; B1=positive") == [("A1", 2), ("B1", 1)]
    assert parse_annotations("UNKNOWN", None) == []
    assert parse_annotations("A1", 2.0) == [("A1", 2)]
    assert parse_annotations("A1:2, A1:2") == [("A1", 2)]
    with pytest.raises(ConflictingSentimentsError, match="conflicting sentiments"):
        parse_annotations("A1:1, A1:2")
    with pytest.raises(ValueError, match="equal item counts"):
        parse_annotations("A1, B1", "2")
    with pytest.raises(ValueError, match="Sentiment is missing"):
        parse_annotations("A1, B1")


def test_after_semicolon_prefix_only_changes_the_suffix() -> None:
    assert (
        add_after_semicolon_prefix("8;зарплату и офис", "нужно улучшить: ")
        == "8; нужно улучшить: зарплату и офис"
    )
    assert add_after_semicolon_prefix("зарплату", "нужно улучшить:") == "зарплату"


def test_code_descriptions_exclude_ids_by_default(tmp_path: Path) -> None:
    codebook_path = tmp_path / "codes.csv"
    _write_codebook(codebook_path)

    codebook = parse_codebook(codebook_path)
    legacy = parse_codebook(
        codebook_path,
        description_format=CODE_DESCRIPTION_FORMAT_LEGACY,
    )

    assert codebook.loc[0, "description"] == (
        "Категория: Условия труда. Подкатегория: "
        "Уровень заработной платы и соответствие рынку"
    )
    assert legacy.loc[0, "description"].startswith("A1. Категория:")


def test_loading_and_pair_generation(tmp_path: Path) -> None:
    codebook_path = tmp_path / "codes.csv"
    source_path = tmp_path / "answers.csv"
    _write_codebook(codebook_path)
    pd.DataFrame(
        {
            "Ответ": [
                "8; Зарплата низкая, но коллектив отличный.",
                "Нормальное рабочее место.",
                "Ничего не могу сказать.",
                "Зарплата нравится, но иногда расстраивает.",
            ],
            "Коды_новые": ["A1, B1", "A2", "UNKNOWN", "A1, A1"],
            "Тональности": ["2, 1", "0", "", "1, 2"],
        }
    ).to_csv(source_path, index=False, encoding="utf-8-sig")

    data, codebook = load_labeled_data(
        source_path,
        codebook_path,
        after_semicolon_prefix="нужно улучшить: ",
    )
    assert len(data) == 3
    assert data.iloc[0]["annotations"] == [("A1", 2), ("B1", 1)]
    assert data.iloc[0]["text"] == (
        "8; нужно улучшить: Зарплата низкая, но коллектив отличный."
    )
    assert data.attrs["load_report"] == {
        "input_rows": 4,
        "loaded_rows": 3,
        "empty_text_rows": 0,
        "skipped_conflicting_sentiment_rows": 1,
        "skipped_conflicting_sentiment_source_rows": [5],
    }

    all_pairs = build_pairs(data.iloc[[0]], codebook, negative_ratio=None)
    assert all_pairs["code"].tolist() == ["A1", "A2", "B1"]
    assert all_pairs["label"].tolist() == [3, 0, 2]

    sampled = build_pairs(data.iloc[[1]], codebook, negative_ratio=1.0, seed=42)
    assert len(sampled) == 2
    assert (sampled["label"] == 0).sum() == 1
    assert (sampled["label"] == 1).sum() == 1


def test_split_happens_before_pairs_and_is_deterministic(tmp_path: Path) -> None:
    codebook_path = tmp_path / "codes.csv"
    _write_codebook(codebook_path)
    codebook = parse_codebook(codebook_path)
    annotations = [[("A1", 0)], [("A2", 1)], [("B1", 2)], [("A1", 2)]]
    rows = []
    for index in range(40):
        values = annotations[index % len(annotations)]
        rows.append(
            {
                "row_id": index,
                "text": f"Ответ {index}",
                "answer": f"Ответ {index}",
                "context": "",
                "codes": [code for code, _ in values],
                "annotations": values,
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

    split_pairs = {
        name: build_pairs(part, codebook, negative_ratio=None)
        for name, part in first.items()
    }
    pair_ids = {name: set(part["row_id"]) for name, part in split_pairs.items()}
    assert pair_ids == first_ids
