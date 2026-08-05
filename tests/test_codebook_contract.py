from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bert_classifier.src.data_io import parse_codebook as parse_bert_codebook
from cross_encoder_classifier.src.data_io import (
    parse_codebook as parse_cross_encoder_codebook,
)
from llm_classifier.src.data_io import parse_codebook as parse_llm_codebook
from survey_classifier.src.data_io import parse_codebook as parse_survey_codebook
from tfidf_classifier.src.data_io import parse_codebook as parse_tfidf_codebook

PARSERS = [
    parse_survey_codebook,
    parse_bert_codebook,
    parse_tfidf_codebook,
    parse_llm_codebook,
    parse_cross_encoder_codebook,
]


def _write_codebook(path: Path) -> None:
    pd.DataFrame(
        {
            "Код": ["А1", "A2", "B1"],
            "Категория": ["Финансы", "Финансы", "Условия труда"],
            "Подкатегория": ["Зарплата", "Премии", "Рабочее место"],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


@pytest.mark.parametrize("parser", PARSERS)
def test_all_projects_share_csv_codebook_contract(parser, tmp_path: Path) -> None:
    path = tmp_path / "codes.csv"
    _write_codebook(path)

    codebook = parser(path).set_index("code")

    assert list(codebook.index) == ["A1", "A2", "B1"]
    assert codebook.loc["A1", "name"] == "Зарплата"
    assert codebook.loc["A1", "parent_code"] == "A"
    assert codebook.loc["A1", "parent_name"] == "Финансы"
    assert not bool(codebook.loc["A1", "is_parent"])


@pytest.mark.parametrize("parser", PARSERS)
def test_all_projects_reject_inconsistent_parent_names(parser, tmp_path: Path) -> None:
    path = tmp_path / "codes.csv"
    pd.DataFrame(
        {
            "Код": ["A1", "A2"],
            "Категория": ["Финансы", "Другое название"],
            "Подкатегория": ["Зарплата", "Премии"],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="Different Категория"):
        parser(path)
