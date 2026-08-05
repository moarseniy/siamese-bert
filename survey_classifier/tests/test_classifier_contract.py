from __future__ import annotations

import numpy as np
import pandas as pd

from survey_classifier.src.classifier import SurveyClassifier
from survey_classifier.src.data_io import load_train_data


def test_classifier_uses_common_prediction_columns() -> None:
    classifier = SurveyClassifier(
        model=None,
        example_embeddings=np.empty((0, 2), dtype=np.float32),
        example_metadata=pd.DataFrame(columns=["code"]),
        subcategory_centroids=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        subcategory_metadata=pd.DataFrame(
            [
                {
                    "code": "A1",
                    "code_name": "Зарплата",
                    "parent_code": "A",
                    "parent_name": "Финансы",
                },
                {
                    "code": "B1",
                    "code_name": "Офис",
                    "parent_code": "B",
                    "parent_name": "Условия",
                },
            ]
        ),
        parent_centroids=np.empty((0, 2), dtype=np.float32),
        parent_metadata=pd.DataFrame(),
        codebook=pd.DataFrame(),
        config={},
    )

    result = classifier._predict_from_embedding(
        text="маленькая зарплата",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        top_k=2,
        threshold=0.5,
        max_labels=6,
        margin_threshold=0.05,
        nearest_k=0,
    )

    common_columns = {
        "predicted_codes",
        "predicted_names",
        "predicted_parent_codes",
        "predicted_parent_names",
        "confidence",
        "margin",
        "top_candidates",
        "needs_review",
    }
    assert common_columns <= set(result)
    assert result["predicted_codes"] == ["A1"]
    assert result["predicted_parent_codes"] == ["A"]


def test_training_loader_supports_common_csv_and_context_keys(
    tmp_path,
) -> None:
    codebook_path = tmp_path / "codes.csv"
    pd.DataFrame(
        {
            "Код": ["A1"],
            "Категория": ["Финансы"],
            "Подкатегория": ["Зарплата"],
        }
    ).to_csv(codebook_path, index=False, encoding="utf-8-sig")
    input_path = tmp_path / "answers.csv"
    pd.DataFrame(
        {
            "Ответ": ["маленькая зарплата"],
            "Вопрос": ["Что хотелось бы улучшить?"],
            "Коды_новые": ["A1"],
        }
    ).to_csv(input_path, index=False, encoding="utf-8-sig")

    loaded = load_train_data(
        input_path,
        codebook_path,
        text_col="Ответ",
        codes_col="Коды_новые",
        context_col="Вопрос",
    )

    assert loaded.iloc[0]["text"] == (
        "Контекст: Что хотелось бы улучшить?\nОтвет: маленькая зарплата"
    )
