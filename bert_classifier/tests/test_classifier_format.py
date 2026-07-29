from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import BertSurveyClassifier  # noqa: E402


def test_format_predictions_supports_multiple_codes() -> None:
    classifier = BertSurveyClassifier.__new__(BertSurveyClassifier)
    classifier.classes = ["A1", "A2", "B1"]
    classifier.threshold = 0.5
    classifier.max_labels = 2
    classifier.name_by_code = {
        "A": "Сервис",
        "A1": "Скорость",
        "A2": "Вежливость",
        "B": "Продукт",
        "B1": "Качество",
    }
    classifier.parent_by_code = {"A1": "A", "A2": "A", "B1": "B"}
    classifier.parent_name_by_code = {
        "A1": "Сервис",
        "A2": "Сервис",
        "B1": "Продукт",
    }

    result = classifier.format_predictions(
        np.array([[0.8, 0.7, 0.1], [0.2, 0.1, 0.3]], dtype=np.float32)
    )

    assert result.loc[0, "predicted_codes"] == "A1, A2"
    assert result.loc[0, "predicted_parent_codes"] == "A"
    assert result.loc[0, "predicted_parent_names"] == "Сервис"
    assert result.loc[1, "predicted_codes"] == "UNKNOWN"
    assert bool(result.loc[1, "needs_review"])
