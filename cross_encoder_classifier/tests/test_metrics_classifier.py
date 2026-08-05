from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cross_encoder_classifier.src.classifier import CrossEncoderSurveyClassifier
from cross_encoder_classifier.src.metrics import (
    calculate_response_metrics,
    decode_response_predictions,
    select_presence_threshold,
)
from cross_encoder_classifier.src.predict import _gold_targets


def _probabilities() -> np.ndarray:
    return np.array(
        [
            [
                [0.05, 0.05, 0.10, 0.80],
                [0.90, 0.05, 0.03, 0.02],
                [0.10, 0.10, 0.75, 0.05],
            ],
            [
                [0.85, 0.05, 0.05, 0.05],
                [0.10, 0.75, 0.10, 0.05],
                [0.90, 0.03, 0.04, 0.03],
            ],
        ],
        dtype=np.float32,
    )


def test_response_metrics_include_code_and_sentiment_quality() -> None:
    probabilities = _probabilities()
    y_true = np.array([[3, 0, 2], [0, 1, 0]], dtype=np.int8)
    predicted = decode_response_predictions(probabilities, threshold=0.5, max_labels=6)
    metrics, per_code = calculate_response_metrics(
        y_true, probabilities, ["A1", "A2", "B1"], threshold=0.5
    )
    threshold, search = select_presence_threshold(
        y_true, probabilities, ["A1", "A2", "B1"]
    )

    assert np.array_equal(predicted, y_true)
    assert metrics["micro_f1"] == 1.0
    assert metrics["joint_micro_f1"] == 1.0
    assert metrics["gold_code_sentiment_accuracy"] == 1.0
    assert set(per_code["code"]) == {"A1", "A2", "B1"}
    assert 0.1 <= threshold <= 0.9
    assert not search.empty


def test_prediction_format_keeps_sentiment_attached_to_each_code() -> None:
    classifier = CrossEncoderSurveyClassifier.__new__(CrossEncoderSurveyClassifier)
    classifier.codes = ["A1", "A2", "B1"]
    classifier.threshold = 0.5
    classifier.max_labels = 6
    classifier.name_by_code = {
        "A": "Условия труда",
        "A1": "Зарплата",
        "A2": "Рабочее место",
        "B": "Команда",
        "B1": "Коллектив",
    }
    classifier.parent_by_code = {"A1": "A", "A2": "A", "B1": "B"}
    classifier.parent_name_by_code = {
        "A1": "Условия труда",
        "A2": "Условия труда",
        "B1": "Команда",
    }

    result = classifier.format_predictions(_probabilities(), threshold=0.5)

    assert result.loc[0, "predicted_codes"] == "A1, B1"
    assert result.loc[0, "predicted_sentiments"] == "2, 1"
    assert result.loc[0, "predicted_code_sentiments"] == "A1:2, B1:1"
    assert result.loc[0, "predicted_parent_codes"] == "A, B"


def test_pair_inference_uses_answer_major_code_order() -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] = []

        def __call__(self, answers, text_pair, **kwargs):
            self.pairs.extend(zip(answers, text_pair))
            return {"input_ids": torch.zeros((len(answers), 1), dtype=torch.long)}

    class FakeModel:
        def __call__(self, **kwargs):
            batch_size = len(kwargs["input_ids"])
            logits = torch.tensor([[4.0, 1.0, 0.0, -1.0]]).repeat(batch_size, 1)
            return type("Output", (), {"logits": logits})()

    classifier = CrossEncoderSurveyClassifier.__new__(CrossEncoderSurveyClassifier)
    classifier.codes = ["A1", "B1"]
    classifier.descriptions = ["A1. Зарплата", "B1. Коллектив"]
    classifier.max_length = 32
    classifier.torch = torch
    classifier.device = torch.device("cpu")
    classifier.tokenizer = FakeTokenizer()
    classifier.model = FakeModel()

    probabilities = classifier.predict_probabilities(
        ["Первый", "Второй"], batch_size=3, show_progress=False
    )

    assert probabilities.shape == (2, 2, 4)
    assert classifier.tokenizer.pairs == [
        ("Первый", "A1. Зарплата"),
        ("Первый", "B1. Коллектив"),
        ("Второй", "A1. Зарплата"),
        ("Второй", "B1. Коллектив"),
    ]


def test_gold_metrics_skip_conflicting_sentiment_rows() -> None:
    classifier = CrossEncoderSurveyClassifier.__new__(CrossEncoderSurveyClassifier)
    classifier.codes = ["A1", "B1"]
    source = pd.DataFrame(
        {
            "Коды_новые": ["A1:1, A1:2", "B1:0"],
        }
    )

    targets, valid_mask, conflicting_rows = _gold_targets(
        source,
        codes_col="Коды_новые",
        sentiments_col=None,
        classifier=classifier,
    )

    assert valid_mask.tolist() == [False, True]
    assert conflicting_rows == [2]
    assert targets[1].tolist() == [0, 1]
