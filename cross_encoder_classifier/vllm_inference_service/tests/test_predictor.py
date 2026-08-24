import asyncio

from app.codebook import CodeEntry
from app.predictor import SurveyPredictor, decode_prediction


CODEBOOK = [
    CodeEntry("A1", "Оплата", "Зарплата", "зарплата"),
    CodeEntry("B1", "Команда", "Коллектив", "коллектив"),
]


class FakeTokenizer:
    def encode_batch(self, pairs):
        return [[index] for index, _ in enumerate(pairs)]


class FakeClient:
    async def classify(self, token_ids):
        return [
            [0.1, 0.1, 0.2, 0.6] if index % 2 == 0 else [0.95, 0.02, 0.02, 0.01]
            for index, _ in enumerate(token_ids)
        ]


def test_decode_prediction_maps_model_class_to_raw_sentiment() -> None:
    result = decode_prediction(
        [[0.1, 0.1, 0.2, 0.6], [0.9, 0.05, 0.03, 0.02]],
        CODEBOOK,
        threshold=0.5,
        max_labels=6,
    )
    assert result == "A1:2"


def test_predictor_handles_blank_answers() -> None:
    predictor = SurveyPredictor(
        CODEBOOK,
        FakeTokenizer(),
        FakeClient(),
        threshold=0.5,
        max_labels=6,
        pair_batch_size=10,
    )

    result = asyncio.run(predictor.predict(["текст", ""]))

    assert result == ["A1:2", "UNKNOWN"]

