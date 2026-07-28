from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from survey_classifier.src.build_index import build_index
from survey_classifier.src.classifier import SurveyClassifier
from survey_classifier.src.model_input import (
    configure_model_input,
    prepare_model_text,
    prepare_model_texts,
    resolve_input_prefix,
)
from survey_classifier.src.train import train_model
from survey_classifier.src.utils import read_json


class FakePromptModel:
    prompts = {
        "categorize_topic": "categorize_topic: ",
        "paraphrase": "paraphrase: ",
    }
    default_prompt_name = "paraphrase"


class RecordingEncodeModel:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.texts = texts
        return np.asarray([[3.0, 4.0] for _ in texts], dtype=np.float32)


class FakeInputExample:
    def __init__(self, texts: list[str], label: float | None = None) -> None:
        self.texts = texts
        self.label = label


class FakeSentenceTransformer:
    prompts = {"categorize_topic": "categorize_topic: "}
    default_prompt_name = "categorize_topic"
    last_fit_texts: list[list[str]] = []
    last_encode_texts: list[str] = []

    def __init__(self, _: str, **__: object) -> None:
        self.prompts = dict(type(self).prompts)
        self.default_prompt_name = type(self).default_prompt_name

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        type(self).last_encode_texts = texts
        rows = [
            [1.0, 0.0] if "зарплат" in text else [0.0, 1.0]
            for text in texts
        ]
        return np.asarray(rows, dtype=np.float32)

    def fit(self, train_objectives: list[tuple[object, object]], output_path: str, **_: object) -> None:
        dataloader = train_objectives[0][0]
        type(self).last_fit_texts = [example.texts for example in dataloader.dataset]
        Path(output_path).mkdir(parents=True, exist_ok=True)


class FakeLoss:
    def __init__(self, *_: object, **__: object) -> None:
        pass


def fake_sentence_transformers_modules() -> dict[str, ModuleType]:
    losses_module = ModuleType("sentence_transformers.losses")
    losses_module.CosineSimilarityLoss = FakeLoss
    losses_module.MultipleNegativesRankingLoss = FakeLoss
    losses_module.TripletLoss = FakeLoss
    losses_module.TripletDistanceMetric = SimpleNamespace(COSINE="cosine")

    sentence_module = ModuleType("sentence_transformers")
    sentence_module.InputExample = FakeInputExample
    sentence_module.SentenceTransformer = FakeSentenceTransformer
    sentence_module.losses = losses_module
    return {
        "sentence_transformers": sentence_module,
        "sentence_transformers.losses": losses_module,
    }


class ModelInputTests(unittest.TestCase):
    def test_resolves_named_prompt_and_disables_default(self) -> None:
        model = FakePromptModel()

        prefix = configure_model_input(model, prompt_name="categorize_topic")

        self.assertEqual(prefix, "categorize_topic: ")
        self.assertIsNone(model.default_prompt_name)

    def test_rejects_unknown_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "Available prompts"):
            resolve_input_prefix(FakePromptModel(), prompt_name="missing")

    def test_rejects_prompt_and_manual_prefix_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "either prompt_name or input_prefix"):
            resolve_input_prefix(
                FakePromptModel(),
                prompt_name="categorize_topic",
                input_prefix="topic: ",
            )

    def test_prepares_texts_without_modifying_source_text(self) -> None:
        source = ["низкая зарплата", "нет премии"]

        prepared = prepare_model_texts(source, "categorize_topic: ")

        self.assertEqual(
            prepared,
            [
                "categorize_topic: низкая зарплата",
                "categorize_topic: нет премии",
            ],
        )
        self.assertEqual(source, ["низкая зарплата", "нет премии"])
        self.assertEqual(prepare_model_text("текст"), "текст")

    def test_classifier_uses_saved_prefix_for_inference(self) -> None:
        model = RecordingEncodeModel()
        classifier_like = SimpleNamespace(
            model=model,
            input_prefix="categorize_topic: ",
        )

        embeddings = SurveyClassifier._encode(classifier_like, ["низкая зарплата"])

        self.assertEqual(model.texts, ["categorize_topic: низкая зарплата"])
        np.testing.assert_allclose(embeddings, np.asarray([[0.6, 0.8]], dtype=np.float32))

    def test_training_and_index_store_the_same_resolved_prefix(self) -> None:
        train_df = pd.DataFrame(
            [
                {
                    "row_id": 1,
                    "text": "низкая зарплата",
                    "codes": "A1",
                    "code": "A1",
                    "code_name": "Зарплата",
                    "parent_code": "A",
                    "parent_name": "Финансы",
                },
                {
                    "row_id": 2,
                    "text": "маленький оклад",
                    "codes": "A1",
                    "code": "A1",
                    "code_name": "Зарплата",
                    "parent_code": "A",
                    "parent_name": "Финансы",
                },
                {
                    "row_id": 3,
                    "text": "плохой офис",
                    "codes": "B1",
                    "code": "B1",
                    "code_name": "Офис",
                    "parent_code": "B",
                    "parent_name": "Условия",
                },
                {
                    "row_id": 4,
                    "text": "неудобное помещение",
                    "codes": "B1",
                    "code": "B1",
                    "code_name": "Офис",
                    "parent_code": "B",
                    "parent_name": "Условия",
                },
            ]
        )
        codebook = pd.DataFrame(
            [
                {
                    "code": "A1",
                    "name": "Зарплата",
                    "parent_code": "A",
                    "parent_name": "Финансы",
                    "is_parent": False,
                },
                {
                    "code": "B1",
                    "name": "Офис",
                    "parent_code": "B",
                    "parent_name": "Условия",
                    "is_parent": False,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.dict(sys.modules, fake_sentence_transformers_modules()):
                model_dir = train_model(
                    train_df=train_df,
                    out_dir=output_dir,
                    base_model="ai-forever/FRIDA",
                    prompt_name="categorize_topic",
                    min_class_size=2,
                    max_pairs_per_code=1,
                    negative_ratio=1.0,
                    epochs=1,
                    batch_size=2,
                    show_progress_bar=False,
                )
                build_index(
                    train_df=train_df,
                    codebook_df=codebook,
                    out_dir=output_dir,
                    model_dir=model_dir,
                    input_prefix="categorize_topic: ",
                    prompt_name_for_config="categorize_topic",
                    show_progress_bar=False,
                )

            train_config = read_json(output_dir / "train_config.json")
            index_config = read_json(output_dir / "index" / "index_config.json")

        self.assertEqual(train_config["input_prefix"], "categorize_topic: ")
        self.assertEqual(index_config["input_prefix"], "categorize_topic: ")
        self.assertEqual(index_config["prompt_name"], "categorize_topic")
        self.assertTrue(FakeSentenceTransformer.last_fit_texts)
        self.assertTrue(
            all(
                text.startswith("categorize_topic: ")
                for example_texts in FakeSentenceTransformer.last_fit_texts
                for text in example_texts
            )
        )
        self.assertTrue(
            all(
                text.startswith("categorize_topic: ")
                for text in FakeSentenceTransformer.last_encode_texts
            )
        )


if __name__ == "__main__":
    unittest.main()
