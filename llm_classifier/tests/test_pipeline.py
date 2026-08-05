from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd

from llm_classifier.src.client import VLLMSurveyClassifier, parse_classification
from llm_classifier.src.metrics import calculate_metrics
from llm_classifier.src.pipeline import run_pipeline


class FakeCompletions:
    requests: list[dict[str, object]] = []
    active_requests = 0
    max_active_requests = 0
    lock = threading.Lock()

    def create(self, **request: object) -> object:
        with type(self).lock:
            type(self).requests.append(request)
            type(self).active_requests += 1
            type(self).max_active_requests = max(
                type(self).max_active_requests,
                type(self).active_requests,
            )
        try:
            time.sleep(0.05)
            messages = request["messages"]
            answer_prompt = messages[1]["content"]
            code = "A1" if "зарплата" in answer_prompt else "B1"
            content = json.dumps(
                {
                    "labels": [{"code": code, "sentiment": 2}],
                },
                ensure_ascii=False,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            )
        finally:
            with type(self).lock:
                type(self).active_requests -= 1


class FakeOpenAI:
    def __init__(self, **_: object) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="Qwen/Qwen3.5-test")])
        )


class TruncatedCompletions:
    requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        type(self).requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content="длинное рассуждение",
                    ),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=777),
        )


class TruncatedOpenAI:
    def __init__(self, **_: object) -> None:
        self.chat = SimpleNamespace(completions=TruncatedCompletions())
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="Qwen/test")])
        )


def fake_openai_module() -> ModuleType:
    module = ModuleType("openai")
    module.OpenAI = FakeOpenAI
    return module


def truncated_openai_module() -> ModuleType:
    module = ModuleType("openai")
    module.OpenAI = TruncatedOpenAI
    return module


class LLMPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCompletions.requests = []
        FakeCompletions.active_requests = 0
        FakeCompletions.max_active_requests = 0
        TruncatedCompletions.requests = []

    def test_parser_rejects_invented_codes_and_marks_review(self) -> None:
        codes, sentiments, review, invalid = parse_classification(
            '{"labels":['
            '{"code":"A1","sentiment":1},'
            '{"code":"ZZ9","sentiment":2}]}',
            allowed_codes=["A1", "B1"],
        )

        self.assertEqual(codes, ["A1"])
        self.assertEqual(sentiments, {"A1": 1})
        self.assertTrue(review)
        self.assertEqual(invalid, ["ZZ9"])

    def test_parser_deduplicates_equal_labels_and_rejects_bad_sentiment(self) -> None:
        codes, sentiments, review, invalid = parse_classification(
            '{"labels":['
            '{"code":"A1","sentiment":0},'
            '{"code":"A1","sentiment":0},'
            '{"code":"B1","sentiment":4}]}',
            allowed_codes=["A1", "B1"],
        )

        self.assertEqual(codes, ["A1"])
        self.assertEqual(sentiments, {"A1": 0})
        self.assertTrue(review)
        self.assertEqual(invalid, ["B1:4"])

    def test_parser_maps_empty_labels_to_unknown(self) -> None:
        codes, sentiments, review, invalid = parse_classification(
            '{"labels":[]}',
            allowed_codes=["A1", "B1"],
        )

        self.assertEqual(codes, ["UNKNOWN"])
        self.assertEqual(sentiments, {})
        self.assertTrue(review)
        self.assertEqual(invalid, [])

    def test_metrics_distinguish_code_match_from_sentiment_match(self) -> None:
        frame = pd.DataFrame(
            {
                "Коды_новые": ["A1:2"],
                "predicted_code_sentiments": ["A1:1"],
            }
        )

        metrics, per_class, errors = calculate_metrics(
            frame,
            gold_codes_col="Коды_новые",
            known_codes={"A1"},
        )

        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["joint_micro_f1"], 0.0)
        self.assertEqual(metrics["gold_code_sentiment_accuracy"], 0.0)
        self.assertEqual(per_class.iloc[0]["gold_code_sentiment_accuracy"], 0.0)
        self.assertEqual(len(errors), 1)

    def test_thinking_uses_separate_budget_and_reports_truncation(self) -> None:
        with patch.dict(sys.modules, {"openai": truncated_openai_module()}):
            classifier = VLLMSurveyClassifier(
                base_url="http://127.0.0.1:8000/v1",
                api_key="EMPTY",
                model="Qwen/test",
                codebook_text="A1. Зарплата",
                allowed_codes=["A1"],
                max_retries=2,
                max_tokens=64,
                thinking_max_tokens=777,
                enable_thinking=True,
            )
            result = classifier.classify("низкая зарплата")

        self.assertEqual(len(TruncatedCompletions.requests), 1)
        self.assertEqual(TruncatedCompletions.requests[0]["max_tokens"], 777)
        self.assertEqual(TruncatedCompletions.requests[0]["temperature"], 0.6)
        self.assertEqual(TruncatedCompletions.requests[0]["top_p"], 0.95)
        self.assertEqual(
            TruncatedCompletions.requests[0]["extra_body"]["top_k"],
            20,
        )
        self.assertIn("ThinkingOutputTruncatedError", result.error)
        self.assertIn("--thinking-max-tokens", result.error)

    def test_concurrent_pipeline_preserves_order_and_calculates_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "answers.csv"
            output_path = root / "predictions.csv"
            codebook_path = root / "codes.csv"
            pd.DataFrame(
                [
                    {"Ответ": "низкая зарплата", "Коды_новые": "A1:2"},
                    {"Ответ": "неудобный офис", "Коды_новые": "B1:2"},
                    {"Ответ": "", "Коды_новые": "UNKNOWN"},
                ]
            ).to_csv(input_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                {
                    "Код": ["A1", "B1"],
                    "Категория": ["Финансы", "Условия"],
                    "Подкатегория": ["Зарплата", "Офис"],
                }
            ).to_csv(codebook_path, index=False, encoding="utf-8-sig")

            with patch.dict(sys.modules, {"openai": fake_openai_module()}):
                saved_path, stats = run_pipeline(
                    input_path=input_path,
                    output_path=output_path,
                    codebook_path=codebook_path,
                    model="Qwen/Qwen3.5-test",
                    gold_codes_col="Коды_новые",
                    checkpoint_every=1,
                )

            result = pd.read_csv(saved_path, encoding="utf-8-sig")
            stats_path = root / "predictions_stats.json"
            per_class_path = root / "predictions_per_class.csv"
            stats_exists = stats_path.exists()
            per_class_exists = per_class_path.exists()

        self.assertEqual(len(FakeCompletions.requests), 2)
        self.assertEqual(FakeCompletions.max_active_requests, 2)
        first_request = FakeCompletions.requests[0]
        self.assertIn("response_format", first_request)
        schema = first_request["response_format"]["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), {"labels"})
        labels_schema = schema["properties"]["labels"]
        self.assertEqual(labels_schema["maxItems"], 6)
        self.assertNotIn("uniqueItems", labels_schema)
        self.assertEqual(
            set(labels_schema["items"]["properties"]), {"code", "sentiment"}
        )
        self.assertEqual(first_request["max_tokens"], 128)
        self.assertEqual(
            first_request["extra_body"]["chat_template_kwargs"]["enable_thinking"],
            False,
        )
        self.assertEqual(result["predicted_codes"].tolist(), ["A1", "B1", "UNKNOWN"])
        self.assertEqual(
            result["predicted_code_sentiments"].tolist(),
            ["A1:2", "B1:2", "UNKNOWN"],
        )
        self.assertEqual(result["predicted_sentiments"].iloc[:2].tolist(), [2.0, 2.0])
        self.assertIn("confidence", result.columns)
        self.assertIn("predicted_names", result.columns)
        self.assertIn("predicted_parent_codes", result.columns)
        self.assertIn("predicted_parent_names", result.columns)
        self.assertNotIn("explanation", result.columns)
        self.assertEqual(stats["micro_f1"], 1.0)
        self.assertEqual(stats["joint_micro_f1"], 1.0)
        self.assertEqual(stats["gold_code_sentiment_accuracy"], 1.0)
        self.assertEqual(stats["evaluated_rows"], 2)
        self.assertEqual(stats["failed_rows"], 1)
        self.assertEqual(stats["concurrency"], 8)
        self.assertGreater(stats["throughput_rows_per_second"], 0)
        self.assertTrue(stats_exists)
        self.assertTrue(per_class_exists)


if __name__ == "__main__":
    unittest.main()
