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

from llm_classifier.src.client import parse_classification
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
                    "codes": [code],
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


def fake_openai_module() -> ModuleType:
    module = ModuleType("openai")
    module.OpenAI = FakeOpenAI
    return module


class LLMPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCompletions.requests = []
        FakeCompletions.active_requests = 0
        FakeCompletions.max_active_requests = 0

    def test_parser_rejects_invented_codes_and_marks_review(self) -> None:
        codes, review, invalid = parse_classification(
            '{"codes":["A1","ZZ9"]}',
            allowed_codes=["A1", "B1"],
        )

        self.assertEqual(codes, ["A1"])
        self.assertTrue(review)
        self.assertEqual(invalid, ["ZZ9"])

    def test_concurrent_pipeline_preserves_order_and_calculates_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "answers.csv"
            output_path = root / "predictions.csv"
            codebook_path = root / "codes.txt"
            pd.DataFrame(
                [
                    {"Ответ": "низкая зарплата", "Коды_новые": "A1"},
                    {"Ответ": "неудобный офис", "Коды_новые": "B1"},
                    {"Ответ": "", "Коды_новые": "UNKNOWN"},
                ]
            ).to_csv(input_path, index=False, encoding="utf-8-sig")
            codebook_path.write_text(
                "A. Финансы\nA1. Зарплата\nB. Условия\nB1. Офис\n",
                encoding="utf-8",
            )

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
        self.assertEqual(set(schema["properties"]), {"codes"})
        self.assertEqual(schema["properties"]["codes"]["maxItems"], 6)
        self.assertNotIn("uniqueItems", schema["properties"]["codes"])
        self.assertEqual(first_request["max_tokens"], 64)
        self.assertEqual(
            first_request["extra_body"]["chat_template_kwargs"]["enable_thinking"],
            False,
        )
        self.assertEqual(result["predicted_codes"].tolist(), ["A1", "B1", "UNKNOWN"])
        self.assertIn("confidence", result.columns)
        self.assertIn("predicted_names", result.columns)
        self.assertIn("predicted_parent_codes", result.columns)
        self.assertIn("predicted_parent_names", result.columns)
        self.assertNotIn("explanation", result.columns)
        self.assertEqual(stats["micro_f1"], 1.0)
        self.assertEqual(stats["evaluated_rows"], 2)
        self.assertEqual(stats["failed_rows"], 1)
        self.assertEqual(stats["concurrency"], 8)
        self.assertGreater(stats["throughput_rows_per_second"], 0)
        self.assertTrue(stats_exists)
        self.assertTrue(per_class_exists)


if __name__ == "__main__":
    unittest.main()
