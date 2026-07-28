from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from survey_classifier.src.tfidf import TfidfSurveyClassifier, train_tfidf_from_data
from survey_classifier.src.utils import read_json


class TfidfPipelineTests(unittest.TestCase):
    def test_train_save_load_and_predict_russian_texts(self) -> None:
        samples = []
        categories = {
            "A1": ("Зарплата", "A", "Финансы", ["низкая зарплата", "маленький оклад", "хочу больше денег"]),
            "B1": ("Офис", "B", "Условия", ["плохой офис", "неудобное рабочее место", "холодно в помещении"]),
            "C1": (
                "Руководитель",
                "C",
                "Управление",
                ["плохой начальник", "проблемы с руководителем", "руководство не слушает"],
            ),
        }
        row_id = 0
        for code, (name, parent, parent_name, texts) in categories.items():
            for repetition in range(4):
                for text in texts:
                    samples.append(
                        {
                            "row_id": row_id,
                            "text": f"{text} {repetition}",
                            "codes": code,
                            "code": code,
                            "code_name": name,
                            "parent_code": parent,
                            "parent_name": parent_name,
                        }
                    )
                    row_id += 1
        codebook = pd.DataFrame(
            [
                {
                    "code": code,
                    "name": name,
                    "parent_code": parent,
                    "parent_name": parent_name,
                    "is_parent": False,
                }
                for code, (name, parent, parent_name, _) in categories.items()
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "tfidf_out"

            artifact_path = train_tfidf_from_data(
                long_df=pd.DataFrame(samples),
                codebook=codebook,
                out_dir=output_dir,
                val_size=0.15,
                test_size=0.15,
                seed=42,
                min_df=1,
                word_max_features=2_000,
                char_max_features=3_000,
                n_jobs=1,
            )
            classifier = TfidfSurveyClassifier.load(output_dir)
            predictions = classifier.predict_batch(
                ["зарплата и оклад очень маленькие"],
                top_k=3,
                margin_threshold=0.0,
            )
            config = read_json(output_dir / "tfidf_config.json")

            self.assertTrue(artifact_path.exists())
            self.assertTrue((output_dir / "metrics_test.json").exists())
            self.assertTrue((output_dir / "metrics_test_per_class.csv").exists())
            self.assertTrue((output_dir / "predictions_test.csv").exists())
            self.assertIn("A1", predictions.iloc[0]["predicted_codes"])
            self.assertEqual(config["seed"], 42)
            self.assertGreater(config["metrics"]["test"]["micro_f1"], 0.0)

    def test_empty_batch_has_stable_columns(self) -> None:
        classifier = TfidfSurveyClassifier(
            vectorizer=None,
            classifier=None,
            classes=[],
            threshold=0.5,
            codebook=pd.DataFrame(),
        )

        result = classifier.predict_batch([])

        self.assertTrue(result.empty)
        self.assertIn("predicted_codes", result.columns)


if __name__ == "__main__":
    unittest.main()
