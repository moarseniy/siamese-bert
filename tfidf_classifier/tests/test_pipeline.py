from __future__ import annotations

from pathlib import Path

import pandas as pd

from tfidf_classifier.src.data_io import load_labeled_data, split_train_val_test
from tfidf_classifier.src.model import TfidfSurveyClassifier, train_tfidf
from tfidf_classifier.src.predict import classify_file
from tfidf_classifier.src.utils import read_json


def _write_dataset(root: Path) -> tuple[Path, Path]:
    codebook_path = root / "codes.csv"
    pd.DataFrame(
        {
            "Код": ["A1", "B1", "C1"],
            "Категория": ["Финансы", "Условия", "Управление"],
            "Подкатегория": ["Зарплата", "Офис", "Руководитель"],
        }
    ).to_csv(codebook_path, index=False, encoding="utf-8-sig")
    examples = {
        "A1": ["низкая зарплата", "маленький оклад", "хочу больше денег"],
        "B1": ["плохой офис", "неудобное рабочее место", "холодно в помещении"],
        "C1": [
            "плохой начальник",
            "проблемы с руководителем",
            "руководство не слушает",
        ],
    }
    rows = []
    for code, texts in examples.items():
        for repetition in range(5):
            for text in texts:
                rows.append(
                    {
                        "Ответ": f"{text} {repetition}",
                        "Коды_новые": code,
                    }
                )
    rows.append(
        {
            "Ответ": "маленькая зарплата и плохой офис",
            "Коды_новые": "A1, B1",
        }
    )
    train_path = root / "train.csv"
    pd.DataFrame(rows).to_csv(train_path, index=False, encoding="utf-8-sig")
    return train_path, codebook_path


def test_train_save_load_predict_and_reports(tmp_path: Path) -> None:
    train_path, codebook_path = _write_dataset(tmp_path)
    output_dir = tmp_path / "model_out"

    artifact_path = train_tfidf(
        train_path=train_path,
        codebook_path=codebook_path,
        output_dir=output_dir,
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

    assert artifact_path.exists()
    assert (output_dir / "test_metrics.json").exists()
    assert (output_dir / "test_per_class.csv").exists()
    assert (output_dir / "test_errors.csv").exists()
    assert (output_dir / "label_distribution.csv").exists()
    assert "A1" in predictions.iloc[0]["predicted_codes"]
    assert config["seed"] == 42
    assert config["metrics"]["test"]["micro_f1"] > 0.0

    prediction_path = tmp_path / "checked.csv"
    classify_file(
        input_path=train_path,
        output_path=prediction_path,
        model_dir=output_dir,
        gold_codes_col="Коды_новые",
    )
    assert prediction_path.exists()
    assert (tmp_path / "checked_stats.json").exists()
    assert (tmp_path / "checked_per_class.csv").exists()
    assert (tmp_path / "checked_errors.csv").exists()


def test_split_is_deterministic_and_keeps_multilabel_row_together(
    tmp_path: Path,
) -> None:
    train_path, codebook_path = _write_dataset(tmp_path)
    data, _ = load_labeled_data(train_path, codebook_path)

    first = split_train_val_test(data, val_size=0.2, test_size=0.2, seed=42)
    second = split_train_val_test(data, val_size=0.2, test_size=0.2, seed=42)

    first_ids = {name: set(frame["row_id"]) for name, frame in first.items()}
    second_ids = {name: set(frame["row_id"]) for name, frame in second.items()}
    assert first_ids == second_ids
    assert first_ids["train"].isdisjoint(first_ids["val"])
    assert first_ids["train"].isdisjoint(first_ids["test"])
    assert first_ids["val"].isdisjoint(first_ids["test"])


def test_empty_batch_has_stable_columns() -> None:
    classifier = TfidfSurveyClassifier(
        vectorizer=None,
        classifier=None,
        classes=[],
        threshold=0.5,
        max_labels=6,
        codebook=pd.DataFrame(),
    )

    result = classifier.predict_batch([])

    assert result.empty
    assert "predicted_codes" in result.columns
