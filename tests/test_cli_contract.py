from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_SCRIPTS = [
    ROOT / "survey_classifier/scripts/train.py",
    ROOT / "bert_classifier/scripts/train.py",
    ROOT / "tfidf_classifier/scripts/train.py",
    ROOT / "cross_encoder_classifier/scripts/train.py",
]
PREDICT_SCRIPTS = [
    ROOT / "survey_classifier/scripts/predict.py",
    ROOT / "bert_classifier/scripts/predict.py",
    ROOT / "tfidf_classifier/scripts/predict.py",
    ROOT / "llm_classifier/scripts/predict.py",
    ROOT / "cross_encoder_classifier/scripts/predict.py",
]


def _help(script: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_training_commands_share_common_keys() -> None:
    for script in TRAIN_SCRIPTS:
        help_text = _help(script)
        for key in (
            "--train",
            "--codebook",
            "--out-dir",
            "--text-col",
            "--codes-col",
            "--context-col",
            "--csv-sep",
            "--val-size",
            "--test-size",
            "--seed",
        ):
            assert key in help_text, (script, key)


def test_prediction_commands_share_common_keys() -> None:
    for script in PREDICT_SCRIPTS:
        help_text = _help(script)
        for key in (
            "--input",
            "--output",
            "--text-col",
            "--context-col",
            "--gold-codes-col",
            "--csv-sep",
            "--max-labels",
        ):
            assert key in help_text, (script, key)
        for old_key in (
            "--input-xlsx",
            "--output-xlsx",
            "--gold-col",
            "--codebook-txt",
        ):
            assert old_key not in help_text, (script, old_key)
