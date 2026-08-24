import json
from pathlib import Path

import pytest

from app.config import validate_bge_reranker_v2_m3


def write_model_config(path: Path, **overrides: object) -> None:
    value = {
        "architectures": ["XLMRobertaForSequenceClassification"],
        "model_type": "xlm-roberta",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "vocab_size": 250002,
        "id2label": {
            "0": "absent",
            "1": "neutral",
            "2": "positive",
            "3": "negative",
        },
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_accepts_fine_tuned_bge_reranker_v2_m3(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_model_config(path)

    validate_bge_reranker_v2_m3(path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_size": 768}, "hidden_size"),
        (
            {"architectures": ["BertForSequenceClassification"]},
            "XLMRobertaForSequenceClassification",
        ),
        ({"id2label": {"0": "LABEL_0"}}, "id2label"),
    ],
)
def test_rejects_other_model_artifacts(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    path = tmp_path / "config.json"
    write_model_config(path, **override)

    with pytest.raises(ValueError, match=message):
        validate_bge_reranker_v2_m3(path)
