from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_MODEL = "BAAI/bge-reranker-v2-m3"
MODEL_CLASS_NAMES = ["absent", "neutral", "positive", "negative"]
_BGE_M3_SIGNATURE = {
    "model_type": "xlm-roberta",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "vocab_size": 250002,
}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    return None if raw is None or not raw.strip() else float(raw)


def _optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    return None if raw is None or not raw.strip() else int(raw)


@dataclass(frozen=True)
class ClassifierConfig:
    threshold: float
    max_labels: int
    max_length: int
    code_description_format: str
    after_semicolon_prefix: str

    @classmethod
    def load(
        cls,
        path: Path,
        threshold_override: float | None = None,
        max_labels_override: int | None = None,
    ) -> "ClassifierConfig":
        if not path.is_file():
            raise FileNotFoundError(f"Classifier config not found: {path}")
        value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if value.get("pipeline") != "cross_encoder_code_sentiment_classifier":
            raise ValueError(
                "classifier_config.json was not created by the cross-encoder "
                "code/sentiment training pipeline."
            )
        if int(value.get("num_model_classes", 4)) != 4:
            raise ValueError("The service requires a four-class model.")
        model_classes = value.get("model_classes")
        if model_classes is not None:
            actual_classes = [
                str(model_classes.get(str(index), model_classes.get(index, "")))
                for index in range(4)
            ]
            if actual_classes != MODEL_CLASS_NAMES:
                raise ValueError(
                    "classifier_config.json has an incompatible class mapping."
                )

        threshold = (
            float(value["threshold"])
            if threshold_override is None
            else threshold_override
        )
        max_labels = (
            int(value.get("max_labels", 6))
            if max_labels_override is None
            else max_labels_override
        )
        max_length = int(value.get("max_length", 256))
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        if max_labels < 1 or max_length < 1:
            raise ValueError("max_labels and max_length must be positive.")
        return cls(
            threshold=threshold,
            max_labels=max_labels,
            max_length=max_length,
            code_description_format=str(
                value.get("code_description_format", "category_subcategory_v1")
            ),
            after_semicolon_prefix=str(
                value.get("after_semicolon_prefix", "") or ""
            ).strip(),
        )


def validate_bge_reranker_v2_m3(path: Path) -> None:
    """Reject artifacts that do not match the trained BGE-M3 classifier."""
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    architectures = value.get("architectures", [])
    if "XLMRobertaForSequenceClassification" not in architectures:
        errors.append("architecture must be XLMRobertaForSequenceClassification")
    for key, expected in _BGE_M3_SIGNATURE.items():
        if value.get(key) != expected:
            errors.append(f"{key} must be {expected!r}")

    id2label = value.get("id2label", {})
    actual_labels = [str(id2label.get(str(index), "")) for index in range(4)]
    if actual_labels != MODEL_CLASS_NAMES:
        errors.append(
            "id2label must be absent, neutral, positive, negative in class order"
        )

    if errors:
        details = "; ".join(errors)
        raise ValueError(
            f"Only a four-class model fine-tuned from {SUPPORTED_MODEL} is "
            f"supported: {details}."
        )


@dataclass(frozen=True)
class Settings:
    classifier_config_path: Path
    model_config_path: Path
    codebook_path: Path
    tokenizer_path: Path
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str | None
    text_column: str
    output_column: str
    sheet_name: str | None
    pair_batch_size: int
    request_timeout_seconds: float
    startup_timeout_seconds: float
    max_upload_bytes: int
    max_uncompressed_xlsx_bytes: int
    max_rows: int
    threshold_override: float | None
    max_labels_override: int | None

    @classmethod
    def from_env(cls) -> "Settings":
        sheet_name = os.getenv("SHEET_NAME", "").strip() or None
        api_key = os.getenv("VLLM_API_KEY", "").strip() or None
        text_column = os.getenv("TEXT_COLUMN", "Ответ").strip()
        output_column = os.getenv("OUTPUT_COLUMN", "Предсказание").strip()
        if not text_column or not output_column:
            raise ValueError("TEXT_COLUMN and OUTPUT_COLUMN must not be empty.")
        if text_column == output_column:
            raise ValueError("TEXT_COLUMN and OUTPUT_COLUMN must be different.")
        vllm_model = os.getenv("VLLM_MODEL", "survey-cross-encoder").strip()
        if not vllm_model:
            raise ValueError("VLLM_MODEL must not be empty.")
        return cls(
            classifier_config_path=Path(
                os.getenv(
                    "CLASSIFIER_CONFIG_PATH", "/artifacts/classifier_config.json"
                )
            ),
            model_config_path=Path(
                os.getenv(
                    "MODEL_CONFIG_PATH",
                    "/artifacts/models/bge-reranker-v2-m3/config.json",
                )
            ),
            codebook_path=Path(
                os.getenv("CODEBOOK_PATH", "/artifacts/codebook.xlsx")
            ),
            tokenizer_path=Path(
                os.getenv(
                    "TOKENIZER_PATH",
                    "/artifacts/models/bge-reranker-v2-m3/tokenizer.json",
                )
            ),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://vllm:8000").rstrip(
                "/"
            ),
            vllm_model=vllm_model,
            vllm_api_key=api_key,
            text_column=text_column,
            output_column=output_column,
            sheet_name=sheet_name,
            pair_batch_size=_positive_int("PAIR_BATCH_SIZE", 512),
            request_timeout_seconds=_positive_float("REQUEST_TIMEOUT_SECONDS", 600),
            startup_timeout_seconds=_positive_float("STARTUP_TIMEOUT_SECONDS", 900),
            max_upload_bytes=_positive_int("MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
            max_uncompressed_xlsx_bytes=_positive_int(
                "MAX_UNCOMPRESSED_XLSX_BYTES", 250 * 1024 * 1024
            ),
            max_rows=_positive_int("MAX_ROWS", 100_000),
            threshold_override=_optional_float("THRESHOLD"),
            max_labels_override=_optional_int("MAX_LABELS"),
        )
