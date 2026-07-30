from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data_io import UNKNOWN_CODE, clean_text, combine_text


class BertSurveyClassifier:
    """Loads a trained artifact and predicts one or more survey codes."""

    def __init__(
        self,
        artifact_dir: str | Path,
        device: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        config_path = self.artifact_dir / "classifier_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Classifier config not found: {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.classes = [str(code) for code in self.config["classes"]]
        self.threshold = float(self.config["threshold"])
        self.max_labels = int(self.config.get("max_labels", 6))
        self.max_length = int(self.config.get("max_length", 128))

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install inference dependencies first: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        remote_code = (
            bool(self.config.get("trust_remote_code", False))
            if trust_remote_code is None
            else trust_remote_code
        )
        model_dir = self.artifact_dir / self.config.get("model_subdir", "model")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            trust_remote_code=remote_code,
        ).to(self.device)
        self.model.eval()

        codebook_path = self.artifact_dir / "codebook.csv"
        codebook = pd.read_csv(codebook_path, encoding="utf-8-sig")
        codebook["code"] = codebook["code"].astype(str)
        self.name_by_code = codebook.set_index("code")["name"].astype(str).to_dict()
        self.parent_by_code = (
            codebook.set_index("code")["parent_code"].fillna("").astype(str).to_dict()
        )
        self.parent_name_by_code = (
            codebook.set_index("code")["parent_name"].fillna("").astype(str).to_dict()
        )

    def predict_probabilities(
        self,
        answers: Sequence[Any],
        contexts: Sequence[Any] | None = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if contexts is not None and len(contexts) != len(answers):
            raise ValueError("answers and contexts must have equal lengths.")
        if len(answers) == 0:
            return np.empty((0, len(self.classes)), dtype=np.float32)

        texts = [
            combine_text(answer, contexts[index] if contexts is not None else None)
            for index, answer in enumerate(answers)
        ]
        probabilities: list[np.ndarray] = []
        starts = range(0, len(texts), batch_size)
        iterator = tqdm(
            starts,
            total=math_ceil_div(len(texts), batch_size),
            desc="Classifying",
            unit="batch",
            disable=not show_progress,
        )
        with self.torch.inference_mode():
            for start in iterator:
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                probabilities.append(
                    self.torch.sigmoid(logits).float().cpu().numpy()
                )
        result = np.concatenate(probabilities)
        empty_mask = np.array([not clean_text(answer) for answer in answers])
        result[empty_mask] = 0.0
        return result

    def format_predictions(
        self,
        probabilities: np.ndarray,
        threshold: float | None = None,
        max_labels: int | None = None,
        top_k: int = 5,
        margin_threshold: float = 0.05,
    ) -> pd.DataFrame:
        chosen_threshold = self.threshold if threshold is None else threshold
        chosen_max_labels = self.max_labels if max_labels is None else max_labels
        if not 0 <= chosen_threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        if chosen_max_labels < 1 or top_k < 1:
            raise ValueError("max_labels and top_k must be positive.")
        if margin_threshold < 0:
            raise ValueError("margin_threshold must be non-negative.")
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes):
            raise ValueError("Probability matrix does not match classifier classes.")

        columns = [
            "predicted_codes",
            "predicted_names",
            "predicted_parent_codes",
            "predicted_parent_names",
            "confidence",
            "margin",
            "top_candidates",
            "needs_review",
        ]
        rows: list[dict[str, Any]] = []
        for scores in probabilities:
            ranked = np.argsort(-scores)
            accepted = [
                int(index)
                for index in ranked
                if scores[index] >= chosen_threshold
            ][:chosen_max_labels]
            codes = [self.classes[index] for index in accepted]
            parent_codes: list[str] = []
            parent_names: list[str] = []
            for code in codes:
                parent = self.parent_by_code.get(code, "")
                if parent and parent not in parent_codes:
                    parent_codes.append(parent)
                    parent_names.append(
                        self.parent_name_by_code.get(code, "")
                        or self.name_by_code.get(parent, parent)
                    )
            top_score = float(scores[ranked[0]]) if len(ranked) else 0.0
            second_score = float(scores[ranked[1]]) if len(ranked) > 1 else 0.0
            margin = top_score - second_score
            rows.append(
                {
                    "predicted_codes": ", ".join(codes) or UNKNOWN_CODE,
                    "predicted_names": (
                        "; ".join(self.name_by_code.get(code, code) for code in codes)
                        or UNKNOWN_CODE
                    ),
                    "predicted_parent_codes": ", ".join(parent_codes),
                    "predicted_parent_names": "; ".join(parent_names),
                    "confidence": top_score,
                    "margin": margin,
                    "top_candidates": "; ".join(
                        f"{self.classes[index]}:{scores[index]:.4f}"
                        for index in ranked[: min(top_k, len(ranked))]
                    ),
                    "needs_review": not codes or margin < margin_threshold,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def predict_batch(
        self,
        answers: Sequence[Any],
        contexts: Sequence[Any] | None = None,
        batch_size: int = 64,
        threshold: float | None = None,
        max_labels: int | None = None,
        top_k: int = 5,
        margin_threshold: float = 0.05,
        show_progress: bool = True,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        probabilities = self.predict_probabilities(
            answers,
            contexts=contexts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return (
            self.format_predictions(
                probabilities,
                threshold=threshold,
                max_labels=max_labels,
                top_k=top_k,
                margin_threshold=margin_threshold,
            ),
            probabilities,
        )


def math_ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
