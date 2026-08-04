from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data_io import (
    SENTIMENT_BY_MODEL_CLASS,
    SENTIMENT_NAMES,
    UNKNOWN_CODE,
    clean_text,
    combine_text,
    leaf_codebook,
    parse_codebook,
)
from .metrics import decode_response_predictions


class CrossEncoderSurveyClassifier:
    """Score every answer/leaf-code pair with one four-class transformer."""

    def __init__(
        self,
        artifact_dir: str | Path,
        codebook_path: str | Path | None = None,
        device: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        config_path = self.artifact_dir / "classifier_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Classifier config not found: {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.threshold = float(self.config["threshold"])
        self.max_labels = int(self.config.get("max_labels", 6))
        self.max_length = int(self.config.get("max_length", 256))

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
            model_dir, trust_remote_code=remote_code
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, trust_remote_code=remote_code
        ).to(self.device)
        self.model.eval()

        if codebook_path is None:
            saved_path = self.artifact_dir / "codebook.csv"
            codebook = pd.read_csv(saved_path, encoding="utf-8-sig")
            codebook["code"] = codebook["code"].astype(str)
            if not pd.api.types.is_bool_dtype(codebook["is_parent"]):
                codebook["is_parent"] = codebook["is_parent"].apply(
                    lambda value: str(value).strip().casefold() in {"true", "1"}
                )
        else:
            codebook = parse_codebook(codebook_path)
        self._set_codebook(codebook)

    def _set_codebook(self, codebook: pd.DataFrame) -> None:
        leaves = leaf_codebook(codebook)
        self.codebook = codebook.copy()
        self.codes = leaves["code"].astype(str).tolist()
        self.descriptions = leaves["description"].astype(str).tolist()
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
            return np.empty((0, len(self.codes), 4), dtype=np.float32)

        texts = [
            combine_text(answer, contexts[index] if contexts is not None else None)
            for index, answer in enumerate(answers)
        ]
        total_pairs = len(texts) * len(self.codes)
        flat_probabilities = np.empty((total_pairs, 4), dtype=np.float32)
        starts = range(0, total_pairs, batch_size)
        iterator = tqdm(
            starts,
            total=(total_pairs + batch_size - 1) // batch_size,
            desc="Classifying pairs",
            unit="batch",
            disable=not show_progress,
        )
        with self.torch.inference_mode():
            for start in iterator:
                stop = min(start + batch_size, total_pairs)
                batch_answers: list[str] = []
                batch_descriptions: list[str] = []
                for flat_index in range(start, stop):
                    answer_index, code_index = divmod(flat_index, len(self.codes))
                    batch_answers.append(texts[answer_index])
                    batch_descriptions.append(self.descriptions[code_index])
                encoded = self.tokenizer(
                    batch_answers,
                    text_pair=batch_descriptions,
                    padding=True,
                    truncation="longest_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                flat_probabilities[start:stop] = (
                    self.torch.softmax(logits, dim=1).float().cpu().numpy()
                )
        probabilities = flat_probabilities.reshape(len(texts), len(self.codes), 4)
        empty_mask = np.array([not clean_text(answer) for answer in answers])
        probabilities[empty_mask] = 0.0
        probabilities[empty_mask, :, 0] = 1.0
        return probabilities

    def format_predictions(
        self,
        probabilities: np.ndarray,
        threshold: float | None = None,
        max_labels: int | None = None,
        top_k: int = 5,
        margin_threshold: float = 0.05,
    ) -> pd.DataFrame:
        chosen_threshold = self.threshold if threshold is None else float(threshold)
        chosen_max_labels = self.max_labels if max_labels is None else int(max_labels)
        if top_k < 1 or margin_threshold < 0:
            raise ValueError(
                "top_k must be positive and margin_threshold non-negative."
            )
        if probabilities.ndim != 3 or probabilities.shape[1:] != (len(self.codes), 4):
            raise ValueError("Probability tensor does not match the active codebook.")

        predicted = decode_response_predictions(
            probabilities, chosen_threshold, chosen_max_labels
        )
        columns = [
            "predicted_codes",
            "predicted_names",
            "predicted_parent_codes",
            "predicted_parent_names",
            "predicted_sentiments",
            "predicted_code_sentiments",
            "predicted_sentiment_names",
            "confidence",
            "margin",
            "top_candidates",
            "needs_review",
        ]
        rows: list[dict[str, Any]] = []
        for row_index, response_predictions in enumerate(predicted):
            presence = 1.0 - probabilities[row_index, :, 0]
            ranked = np.argsort(-presence)
            accepted = [
                int(index) for index in ranked if response_predictions[index] > 0
            ]
            codes = [self.codes[index] for index in accepted]
            sentiments = [
                SENTIMENT_BY_MODEL_CLASS[int(response_predictions[index])]
                for index in accepted
            ]
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
            top_score = float(presence[ranked[0]]) if len(ranked) else 0.0
            second_score = float(presence[ranked[1]]) if len(ranked) > 1 else 0.0
            margin = top_score - second_score
            top_candidates = []
            for index in ranked[: min(top_k, len(ranked))]:
                raw_sentiment = int(probabilities[row_index, index, 1:].argmax())
                top_candidates.append(
                    f"{self.codes[index]}:{presence[index]:.4f}:"
                    f"{SENTIMENT_NAMES[raw_sentiment]}"
                )
            rows.append(
                {
                    "predicted_codes": ", ".join(codes) or UNKNOWN_CODE,
                    "predicted_names": (
                        "; ".join(self.name_by_code.get(code, code) for code in codes)
                        or UNKNOWN_CODE
                    ),
                    "predicted_parent_codes": ", ".join(parent_codes),
                    "predicted_parent_names": "; ".join(parent_names),
                    "predicted_sentiments": ", ".join(map(str, sentiments)),
                    "predicted_code_sentiments": (
                        ", ".join(
                            f"{code}:{sentiment}"
                            for code, sentiment in zip(codes, sentiments)
                        )
                        or UNKNOWN_CODE
                    ),
                    "predicted_sentiment_names": "; ".join(
                        f"{code}:{SENTIMENT_NAMES[sentiment]}"
                        for code, sentiment in zip(codes, sentiments)
                    ),
                    "confidence": top_score,
                    "margin": margin,
                    "top_candidates": "; ".join(top_candidates),
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
