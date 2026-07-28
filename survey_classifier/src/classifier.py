from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_io import UNKNOWN_CODE
from .model_input import prepare_model_texts
from .utils import compact_float, cosine_scores, normalize_rows, read_json


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


@dataclass
class SurveyClassifier:
    model: Any
    example_embeddings: np.ndarray
    example_metadata: pd.DataFrame
    subcategory_centroids: np.ndarray
    subcategory_metadata: pd.DataFrame
    parent_centroids: np.ndarray
    parent_metadata: pd.DataFrame
    codebook: pd.DataFrame
    config: dict[str, Any]
    input_prefix: str = ""

    @classmethod
    def load(cls, model_dir: str | Path) -> "SurveyClassifier":
        root = Path(model_dir)
        index_dir = root / "index"
        if not index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")

        config = read_json(index_dir / "index_config.json")
        configured_model_path = Path(config.get("model_path", "model"))
        sentence_model_dir = (
            configured_model_path
            if configured_model_path.is_absolute()
            else root / configured_model_path
        )
        if not sentence_model_dir.exists():
            raise FileNotFoundError(f"SentenceTransformer model directory not found: {sentence_model_dir}")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(sentence_model_dir))
        input_prefix = str(config.get("input_prefix", ""))
        if "input_prefix" in config and hasattr(model, "default_prompt_name"):
            model.default_prompt_name = None
        example_embeddings = normalize_rows(np.load(index_dir / "example_embeddings.npy"))
        subcategory_centroids = normalize_rows(np.load(index_dir / "subcategory_centroids.npy"))
        parent_centroids = normalize_rows(np.load(index_dir / "parent_centroids.npy"))
        if len(subcategory_centroids) == 0:
            raise ValueError("Index does not contain subcategory centroids.")

        return cls(
            model=model,
            example_embeddings=example_embeddings,
            example_metadata=pd.read_csv(index_dir / "example_metadata.csv", dtype=str, keep_default_na=False),
            subcategory_centroids=subcategory_centroids,
            subcategory_metadata=pd.read_csv(
                index_dir / "subcategory_metadata.csv",
                dtype=str,
                keep_default_na=False,
            ),
            parent_centroids=parent_centroids,
            parent_metadata=pd.read_csv(index_dir / "parent_metadata.csv", dtype=str, keep_default_na=False),
            codebook=pd.read_csv(index_dir / "codebook.csv", dtype=str, keep_default_na=False),
            config=config,
            input_prefix=input_prefix,
        )

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            prepare_model_texts(texts, input_prefix=self.input_prefix),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return normalize_rows(np.asarray(embeddings, dtype=np.float32))

    def _nearest_examples(
        self,
        embedding: np.ndarray,
        code: str,
        nearest_k: int,
    ) -> list[dict[str, Any]]:
        if nearest_k < 1:
            return []
        code_values = self.example_metadata["code"].astype(str).to_numpy()
        indices = np.flatnonzero(code_values == code)
        if len(indices) == 0:
            return []
        scores = cosine_scores(embedding, self.example_embeddings[indices])
        order = np.argsort(-scores)[:nearest_k]
        examples: list[dict[str, Any]] = []
        for local_index in order:
            row_index = int(indices[int(local_index)])
            row = self.example_metadata.iloc[row_index]
            examples.append(
                {
                    "row_id": row.get("row_id", ""),
                    "text": row.get("text", ""),
                    "similarity": compact_float(float(scores[int(local_index)])),
                }
            )
        return examples

    def _predict_from_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        top_k: int,
        threshold: float,
        margin_threshold: float,
        nearest_k: int,
    ) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        scores = cosine_scores(embedding, self.subcategory_centroids)
        top_count = min(top_k, len(scores))
        order = np.argsort(-scores)[:top_count]

        top_candidates: list[dict[str, Any]] = []
        top_scores: list[float] = []
        for index in order:
            row = self.subcategory_metadata.iloc[int(index)]
            score = float(scores[int(index)])
            top_scores.append(score)
            top_candidates.append(
                {
                    "code": row.get("code", ""),
                    "name": row.get("code_name", ""),
                    "parent_code": row.get("parent_code", ""),
                    "parent_name": row.get("parent_name", ""),
                    "similarity": compact_float(score),
                }
            )

        top1_similarity = top_scores[0] if top_scores else 0.0
        top2_similarity = top_scores[1] if len(top_scores) > 1 else 0.0
        margin = top1_similarity - top2_similarity if len(top_candidates) > 1 else 1.0
        predicted_candidates = [
            candidate
            for candidate, score in zip(top_candidates, top_scores, strict=True)
            if score >= threshold
        ]

        needs_review = (
            top1_similarity < threshold
            or margin < margin_threshold
            or len(predicted_candidates) == 0
        )

        if predicted_candidates:
            predicted_codes = [candidate["code"] for candidate in predicted_candidates]
            predicted_names = [candidate["name"] for candidate in predicted_candidates]
            parent_codes = [
                code
                for code in dict.fromkeys(candidate["parent_code"] for candidate in predicted_candidates)
                if code
            ]
            nearest_examples = [
                {
                    "code": candidate["code"],
                    "name": candidate["name"],
                    "examples": self._nearest_examples(
                        embedding=embedding,
                        code=candidate["code"],
                        nearest_k=nearest_k,
                    ),
                }
                for candidate in predicted_candidates
            ]
        else:
            predicted_codes = [UNKNOWN_CODE]
            predicted_names = [UNKNOWN_CODE]
            parent_codes = []
            nearest_examples = []

        return {
            "text": text,
            "predicted_codes": predicted_codes,
            "predicted_names": predicted_names,
            "parent_codes": parent_codes,
            "confidence": compact_float(top1_similarity),
            "margin": compact_float(margin),
            "top_candidates": top_candidates,
            "nearest_examples": nearest_examples,
            "needs_review": bool(needs_review),
        }

    def predict_one(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.65,
        margin_threshold: float = 0.05,
        nearest_k: int = 2,
    ) -> dict[str, Any]:
        clean = _clean_text(text)
        embedding = self._encode([clean])[0]
        return self._predict_from_embedding(
            text=clean,
            embedding=embedding,
            top_k=top_k,
            threshold=threshold,
            margin_threshold=margin_threshold,
            nearest_k=nearest_k,
        )

    def predict_batch(
        self,
        texts: Iterable[str],
        top_k: int = 5,
        threshold: float = 0.65,
        margin_threshold: float = 0.05,
        nearest_k: int = 2,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        clean_texts = [_clean_text(text) for text in texts]
        if not clean_texts:
            return pd.DataFrame(
                columns=[
                    "text",
                    "predicted_codes",
                    "predicted_names",
                    "parent_codes",
                    "confidence",
                    "margin",
                    "top_candidates",
                    "nearest_examples",
                    "needs_review",
                ]
            )
        embeddings = self._encode(clean_texts, batch_size=batch_size)
        rows = [
            self._predict_from_embedding(
                text=text,
                embedding=embedding,
                top_k=top_k,
                threshold=threshold,
                margin_threshold=margin_threshold,
                nearest_k=nearest_k,
            )
            for text, embedding in zip(clean_texts, embeddings, strict=True)
        ]
        return pd.DataFrame(rows)
