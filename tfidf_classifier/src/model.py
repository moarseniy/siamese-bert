from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .data_io import (
    UNKNOWN_CODE,
    clean_text,
    load_labeled_data,
    save_splits,
    split_train_val_test,
)
from .metrics import (
    calculate_metrics,
    encode_labels,
    select_threshold,
    threshold_predictions,
)
from .utils import compact_float, ensure_dir, utc_now_iso, write_json


def build_tfidf_vectorizer(
    min_df: int = 2,
    word_max_features: int | None = 100_000,
    char_max_features: int | None = 150_000,
) -> Any:
    if min_df < 1:
        raise ValueError("min_df must be positive.")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    min_df=min_df,
                    max_features=word_max_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    min_df=min_df,
                    max_features=char_max_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def _fit_model(
    train_rows: pd.DataFrame,
    classes: list[str],
    min_df: int,
    word_max_features: int | None,
    char_max_features: int | None,
    classifier_c: float,
    max_iter: int,
    n_jobs: int | None,
    seed: int,
) -> tuple[Any, Any]:
    if train_rows.empty:
        raise ValueError("Cannot train TF-IDF on an empty train split.")
    if not classes:
        raise ValueError("No training codes are available.")
    if classifier_c <= 0 or max_iter < 1:
        raise ValueError("classifier_c and max_iter must be positive.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    vectorizer = build_tfidf_vectorizer(
        min_df=min_df,
        word_max_features=word_max_features,
        char_max_features=char_max_features,
    )
    x_train = vectorizer.fit_transform(train_rows["text"].astype(str).tolist())
    y_train = encode_labels(train_rows["codes"].tolist(), classes)
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=classifier_c,
            class_weight="balanced",
            max_iter=max_iter,
            solver="liblinear",
            random_state=seed,
        ),
        n_jobs=n_jobs,
    )
    classifier.fit(x_train, y_train)
    return vectorizer, classifier


def predict_probabilities(
    vectorizer: Any,
    classifier: Any,
    texts: Iterable[Any],
    n_classes: int,
) -> np.ndarray:
    clean_texts = [clean_text(text) for text in texts]
    if not clean_texts:
        return np.empty((0, n_classes), dtype=np.float32)
    matrix = vectorizer.transform(clean_texts)
    probabilities = np.asarray(classifier.predict_proba(matrix), dtype=np.float32)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    probabilities[np.array([not text for text in clean_texts])] = 0.0
    return probabilities


def _prediction_frame(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
    max_labels: int,
) -> pd.DataFrame:
    result = rows[["row_id", "answer", "context"]].reset_index(drop=True).copy()
    y_true = encode_labels(rows["codes"].tolist(), classes)
    y_pred = threshold_predictions(probabilities, threshold, max_labels)
    result["true_codes"] = [
        ", ".join(classes[index] for index in np.flatnonzero(values))
        for values in y_true
    ]
    result["predicted_codes"] = [
        ", ".join(classes[index] for index in np.flatnonzero(values)) or UNKNOWN_CODE
        for values in y_pred
    ]
    result["top_candidates"] = [
        "; ".join(
            f"{classes[index]}:{scores[index]:.4f}"
            for index in np.argsort(-scores)[: min(5, len(classes))]
        )
        for scores in probabilities
    ]
    result["correct"] = result["true_codes"] == result["predicted_codes"]
    return result


def _save_label_distribution(
    splits: dict[str, pd.DataFrame],
    classes: list[str],
    codebook: pd.DataFrame,
    path: Path,
) -> None:
    names = codebook.set_index("code")["name"].astype(str).to_dict()
    records = []
    for code in classes:
        record: dict[str, Any] = {"code": code, "name": names.get(code, code)}
        for split_name, frame in splits.items():
            record[split_name] = int(
                frame["codes"].apply(lambda values: code in values).sum()
            )
        record["total"] = sum(record[name] for name in splits)
        records.append(record)
    pd.DataFrame(records).sort_values(
        ["train", "code"],
        ascending=[True, True],
    ).to_csv(path, index=False, encoding="utf-8-sig")


def train_tfidf(
    train_path: str | Path,
    codebook_path: str | Path,
    output_dir: str | Path,
    text_col: str = "Ответ",
    codes_col: str = "Коды_новые",
    context_col: str | None = None,
    csv_sep: str | None = None,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    min_df: int = 2,
    word_max_features: int | None = 100_000,
    char_max_features: int | None = 150_000,
    classifier_c: float = 4.0,
    max_iter: int = 1000,
    n_jobs: int | None = -1,
    threshold_metric: str = "micro_f1",
    max_labels: int = 6,
) -> Path:
    if max_labels < 1:
        raise ValueError("max_labels must be positive.")
    random.seed(seed)
    np.random.seed(seed)
    output_dir = ensure_dir(output_dir)

    data, codebook = load_labeled_data(
        train_path,
        codebook_path,
        text_col=text_col,
        codes_col=codes_col,
        context_col=context_col,
        csv_sep=csv_sep,
    )
    classes = sorted({code for values in data["codes"] for code in values})
    splits = split_train_val_test(data, val_size=val_size, test_size=test_size, seed=seed)
    save_splits(splits, output_dir / "splits")
    codebook.to_csv(output_dir / "codebook.csv", index=False, encoding="utf-8-sig")
    _save_label_distribution(
        splits,
        classes,
        codebook,
        output_dir / "label_distribution.csv",
    )

    vectorizer, classifier = _fit_model(
        train_rows=splits["train"],
        classes=classes,
        min_df=min_df,
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        classifier_c=classifier_c,
        max_iter=max_iter,
        n_jobs=n_jobs,
        seed=seed,
    )
    probabilities = {
        name: predict_probabilities(
            vectorizer,
            classifier,
            frame["text"].tolist(),
            len(classes),
        )
        for name, frame in splits.items()
    }
    targets = {
        name: encode_labels(frame["codes"].tolist(), classes)
        for name, frame in splits.items()
    }
    threshold, threshold_search = select_threshold(
        targets["val"],
        probabilities["val"],
        classes,
        metric=threshold_metric,
        max_labels=max_labels,
    )
    threshold_search.to_csv(
        output_dir / "threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        metrics, per_class = calculate_metrics(
            targets[split_name],
            probabilities[split_name],
            classes,
            threshold=threshold,
            max_labels=max_labels,
        )
        all_metrics[split_name] = metrics
        write_json(metrics, output_dir / f"{split_name}_metrics.json")
        per_class.to_csv(
            output_dir / f"{split_name}_per_class.csv",
            index=False,
            encoding="utf-8-sig",
        )
        prediction_frame = _prediction_frame(
            splits[split_name],
            probabilities[split_name],
            classes,
            threshold,
            max_labels,
        )
        prediction_frame.to_csv(
            output_dir / f"{split_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        prediction_frame[~prediction_frame["correct"]].to_csv(
            output_dir / f"{split_name}_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

    artifact_path = output_dir / "tfidf_model.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "classes": classes,
            "threshold": threshold,
            "max_labels": max_labels,
            "codebook": codebook,
        },
        artifact_path,
    )
    feature_count = sum(
        len(transformer.vocabulary_)
        for _, transformer in vectorizer.transformer_list
    )
    config = {
        "created_at": utc_now_iso(),
        "model_type": "tfidf_logistic_regression",
        "model_path": artifact_path.name,
        "text_col": text_col,
        "codes_col": codes_col,
        "context_col": context_col,
        "seed": seed,
        "val_size": val_size,
        "test_size": test_size,
        "min_df": min_df,
        "word_ngram_range": [1, 2],
        "char_ngram_range": [3, 5],
        "word_max_features": word_max_features,
        "char_max_features": char_max_features,
        "classifier_c": classifier_c,
        "max_iter": max_iter,
        "n_jobs": n_jobs,
        "threshold_metric": threshold_metric,
        "threshold": threshold,
        "max_labels": max_labels,
        "n_features": feature_count,
        "n_classes": len(classes),
        "classes": classes,
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "metrics": all_metrics,
    }
    write_json(config, output_dir / "tfidf_config.json")
    write_json(all_metrics, output_dir / "metrics.json")
    return artifact_path


@dataclass
class TfidfSurveyClassifier:
    vectorizer: Any
    classifier: Any
    classes: list[str]
    threshold: float
    max_labels: int
    codebook: pd.DataFrame

    @classmethod
    def load(cls, model_dir: str | Path) -> "TfidfSurveyClassifier":
        path = Path(model_dir)
        artifact_path = path / "tfidf_model.joblib" if path.is_dir() else path
        if not artifact_path.exists():
            raise FileNotFoundError(f"TF-IDF model not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        return cls(
            vectorizer=artifact["vectorizer"],
            classifier=artifact["classifier"],
            classes=list(map(str, artifact["classes"])),
            threshold=float(artifact["threshold"]),
            max_labels=int(artifact.get("max_labels", 6)),
            codebook=artifact["codebook"],
        )

    def predict_probabilities(self, texts: Iterable[Any]) -> np.ndarray:
        return predict_probabilities(
            self.vectorizer,
            self.classifier,
            texts,
            len(self.classes),
        )

    def predict_batch(
        self,
        texts: Iterable[Any],
        top_k: int = 5,
        threshold: float | None = None,
        max_labels: int | None = None,
        margin_threshold: float = 0.05,
        probabilities: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        if max_labels is not None and max_labels < 1:
            raise ValueError("max_labels must be positive.")
        if margin_threshold < 0:
            raise ValueError("margin_threshold must be non-negative.")

        columns = [
            "text",
            "predicted_codes",
            "predicted_names",
            "predicted_parent_codes",
            "predicted_parent_names",
            "confidence",
            "margin",
            "top_candidates",
            "needs_review",
        ]
        clean_texts = [clean_text(text) for text in texts]
        if not clean_texts:
            return pd.DataFrame(columns=columns)

        selected_threshold = self.threshold if threshold is None else threshold
        selected_max_labels = self.max_labels if max_labels is None else max_labels
        if probabilities is None:
            probabilities = self.predict_probabilities(clean_texts)
        elif probabilities.shape != (len(clean_texts), len(self.classes)):
            raise ValueError("Probability matrix does not match texts and classes.")
        codebook_lookup = self.codebook.set_index("code").to_dict(orient="index")
        rows = []
        for text, scores in zip(clean_texts, probabilities, strict=True):
            ranked = np.argsort(-scores)
            accepted_indices = [
                int(index)
                for index in ranked
                if scores[index] >= selected_threshold
            ][:selected_max_labels]
            if not text:
                accepted_indices = []
            codes = [self.classes[index] for index in accepted_indices]
            names = [
                str(codebook_lookup.get(code, {}).get("name", code))
                for code in codes
            ]
            parent_codes: list[str] = []
            parent_names: list[str] = []
            for code in codes:
                info = codebook_lookup.get(code, {})
                parent = str(info.get("parent_code", "") or "")
                if parent and parent not in parent_codes:
                    parent_codes.append(parent)
                    parent_names.append(
                        str(
                            info.get("parent_name", "")
                            or codebook_lookup.get(parent, {}).get("name", parent)
                        )
                    )
            candidates = []
            for index in ranked[: min(top_k, len(ranked))]:
                code = self.classes[int(index)]
                info = codebook_lookup.get(code, {})
                candidates.append(
                    {
                        "code": code,
                        "name": str(info.get("name", "")),
                        "parent_code": str(info.get("parent_code", "") or ""),
                        "probability": compact_float(float(scores[index])),
                    }
                )
            top1 = float(scores[ranked[0]]) if len(ranked) else 0.0
            top2 = float(scores[ranked[1]]) if len(ranked) > 1 else 0.0
            margin = top1 - top2 if len(ranked) > 1 else top1
            rows.append(
                {
                    "text": text,
                    "predicted_codes": codes or [UNKNOWN_CODE],
                    "predicted_names": names or [UNKNOWN_CODE],
                    "predicted_parent_codes": parent_codes,
                    "predicted_parent_names": parent_names,
                    "confidence": compact_float(top1),
                    "margin": compact_float(margin),
                    "top_candidates": candidates,
                    "needs_review": bool(
                        not text
                        or not codes
                        or top1 < selected_threshold
                        or margin < margin_threshold
                    ),
                }
            )
        return pd.DataFrame(rows, columns=columns)
