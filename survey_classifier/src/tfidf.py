from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .data_io import UNKNOWN_CODE, load_train_data, parse_codebook
from .split import save_splits, split_train_val_test
from .utils import compact_float, ensure_dir, read_json, utc_now_iso, write_json


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def collapse_multilabel_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "text", "code"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"Training dataframe is missing columns: {sorted(missing)}")
    if long_df.empty:
        return pd.DataFrame(columns=["row_id", "text", "codes"])

    grouped = (
        long_df.groupby("row_id", sort=False)
        .agg(
            text=("text", "first"),
            codes=("code", lambda values: list(dict.fromkeys(map(str, values)))),
        )
        .reset_index()
    )
    return grouped[grouped["text"].astype(str).str.strip().ne("")].reset_index(drop=True)


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
) -> tuple[Any, Any, Any]:
    if train_rows.empty:
        raise ValueError("Cannot train TF-IDF on an empty train split.")
    if not classes:
        raise ValueError("No training codes are available.")
    if classifier_c <= 0:
        raise ValueError("classifier_c must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    label_binarizer = MultiLabelBinarizer(classes=classes)
    label_binarizer.fit([classes])
    y_train = label_binarizer.transform(train_rows["codes"])

    vectorizer = build_tfidf_vectorizer(
        min_df=min_df,
        word_max_features=word_max_features,
        char_max_features=char_max_features,
    )
    x_train = vectorizer.fit_transform(train_rows["text"].astype(str).tolist())
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
    return vectorizer, classifier, label_binarizer


def _predict_probabilities(vectorizer: Any, classifier: Any, texts: Iterable[str]) -> np.ndarray:
    text_list = [str(text) for text in texts]
    if not text_list:
        n_classes = len(getattr(classifier, "classes_", []))
        return np.empty((0, n_classes), dtype=np.float32)
    matrix = vectorizer.transform(text_list)
    probabilities = np.asarray(classifier.predict_proba(matrix), dtype=np.float32)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    return probabilities


def _binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        label_ranking_average_precision_score,
        precision_score,
        recall_score,
    )

    if len(y_true) == 0:
        return {
            "n_rows": 0,
            "threshold": threshold,
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "macro_f1": None,
            "subset_accuracy": None,
            "hamming_loss": None,
            "lrap": None,
            "single_label_top1_accuracy": None,
        }

    predictions = (probabilities >= threshold).astype(np.int8)
    single_label_mask = y_true.sum(axis=1) == 1
    single_label_accuracy: float | None = None
    if single_label_mask.any():
        true_top1 = y_true[single_label_mask].argmax(axis=1)
        predicted_top1 = probabilities[single_label_mask].argmax(axis=1)
        single_label_accuracy = float((true_top1 == predicted_top1).mean())

    return {
        "n_rows": int(len(y_true)),
        "threshold": float(threshold),
        "micro_precision": float(
            precision_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, predictions)),
        "hamming_loss": float(hamming_loss(y_true, predictions)),
        "lrap": float(label_ranking_average_precision_score(y_true, probabilities)),
        "single_label_top1_accuracy": single_label_accuracy,
    }


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "micro_f1",
) -> tuple[float, pd.DataFrame]:
    if metric not in {"micro_f1", "macro_f1"}:
        raise ValueError("threshold metric must be one of: micro_f1, macro_f1.")
    if len(y_true) == 0:
        return 0.5, pd.DataFrame(columns=["threshold", metric])

    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        metrics = _binary_metrics(y_true, probabilities, float(threshold))
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    scores = pd.DataFrame(rows)
    best = scores.sort_values(
        [metric, "threshold"],
        ascending=[False, False],
    ).iloc[0]
    return float(best["threshold"]), scores


def _per_class_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    classes: list[str],
) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support

    if len(y_true) == 0:
        return pd.DataFrame(columns=["code", "support", "precision", "recall", "f1"])
    predictions = (probabilities >= threshold).astype(np.int8)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "code": classes,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ).sort_values(["support", "code"], ascending=[False, True])


def _save_split_predictions(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
    output_path: Path,
) -> None:
    records = rows.copy()
    predicted_codes = [
        [code for code, score in zip(classes, scores, strict=True) if score >= threshold]
        for scores in probabilities
    ]
    records["true_codes"] = records["codes"].apply(lambda values: ", ".join(values))
    records["predicted_codes"] = [", ".join(values) or UNKNOWN_CODE for values in predicted_codes]
    records["top_candidate"] = [
        classes[int(np.argmax(scores))] if len(scores) else UNKNOWN_CODE
        for scores in probabilities
    ]
    records["top_score"] = [
        compact_float(float(np.max(scores))) if len(scores) else 0.0
        for scores in probabilities
    ]
    records.drop(columns=["codes"]).to_csv(output_path, index=False, encoding="utf-8-sig")


def train_tfidf(
    train_xlsx: str | Path,
    codebook_txt: str | Path,
    out_dir: str | Path,
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
) -> Path:
    long_df = load_train_data(train_xlsx, codebook_txt)
    codebook = parse_codebook(codebook_txt)
    return train_tfidf_from_data(
        long_df=long_df,
        codebook=codebook,
        out_dir=out_dir,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        min_df=min_df,
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        classifier_c=classifier_c,
        max_iter=max_iter,
        n_jobs=n_jobs,
        threshold_metric=threshold_metric,
    )


def train_tfidf_from_data(
    long_df: pd.DataFrame,
    codebook: pd.DataFrame,
    out_dir: str | Path,
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
) -> Path:
    output_dir = ensure_dir(out_dir)
    random.seed(seed)
    np.random.seed(seed)

    splits = split_train_val_test(
        long_df,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    save_splits(splits, output_dir / "splits", seed=seed)
    grouped_splits = {name: collapse_multilabel_rows(frame) for name, frame in splits.items()}

    classes = sorted(long_df["code"].astype(str).unique().tolist())
    vectorizer, classifier, label_binarizer = _fit_model(
        train_rows=grouped_splits["train"],
        classes=classes,
        min_df=min_df,
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        classifier_c=classifier_c,
        max_iter=max_iter,
        n_jobs=n_jobs,
        seed=seed,
    )

    split_probabilities: dict[str, np.ndarray] = {}
    split_targets: dict[str, np.ndarray] = {}
    for name, rows in grouped_splits.items():
        split_probabilities[name] = _predict_probabilities(
            vectorizer,
            classifier,
            rows["text"].astype(str).tolist(),
        )
        split_targets[name] = label_binarizer.transform(rows["codes"])

    threshold, threshold_scores = select_threshold(
        split_targets["val"],
        split_probabilities["val"],
        metric=threshold_metric,
    )
    threshold_scores.to_csv(
        output_dir / "threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics: dict[str, dict[str, Any]] = {}
    for name in ("train", "val", "test"):
        metrics[name] = _binary_metrics(
            split_targets[name],
            split_probabilities[name],
            threshold=threshold,
        )
        write_json(metrics[name], output_dir / f"metrics_{name}.json")
        _per_class_metrics(
            split_targets[name],
            split_probabilities[name],
            threshold=threshold,
            classes=classes,
        ).to_csv(
            output_dir / f"metrics_{name}_per_class.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _save_split_predictions(
            grouped_splits[name],
            split_probabilities[name],
            classes=classes,
            threshold=threshold,
            output_path=output_dir / f"predictions_{name}.csv",
        )

    codebook.to_csv(output_dir / "codebook.csv", index=False, encoding="utf-8-sig")
    artifact_path = output_dir / "tfidf_model.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "classes": classes,
            "threshold": threshold,
            "codebook": codebook,
        },
        artifact_path,
    )

    config = {
        "created_at": utc_now_iso(),
        "model_type": "tfidf_logistic_regression",
        "model_path": artifact_path.name,
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
        "n_classes": len(classes),
        "classes": classes,
        "split_rows": {name: int(len(rows)) for name, rows in grouped_splits.items()},
        "metrics": metrics,
    }
    write_json(config, output_dir / "tfidf_config.json")
    return artifact_path


@dataclass
class TfidfSurveyClassifier:
    vectorizer: Any
    classifier: Any
    classes: list[str]
    threshold: float
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
            codebook=artifact["codebook"],
        )

    def predict_batch(
        self,
        texts: Iterable[Any],
        top_k: int = 5,
        threshold: float | None = None,
        margin_threshold: float = 0.05,
    ) -> pd.DataFrame:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        if margin_threshold < 0:
            raise ValueError("margin_threshold must be non-negative.")
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

        selected_threshold = self.threshold if threshold is None else threshold
        probabilities = _predict_probabilities(self.vectorizer, self.classifier, clean_texts)
        codebook_lookup = self.codebook.set_index("code").to_dict(orient="index")
        rows: list[dict[str, Any]] = []

        for text, scores in zip(clean_texts, probabilities, strict=True):
            order = np.argsort(-scores)[: min(top_k, len(scores))]
            candidates = []
            for index in order:
                code = self.classes[int(index)]
                info = codebook_lookup.get(code, {})
                candidates.append(
                    {
                        "code": code,
                        "name": info.get("name", ""),
                        "parent_code": info.get("parent_code", ""),
                        "parent_name": info.get("parent_name", ""),
                        "probability": compact_float(float(scores[int(index)])),
                        "similarity": compact_float(float(scores[int(index)])),
                    }
                )

            top1 = float(scores[int(order[0])]) if len(order) else 0.0
            top2 = float(scores[int(order[1])]) if len(order) > 1 else 0.0
            margin = top1 - top2 if len(order) > 1 else 1.0
            accepted = [
                candidate
                for candidate in candidates
                if candidate["probability"] >= selected_threshold
            ]
            if not text:
                accepted = []

            if accepted:
                predicted_codes = [candidate["code"] for candidate in accepted]
                predicted_names = [candidate["name"] for candidate in accepted]
                parent_codes = list(
                    dict.fromkeys(
                        candidate["parent_code"]
                        for candidate in accepted
                        if candidate["parent_code"]
                    )
                )
            else:
                predicted_codes = [UNKNOWN_CODE]
                predicted_names = [UNKNOWN_CODE]
                parent_codes = []

            rows.append(
                {
                    "text": text,
                    "predicted_codes": predicted_codes,
                    "predicted_names": predicted_names,
                    "parent_codes": parent_codes,
                    "confidence": compact_float(top1),
                    "margin": compact_float(margin),
                    "top_candidates": candidates,
                    "nearest_examples": [],
                    "needs_review": bool(
                        not text
                        or top1 < selected_threshold
                        or margin < margin_threshold
                        or not accepted
                    ),
                }
            )
        return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a TF-IDF multi-label baseline.")
    parser.add_argument("--train-xlsx", required=True, type=Path)
    parser.add_argument("--codebook-txt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-max-features", type=int, default=100_000)
    parser.add_argument("--char-max-features", type=int, default=150_000)
    parser.add_argument("--classifier-c", type=float, default=4.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--threshold-metric",
        choices=["micro_f1", "macro_f1"],
        default="micro_f1",
    )
    args = parser.parse_args(argv)

    artifact_path = train_tfidf(
        train_xlsx=args.train_xlsx,
        codebook_txt=args.codebook_txt,
        out_dir=args.out_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        min_df=args.min_df,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        classifier_c=args.classifier_c,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        threshold_metric=args.threshold_metric,
    )
    config = read_json(Path(args.out_dir) / "tfidf_config.json")
    test_metrics = config["metrics"]["test"]
    print(f"Saved TF-IDF model to {artifact_path}")
    print(
        "Test metrics: "
        f"micro_f1={test_metrics['micro_f1']}, "
        f"macro_f1={test_metrics['macro_f1']}, "
        f"single_label_top1_accuracy={test_metrics['single_label_top1_accuracy']}"
    )


if __name__ == "__main__":
    main()
