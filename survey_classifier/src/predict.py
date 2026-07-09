from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .classifier import SurveyClassifier
from .data_io import TEXT_COL_DEFAULT


def _join_values(values: Any) -> str:
    if not isinstance(values, list):
        return "" if values is None else str(values)
    return ", ".join(str(value) for value in values if str(value))


def _shorten(text: str, limit: int = 120) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def format_top_candidates(candidates: Any) -> str:
    if not isinstance(candidates, list):
        return ""
    parts = []
    for candidate in candidates:
        code = candidate.get("code", "")
        score = candidate.get("similarity", "")
        if code:
            parts.append(f"{code}:{score:.4f}" if isinstance(score, float) else f"{code}:{score}")
    return "; ".join(parts)


def format_nearest_examples(nearest_examples: Any) -> str:
    if not isinstance(nearest_examples, list):
        return ""
    groups: list[str] = []
    for group in nearest_examples:
        code = group.get("code", "")
        examples = group.get("examples", [])
        texts = [f'"{_shorten(example.get("text", ""))}"' for example in examples if example.get("text")]
        if code and texts:
            groups.append(f"{code} -> " + " | ".join(texts))
    return "; ".join(groups)


def _require_text_column(frame: pd.DataFrame, text_col: str, source: Path) -> None:
    if text_col not in frame.columns:
        existing = ", ".join(map(str, frame.columns.tolist()))
        raise ValueError(f"Missing text column {text_col!r} in {source}. Existing columns: {existing}")


def predict_excel(
    model_dir: str | Path,
    input_xlsx: str | Path,
    output_xlsx: str | Path,
    text_col: str = TEXT_COL_DEFAULT,
    top_k: int = 5,
    threshold: float = 0.65,
    margin_threshold: float = 0.05,
    nearest_k: int = 2,
    batch_size: int = 64,
) -> Path:
    input_path = Path(input_xlsx)
    output_path = Path(output_xlsx)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_path}")

    source = pd.read_excel(input_path, engine="openpyxl")
    _require_text_column(source, text_col, input_path)

    classifier = SurveyClassifier.load(model_dir)
    predictions = classifier.predict_batch(
        source[text_col].tolist(),
        top_k=top_k,
        threshold=threshold,
        margin_threshold=margin_threshold,
        nearest_k=nearest_k,
        batch_size=batch_size,
    )

    result = source.copy()
    result["predicted_codes"] = predictions["predicted_codes"].apply(_join_values)
    result["predicted_names"] = predictions["predicted_names"].apply(_join_values)
    result["parent_codes"] = predictions["parent_codes"].apply(_join_values)
    result["confidence"] = predictions["confidence"]
    result["needs_review"] = predictions["needs_review"]
    result["top_candidates"] = predictions["top_candidates"].apply(format_top_candidates)
    result["nearest_examples"] = predictions["nearest_examples"].apply(format_nearest_examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False, engine="openpyxl")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classify survey answers from an Excel file.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input-xlsx", required=True, type=Path)
    parser.add_argument("--output-xlsx", required=True, type=Path)
    parser.add_argument("--text-col", default=TEXT_COL_DEFAULT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--nearest-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    output_path = predict_excel(
        model_dir=args.model_dir,
        input_xlsx=args.input_xlsx,
        output_xlsx=args.output_xlsx,
        text_col=args.text_col,
        top_k=args.top_k,
        threshold=args.threshold,
        margin_threshold=args.margin_threshold,
        nearest_k=args.nearest_k,
        batch_size=args.batch_size,
    )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
