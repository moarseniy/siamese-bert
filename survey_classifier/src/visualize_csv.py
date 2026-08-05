from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import (
    CODES_COL_DEFAULT,
    TEXT_COL_DEFAULT,
    UNKNOWN_CODE,
    parent_code,
    parse_codebook,
    split_codes,
)
from .model_input import configure_model_input, prepare_model_texts
from .utils import ensure_dir
from .visualize import build_projection_from_embeddings, save_projection_html


def load_single_label_csv(
    input_csv: str | Path,
    text_col: str = TEXT_COL_DEFAULT,
    codes_col: str = CODES_COL_DEFAULT,
    codebook_path: str | Path | None = None,
    encoding: str = "utf-8-sig",
    sep: str | None = None,
) -> pd.DataFrame:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")

    source = pd.read_csv(
        input_path,
        encoding=encoding,
        sep=sep,
        engine="python" if sep is None else "c",
        dtype=str,
        keep_default_na=False,
    )
    missing = [column for column in (text_col, codes_col) if column not in source.columns]
    if missing:
        existing = ", ".join(map(str, source.columns))
        raise ValueError(f"Missing columns in {input_path}: {missing}. Existing columns: {existing}")

    codebook_by_code: dict[str, dict[str, object]] = {}
    if codebook_path is not None:
        codebook = parse_codebook(codebook_path)
        codebook_by_code = codebook.set_index("code").to_dict(orient="index")

    records: list[dict[str, object]] = []
    for row_id, row in source.iterrows():
        text = str(row[text_col]).strip()
        codes = [code for code in split_codes(row[codes_col]) if code != UNKNOWN_CODE]
        if not text or len(codes) != 1:
            continue

        code = codes[0]
        codebook_row = codebook_by_code.get(code, {})
        inferred_parent = parent_code(code)
        records.append(
            {
                "row_id": int(row_id),
                "text": text,
                "codes": code,
                "code": code,
                "code_name": str(codebook_row.get("name", "")),
                "parent_code": str(codebook_row.get("parent_code", inferred_parent)),
                "parent_name": str(codebook_row.get("parent_name", "")),
                "point_type": "example",
                "_point_radius": 4,
            }
        )

    if len(records) < 2:
        raise ValueError(
            "At least two non-empty rows with exactly one non-UNKNOWN label are required."
        )
    return pd.DataFrame(records)


def encode_texts(
    texts: list[str],
    model_path: str | Path,
    batch_size: int = 32,
    prompt_name: str | None = None,
    input_prefix: str | None = None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path))
    resolved_input_prefix = configure_model_input(
        model=model,
        prompt_name=prompt_name,
        input_prefix=input_prefix,
    )
    embeddings = model.encode(
        prepare_model_texts(texts, input_prefix=resolved_input_prefix),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def visualize_csv(
    input_csv: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    text_col: str = TEXT_COL_DEFAULT,
    codes_col: str = CODES_COL_DEFAULT,
    codebook_path: str | Path | None = None,
    method: str = "pca",
    color_by: str = "code",
    sample_size: int | None = 5000,
    seed: int = 42,
    batch_size: int = 32,
    encoding: str = "utf-8-sig",
    sep: str | None = None,
    output_prefix: str | None = None,
    prompt_name: str | None = None,
    input_prefix: str | None = None,
) -> tuple[Path, Path]:
    metadata = load_single_label_csv(
        input_csv=input_csv,
        text_col=text_col,
        codes_col=codes_col,
        codebook_path=codebook_path,
        encoding=encoding,
        sep=sep,
    )
    embeddings = encode_texts(
        metadata["text"].astype(str).tolist(),
        model_path=model_path,
        batch_size=batch_size,
        prompt_name=prompt_name,
        input_prefix=input_prefix,
    )
    projection = build_projection_from_embeddings(
        embeddings=embeddings,
        metadata=metadata,
        method=method,
        sample_size=sample_size,
        seed=seed,
    )

    output_path = ensure_dir(output_dir)
    prefix = output_prefix or f"csv_single_label_{method}"
    csv_path = output_path / f"{prefix}.csv"
    html_path = output_path / f"{prefix}.html"
    projection.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_projection_html(
        projection=projection,
        output_html=html_path,
        color_by=color_by,
        title=f"{method.upper()} projection of single-label CSV examples by {color_by}",
    )
    return csv_path, html_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize single-label rows from a CSV using a local base "
            "Sentence Transformers model."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("visualization"))
    parser.add_argument("--text-col", default=TEXT_COL_DEFAULT)
    parser.add_argument("--codes-col", default=CODES_COL_DEFAULT)
    parser.add_argument("--codebook", type=Path, default=None, help="Codebook CSV.")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--color-by", choices=["code", "parent_code"], default="code")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-name", default=None)
    prompt_group.add_argument("--input-prefix", default=None)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument(
        "--csv-sep",
        default=None,
        help="CSV delimiter. By default it is detected automatically.",
    )
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args(argv)

    csv_path, html_path = visualize_csv(
        input_csv=args.input,
        model_path=args.model_dir,
        output_dir=args.output_dir,
        text_col=args.text_col,
        codes_col=args.codes_col,
        codebook_path=args.codebook,
        method=args.method,
        color_by=args.color_by,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        encoding=args.encoding,
        sep=args.csv_sep,
        output_prefix=args.output_prefix,
        prompt_name=args.prompt_name,
        input_prefix=args.input_prefix,
    )
    print(f"Saved projection CSV to {csv_path}")
    print(f"Saved projection HTML to {html_path}")


if __name__ == "__main__":
    main()
