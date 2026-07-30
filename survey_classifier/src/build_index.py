from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import load_train_data, parse_codebook
from .model_input import configure_model_input, prepare_model_texts
from .utils import ensure_dir, normalize_rows, read_json, utc_now_iso, write_json


def _model_path_for_config(model_dir: Path, out_dir: Path) -> str:
    try:
        return os.path.relpath(model_dir.resolve(), out_dir.resolve())
    except ValueError:
        return str(model_dir.resolve())


def _encode_texts(
    model_dir: Path,
    texts: list[str],
    batch_size: int = 64,
    show_progress_bar: bool = True,
    prompt_name: str | None = None,
    input_prefix: str | None = None,
) -> tuple[np.ndarray, str]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir))
    resolved_input_prefix = configure_model_input(
        model=model,
        prompt_name=prompt_name,
        input_prefix=input_prefix,
    )
    embeddings = model.encode(
        prepare_model_texts(texts, input_prefix=resolved_input_prefix),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return normalize_rows(np.asarray(embeddings, dtype=np.float32)), resolved_input_prefix


def _saved_training_input_config(output_dir: Path) -> tuple[str | None, str | None]:
    config_path = output_dir / "train_config.json"
    if not config_path.exists():
        return None, None
    config = read_json(config_path)
    if "input_prefix" not in config:
        return None, None
    return config.get("prompt_name"), str(config.get("input_prefix", ""))


def build_index(
    train_df: pd.DataFrame,
    codebook_df: pd.DataFrame,
    out_dir: str | Path,
    model_dir: str | Path | None = None,
    batch_size: int = 64,
    prompt_name: str | None = None,
    input_prefix: str | None = None,
    prompt_name_for_config: str | None = None,
    show_progress_bar: bool = True,
) -> Path:
    required = {"row_id", "text", "codes", "code", "code_name", "parent_code", "parent_name"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Training dataframe is missing columns: {sorted(missing)}")
    if train_df.empty:
        raise ValueError("Cannot build index from an empty training dataframe.")

    output_dir = ensure_dir(out_dir)
    resolved_model_dir = Path(model_dir) if model_dir is not None else output_dir / "model"
    if not resolved_model_dir.exists():
        raise FileNotFoundError(f"SentenceTransformer model directory not found: {resolved_model_dir}")

    configured_prompt_name = prompt_name_for_config or prompt_name
    if prompt_name is None and input_prefix is None:
        saved_prompt_name, saved_input_prefix = _saved_training_input_config(output_dir)
        if saved_input_prefix is not None:
            input_prefix = saved_input_prefix
            configured_prompt_name = saved_prompt_name

    index_dir = ensure_dir(output_dir / "index")
    metadata = train_df.reset_index(drop=True).copy()
    metadata.insert(0, "example_id", range(len(metadata)))

    texts = metadata["text"].astype(str).tolist()
    example_embeddings, resolved_input_prefix = _encode_texts(
        model_dir=resolved_model_dir,
        texts=texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        prompt_name=prompt_name,
        input_prefix=input_prefix,
    )
    np.save(index_dir / "example_embeddings.npy", example_embeddings.astype(np.float32))
    metadata.to_csv(index_dir / "example_metadata.csv", index=False, encoding="utf-8-sig")

    codebook_df.to_csv(index_dir / "codebook.csv", index=False, encoding="utf-8-sig")
    codebook_lookup = codebook_df.set_index("code").to_dict(orient="index")

    subcategory_centroids: list[np.ndarray] = []
    subcategory_rows: list[dict[str, Any]] = []
    for code, group in metadata.groupby("code", sort=True):
        indices = group.index.to_numpy()
        centroid = normalize_rows(example_embeddings[indices].mean(axis=0))
        subcategory_centroids.append(centroid)
        info = codebook_lookup.get(code, {})
        subcategory_rows.append(
            {
                "code": code,
                "code_name": info.get("name", group["code_name"].iloc[0]),
                "parent_code": info.get("parent_code", group["parent_code"].iloc[0]),
                "parent_name": info.get("parent_name", group["parent_name"].iloc[0]),
                "is_parent": bool(info.get("is_parent", False)),
                "n_examples": int(len(group)),
            }
        )

    parent_centroids: list[np.ndarray] = []
    parent_rows: list[dict[str, Any]] = []
    for parent, group in metadata.groupby("parent_code", sort=True):
        indices = group.index.to_numpy()
        centroid = normalize_rows(example_embeddings[indices].mean(axis=0))
        parent_centroids.append(centroid)
        child_codes = sorted(group["code"].astype(str).unique().tolist())
        parent_rows.append(
            {
                "parent_code": parent,
                "parent_name": group["parent_name"].iloc[0],
                "child_codes": ", ".join(child_codes),
                "n_examples": int(len(group)),
            }
        )

    np.save(
        index_dir / "subcategory_centroids.npy",
        np.vstack(subcategory_centroids).astype(np.float32),
    )
    pd.DataFrame(subcategory_rows).to_csv(
        index_dir / "subcategory_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.save(index_dir / "parent_centroids.npy", np.vstack(parent_centroids).astype(np.float32))
    pd.DataFrame(parent_rows).to_csv(
        index_dir / "parent_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "created_at": utc_now_iso(),
        "model_path": _model_path_for_config(resolved_model_dir, output_dir),
        "prompt_name": configured_prompt_name,
        "input_prefix": resolved_input_prefix,
        "embeddings_normalized": True,
        "embedding_dim": int(example_embeddings.shape[1]),
        "n_examples": int(len(metadata)),
        "n_subcategory_centroids": int(len(subcategory_rows)),
        "n_parent_centroids": int(len(parent_rows)),
    }
    write_json(config, index_dir / "index_config.json")
    return index_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build production centroid/example index.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--text-col", default="Ответ")
    parser.add_argument("--codes-col", default="Коды_новые")
    parser.add_argument("--context-col", default=None)
    parser.add_argument("--csv-sep", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-name", default=None)
    prompt_group.add_argument("--input-prefix", default=None)
    args = parser.parse_args(argv)

    train_df = load_train_data(
        args.train,
        args.codebook,
        text_col=args.text_col,
        codes_col=args.codes_col,
        context_col=args.context_col,
        csv_sep=args.csv_sep,
    )
    codebook_df = parse_codebook(args.codebook)
    build_index(
        train_df=train_df,
        codebook_df=codebook_df,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        prompt_name=args.prompt_name,
        input_prefix=args.input_prefix,
    )


if __name__ == "__main__":
    main()
