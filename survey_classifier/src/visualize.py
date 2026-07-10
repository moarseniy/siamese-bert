from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .data_io import split_codes
from .utils import ensure_dir, normalize_rows

PlotTarget = Literal["examples", "subcategory-centroids", "parent-centroids"]


def _index_dir(model_dir: str | Path) -> Path:
    return Path(model_dir) / "index"


def _load_examples(index_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings_path = index_dir / "example_embeddings.npy"
    metadata_path = index_dir / "example_metadata.csv"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Index is inconsistent: number of embeddings does not match metadata rows. "
            f"embeddings={len(embeddings)}, metadata={len(metadata)}"
        )
    metadata = metadata.copy()
    metadata["point_type"] = "example"
    metadata["_point_radius"] = 4
    return embeddings, metadata


def _single_label_mask(metadata: pd.DataFrame) -> pd.Series:
    if "codes" in metadata.columns:
        return metadata["codes"].apply(lambda value: len(split_codes(value)) == 1)
    if {"row_id", "code"}.issubset(metadata.columns):
        return metadata.groupby("row_id")["code"].transform("nunique") == 1
    raise ValueError("Cannot apply single-label filter: metadata has neither 'codes' nor row_id/code columns.")


def _filter_single_label(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    mask = _single_label_mask(metadata).to_numpy(dtype=bool)
    filtered_embeddings = embeddings[mask]
    filtered_metadata = metadata[mask].reset_index(drop=True)
    if len(filtered_metadata) == 0:
        raise ValueError("No single-label samples found in index metadata.")
    return filtered_embeddings, filtered_metadata


def _load_saved_centroids(
    index_dir: Path,
    target: PlotTarget,
) -> tuple[np.ndarray, pd.DataFrame]:
    if target == "subcategory-centroids":
        embeddings_path = index_dir / "subcategory_centroids.npy"
        metadata_path = index_dir / "subcategory_metadata.csv"
        point_type = "subcategory_centroid"
    elif target == "parent-centroids":
        embeddings_path = index_dir / "parent_centroids.npy"
        metadata_path = index_dir / "parent_metadata.csv"
        point_type = "parent_centroid"
    else:
        raise ValueError(f"Unsupported centroid target: {target}")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Centroid embeddings file not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Centroid metadata file not found: {metadata_path}")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Index is inconsistent: number of centroid embeddings does not match metadata rows. "
            f"embeddings={len(embeddings)}, metadata={len(metadata)}"
        )
    metadata = metadata.copy()
    metadata["point_type"] = point_type
    metadata["_point_radius"] = 7
    if target == "parent-centroids":
        metadata["code"] = metadata["parent_code"]
        metadata["code_name"] = metadata["parent_name"]
    metadata["text"] = metadata.apply(_centroid_text, axis=1)
    return normalize_rows(embeddings), metadata


def _centroid_text(row: pd.Series) -> str:
    point_type = row.get("point_type", "centroid")
    code = row.get("code", row.get("parent_code", ""))
    name = row.get("code_name", row.get("parent_name", ""))
    n_examples = row.get("n_examples", "")
    return f"{point_type}: {code} {name}; n_examples={n_examples}".strip()


def _recompute_centroids_from_examples(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    target: PlotTarget,
) -> tuple[np.ndarray, pd.DataFrame]:
    if target == "subcategory-centroids":
        group_column = "code"
        point_type = "subcategory_centroid_single_label"
    elif target == "parent-centroids":
        group_column = "parent_code"
        point_type = "parent_centroid_single_label"
    else:
        raise ValueError(f"Unsupported centroid target: {target}")

    if group_column not in metadata.columns:
        raise ValueError(f"Cannot recompute centroids: metadata has no {group_column!r} column.")

    centroid_embeddings: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for group_value, group in metadata.groupby(group_column, sort=True):
        indices = group.index.to_numpy()
        centroid = normalize_rows(embeddings[indices].mean(axis=0))
        centroid_embeddings.append(centroid)

        if target == "subcategory-centroids":
            row = {
                "code": str(group_value),
                "code_name": group["code_name"].iloc[0] if "code_name" in group.columns else "",
                "parent_code": group["parent_code"].iloc[0] if "parent_code" in group.columns else "",
                "parent_name": group["parent_name"].iloc[0] if "parent_name" in group.columns else "",
                "n_examples": int(len(group)),
                "point_type": point_type,
                "_point_radius": 7,
            }
        else:
            child_codes = sorted(group["code"].astype(str).unique().tolist()) if "code" in group.columns else []
            row = {
                "code": str(group_value),
                "code_name": group["parent_name"].iloc[0] if "parent_name" in group.columns else "",
                "parent_code": str(group_value),
                "parent_name": group["parent_name"].iloc[0] if "parent_name" in group.columns else "",
                "child_codes": ", ".join(child_codes),
                "n_examples": int(len(group)),
                "point_type": point_type,
                "_point_radius": 7,
            }
        rows.append(row)

    if not centroid_embeddings:
        raise ValueError(f"No centroids could be built for target={target}.")
    centroid_metadata = pd.DataFrame(rows)
    centroid_metadata["text"] = centroid_metadata.apply(_centroid_text, axis=1)
    return np.vstack(centroid_embeddings).astype(np.float32), centroid_metadata


def _sample(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    sample_size: int | None,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(metadata):
        return embeddings, metadata.reset_index(drop=True)
    sampled = metadata.sample(n=sample_size, random_state=seed).sort_index()
    return embeddings[sampled.index.to_numpy()], sampled.reset_index(drop=True)


def _project_embeddings(
    embeddings: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    if len(embeddings) < 2:
        raise ValueError("At least two embeddings are required for visualization.")

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(1, (len(embeddings) - 1) // 3))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(embeddings)
    raise ValueError("method must be one of: pca, tsne.")


def build_projection(
    model_dir: str | Path,
    method: str = "pca",
    sample_size: int | None = 5000,
    seed: int = 42,
    target: PlotTarget = "examples",
    single_label_only: bool = False,
) -> pd.DataFrame:
    index_dir = _index_dir(model_dir)
    if target == "examples":
        embeddings, metadata = _load_examples(index_dir)
        if single_label_only:
            embeddings, metadata = _filter_single_label(embeddings, metadata)
        sampled_embeddings, sampled_metadata = _sample(embeddings, metadata, sample_size, seed)
    elif target in ("subcategory-centroids", "parent-centroids"):
        if single_label_only:
            examples, example_metadata = _load_examples(index_dir)
            examples, example_metadata = _filter_single_label(examples, example_metadata)
            sampled_embeddings, sampled_metadata = _recompute_centroids_from_examples(
                examples,
                example_metadata,
                target=target,
            )
        else:
            sampled_embeddings, sampled_metadata = _load_saved_centroids(index_dir, target)
    else:
        raise ValueError("target must be one of: examples, subcategory-centroids, parent-centroids.")

    points = _project_embeddings(sampled_embeddings, method=method, seed=seed)
    projection = sampled_metadata.copy()
    projection.insert(0, "x", points[:, 0])
    projection.insert(1, "y", points[:, 1])
    return projection


_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#5f4b8b",
]


def _scale(values: pd.Series, output_min: float, output_max: float) -> list[float]:
    numeric = values.astype(float)
    source_min = float(numeric.min())
    source_max = float(numeric.max())
    if source_max == source_min:
        midpoint = (output_min + output_max) / 2
        return [midpoint for _ in numeric]
    return (
        output_min
        + (numeric - source_min) * (output_max - output_min) / (source_max - source_min)
    ).tolist()


def save_projection_html(
    projection: pd.DataFrame,
    output_html: str | Path,
    color_by: str = "code",
    title: str | None = None,
    max_legend_items: int = 30,
) -> Path:
    if color_by not in projection.columns:
        raise ValueError(f"Column {color_by!r} is absent in projection dataframe.")

    output_path = Path(output_html)
    ensure_dir(output_path.parent)

    width = 1200
    height = 780
    padding = 60
    xs = _scale(projection["x"], padding, width - padding)
    ys = _scale(projection["y"], height - padding, padding)

    categories = projection[color_by].astype(str).fillna("").tolist()
    unique_categories = sorted(set(categories))
    color_lookup = {category: index for index, category in enumerate(unique_categories)}

    circles: list[str] = []
    for index, row in projection.reset_index(drop=True).iterrows():
        category = str(row.get(color_by, ""))
        color = _PALETTE[color_lookup[category] % len(_PALETTE)]
        radius = float(row.get("_point_radius", 4))
        tooltip = html.escape(_tooltip_for_row(row, color_by=color_by))
        circles.append(
            f'<circle cx="{xs[index]:.2f}" cy="{ys[index]:.2f}" r="{radius:.1f}" '
            f'fill="{color}" fill-opacity="0.72"><title>{tooltip}</title></circle>'
        )

    legend = ""
    if len(unique_categories) <= max_legend_items:
        items = []
        for index, category in enumerate(unique_categories):
            color = _PALETTE[index % len(_PALETTE)]
            items.append(
                '<div class="legend-item">'
                f'<span class="swatch" style="background:{color}"></span>'
                f"{html.escape(category)}</div>"
            )
        legend = '<aside class="legend">' + "\n".join(items) + "</aside>"

    page_title = html.escape(title or f"Survey classifier embeddings by {color_by}")
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{page_title}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f8;
      color: #1f2328;
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 6px;
    }}
    .meta {{
      margin: 0 0 18px;
      color: #656d76;
      font-size: 14px;
    }}
    .layout {{
      display: flex;
      gap: 18px;
      align-items: flex-start;
    }}
    svg {{
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      max-width: 100%;
      height: auto;
    }}
    .legend {{
      min-width: 180px;
      max-width: 280px;
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 12px;
      font-size: 13px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 4px 0;
      word-break: break-word;
    }}
    .swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: 0 0 auto;
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>{page_title}</h1>
    <p class="meta">points={len(projection)}; color_by={html.escape(color_by)}; hover a point to inspect metadata</p>
    <div class="layout">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{page_title}">
        <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
        <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#d8dee4"></line>
        <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#d8dee4"></line>
        {"".join(circles)}
      </svg>
      {legend}
    </div>
  </main>
</body>
</html>
"""
    output_path.write_text(body, encoding="utf-8")
    return output_path


def _tooltip_for_row(row: pd.Series, color_by: str) -> str:
    fields = [
        ("point_type", row.get("point_type", "")),
        (color_by, row.get(color_by, "")),
        ("code", row.get("code", "")),
        ("code_name", row.get("code_name", "")),
        ("parent_code", row.get("parent_code", "")),
        ("parent_name", row.get("parent_name", "")),
        ("child_codes", row.get("child_codes", "")),
        ("n_examples", row.get("n_examples", "")),
        ("row_id", row.get("row_id", "")),
        ("text", row.get("text", "")),
    ]
    return " | ".join(f"{name}: {value}" for name, value in fields if str(value))


def _default_output_prefix(target: PlotTarget, method: str, single_label_only: bool) -> str:
    parts = ["embedding_projection", target.replace("-", "_"), method]
    if single_label_only:
        parts.append("single_label")
    return "_".join(parts)


def visualize_index(
    model_dir: str | Path,
    output_dir: str | Path,
    method: str = "pca",
    color_by: str = "code",
    sample_size: int | None = 5000,
    seed: int = 42,
    output_prefix: str | None = None,
    target: PlotTarget = "examples",
    single_label_only: bool = False,
) -> tuple[Path, Path]:
    output_path = ensure_dir(output_dir)
    projection = build_projection(
        model_dir=model_dir,
        method=method,
        sample_size=sample_size,
        seed=seed,
        target=target,
        single_label_only=single_label_only,
    )
    prefix = output_prefix or _default_output_prefix(target, method, single_label_only)
    csv_path = output_path / f"{prefix}.csv"
    html_path = output_path / f"{prefix}.html"
    projection.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_projection_html(
        projection=projection,
        output_html=html_path,
        color_by=color_by,
        title=f"{method.upper()} projection of {target} colored by {color_by}",
    )
    return csv_path, html_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize indexed survey response embeddings.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--target",
        choices=["examples", "subcategory-centroids", "parent-centroids"],
        default="examples",
    )
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--color-by", default="code")
    parser.add_argument("--single-label-only", action="store_true")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or args.model_dir / "reports"
    csv_path, html_path = visualize_index(
        model_dir=args.model_dir,
        output_dir=output_dir,
        method=args.method,
        color_by=args.color_by,
        sample_size=args.sample_size,
        seed=args.seed,
        output_prefix=args.output_prefix,
        target=args.target,
        single_label_only=args.single_label_only,
    )
    print(f"Saved projection CSV to {csv_path}")
    print(f"Saved projection HTML to {html_path}")


if __name__ == "__main__":
    main()
