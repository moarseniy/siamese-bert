from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir


def _load_index(model_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    index_dir = Path(model_dir) / "index"
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
    return embeddings, metadata


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

        perplexity = min(30, max(2, (len(embeddings) - 1) // 3))
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
) -> pd.DataFrame:
    embeddings, metadata = _load_index(model_dir)
    sampled_embeddings, sampled_metadata = _sample(embeddings, metadata, sample_size, seed)
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
        tooltip_parts = [
            f"{color_by}: {category}",
            f"code: {row.get('code', '')}",
            f"parent: {row.get('parent_code', '')}",
            f"row_id: {row.get('row_id', '')}",
            f"text: {row.get('text', '')}",
        ]
        tooltip = html.escape(" | ".join(str(part) for part in tooltip_parts))
        circles.append(
            f'<circle cx="{xs[index]:.2f}" cy="{ys[index]:.2f}" r="4" '
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


def visualize_index(
    model_dir: str | Path,
    output_dir: str | Path,
    method: str = "pca",
    color_by: str = "code",
    sample_size: int | None = 5000,
    seed: int = 42,
    output_prefix: str = "embedding_projection",
) -> tuple[Path, Path]:
    output_path = ensure_dir(output_dir)
    projection = build_projection(
        model_dir=model_dir,
        method=method,
        sample_size=sample_size,
        seed=seed,
    )
    csv_path = output_path / f"{output_prefix}.csv"
    html_path = output_path / f"{output_prefix}.html"
    projection.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_projection_html(
        projection=projection,
        output_html=html_path,
        color_by=color_by,
        title=f"{method.upper()} projection colored by {color_by}",
    )
    return csv_path, html_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize indexed survey response embeddings.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--color-by", default="code")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default="embedding_projection")
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
    )
    print(f"Saved projection CSV to {csv_path}")
    print(f"Saved projection HTML to {html_path}")


if __name__ == "__main__":
    main()
