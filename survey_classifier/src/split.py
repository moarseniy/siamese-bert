from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, unique_preserve_order, utc_now_iso, write_json


def _validate_split_sizes(val_size: float, test_size: float) -> None:
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1).")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must be in [0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1.")


def _response_groups(train_df: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "text", "code"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Training dataframe is missing columns: {sorted(missing)}")

    grouped = (
        train_df.groupby("row_id", sort=False)
        .agg(
            text=("text", "first"),
            codes=("code", lambda values: unique_preserve_order(map(str, values))),
        )
        .reset_index()
    )
    grouped["strata"] = grouped["codes"].apply(lambda codes: codes[0] if codes else "__empty__")
    return grouped


def _can_stratify(labels: list[str], holdout_size: float) -> bool:
    if holdout_size <= 0 or len(labels) < 2:
        return False
    counts = pd.Series(labels).value_counts()
    if counts.empty or counts.min() < 2:
        return False
    n_holdout = math.ceil(len(labels) * holdout_size)
    return n_holdout >= len(counts)


def _random_split_ids(
    row_ids: list[Any],
    holdout_size: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    shuffled = row_ids.copy()
    random.Random(seed).shuffle(shuffled)
    n_holdout = math.ceil(len(shuffled) * holdout_size)
    n_holdout = min(max(n_holdout, 0), max(len(shuffled) - 1, 0))
    holdout = shuffled[:n_holdout]
    train = shuffled[n_holdout:]
    return train, holdout


def _split_ids(
    groups: pd.DataFrame,
    holdout_size: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    row_ids = groups["row_id"].tolist()
    if holdout_size <= 0 or len(row_ids) < 2:
        return row_ids, []

    labels = groups["strata"].astype(str).tolist()
    stratify = labels if _can_stratify(labels, holdout_size) else None
    try:
        from sklearn.model_selection import train_test_split

        train_ids, holdout_ids = train_test_split(
            row_ids,
            test_size=holdout_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        return list(train_ids), list(holdout_ids)
    except (ImportError, ValueError):
        return _random_split_ids(row_ids, holdout_size, seed)


def split_train_val_test(
    train_df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    _validate_split_sizes(val_size, test_size)
    groups = _response_groups(train_df)

    train_val_ids, test_ids = _split_ids(groups, test_size, seed)
    train_val_groups = groups[groups["row_id"].isin(train_val_ids)].copy()

    relative_val_size = 0.0
    if val_size > 0 and len(train_val_groups) > 1:
        relative_val_size = val_size / (1.0 - test_size)
    train_ids, val_ids = _split_ids(train_val_groups, relative_val_size, seed + 1)

    split_ids = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }
    return {
        name: train_df[train_df["row_id"].isin(ids)].reset_index(drop=True)
        for name, ids in split_ids.items()
    }


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path, seed: int) -> Path:
    split_dir = ensure_dir(out_dir)
    assignment_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"created_at": utc_now_iso(), "seed": seed, "splits": {}}

    for split_name in ("train", "val", "test"):
        frame = splits[split_name]
        frame.to_csv(split_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")
        row_ids = unique_preserve_order(frame["row_id"].tolist()) if not frame.empty else []
        for row_id in row_ids:
            assignment_rows.append({"row_id": row_id, "split": split_name})
        summary["splits"][split_name] = {
            "rows_long": int(len(frame)),
            "responses": int(len(row_ids)),
            "codes": int(frame["code"].nunique()) if not frame.empty else 0,
        }

    pd.DataFrame(assignment_rows, columns=["row_id", "split"]).to_csv(
        split_dir / "split_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(summary, split_dir / "split_summary.json")
    return split_dir
