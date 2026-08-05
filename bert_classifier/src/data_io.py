from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_COL_DEFAULT = "Ответ"
CODES_COL_DEFAULT = "Коды_новые"
UNKNOWN_CODE = "UNKNOWN"

_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "Т": "T",
        "Х": "X",
    }
)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str:
    return "" if is_missing(value) else str(value).strip()


def normalize_code(value: Any) -> str:
    if is_missing(value):
        return ""
    code = str(value).strip().upper().translate(_CYRILLIC_TO_LATIN)
    return re.sub(r"\s+", "", code)


def split_codes(value: Any) -> list[str]:
    if is_missing(value):
        return []
    parts = re.sub(r"[;\n\r]+", ",", str(value)).split(",")
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        code = normalize_code(part)
        if code and code not in seen:
            seen.add(code)
            result.append(code)
    return result


def parent_code(code: str) -> str:
    match = re.match(r"^([A-Z]+)", normalize_code(code))
    return match.group(1) if match else normalize_code(code)


def parse_codebook(path: str | Path) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Codebook file not found: {source_path}")
    if source_path.suffix.lower() != ".csv":
        raise ValueError("Codebook must be a .csv file.")

    source = pd.read_csv(
        source_path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    source.columns = [str(column).strip() for column in source.columns]
    required = ["Код", "Категория", "Подкатегория"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(
            f"Missing codebook columns: {missing}. Existing: {list(source.columns)}"
        )

    records: list[dict[str, Any]] = []
    for row_number, row in source.iterrows():
        code = normalize_code(row["Код"])
        category = clean_text(row["Категория"])
        subcategory = clean_text(row["Подкатегория"])
        if not code and not category and not subcategory:
            continue
        if not code or not category or not subcategory:
            raise ValueError(
                f"Invalid codebook row {row_number + 2}: Код, Категория and "
                "Подкатегория must be non-empty."
            )
        parent = parent_code(code)
        if parent == code:
            raise ValueError(
                f"Invalid leaf code {code!r} at codebook row {row_number + 2}."
            )
        records.append(
            {
                "code": code,
                "name": subcategory,
                "parent_code": parent,
                "parent_name": category,
                "is_parent": False,
            }
        )

    if not records:
        raise ValueError(f"Codebook is empty: {source_path}")
    frame = pd.DataFrame(records)
    duplicate_counts = frame["code"].value_counts()
    duplicates = duplicate_counts[duplicate_counts > 1]
    if not duplicates.empty:
        raise ValueError(f"Duplicate codes in codebook: {', '.join(duplicates.index)}")

    category_counts = frame.groupby("parent_code")["parent_name"].nunique()
    inconsistent = category_counts[category_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Different Категория values for parent codes: "
            + ", ".join(inconsistent.index)
        )
    return frame[["code", "name", "parent_code", "parent_name", "is_parent"]]


def read_table(path: str | Path, csv_sep: str | None = None) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")
    if source_path.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(source_path, engine="openpyxl")
    if source_path.suffix.lower() == ".csv":
        return pd.read_csv(
            source_path,
            sep=csv_sep,
            engine="python" if csv_sep is None else "c",
            encoding="utf-8-sig",
        )
    raise ValueError("Input file must be .xlsx, .xlsm or .csv.")


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".xlsx", ".xlsm"}:
        frame.to_excel(output_path, index=False, engine="openpyxl")
    elif output_path.suffix.lower() == ".csv":
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("Output file must be .xlsx, .xlsm or .csv.")
    return output_path


def combine_text(answer: Any, context: Any = None) -> str:
    answer_text = clean_text(answer)
    context_text = clean_text(context)
    if context_text:
        return f"Контекст: {context_text}\nОтвет: {answer_text}"
    return answer_text


def load_labeled_data(
    input_path: str | Path,
    codebook_path: str | Path,
    text_col: str = TEXT_COL_DEFAULT,
    codes_col: str = CODES_COL_DEFAULT,
    context_col: str | None = None,
    csv_sep: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = read_table(input_path, csv_sep=csv_sep)
    required = [text_col, codes_col]
    if context_col:
        required.append(context_col)
    missing = [column for column in required if column not in source.columns]
    if missing:
        existing = ", ".join(map(str, source.columns))
        raise ValueError(f"Missing columns: {missing}. Existing columns: {existing}")

    codebook = parse_codebook(codebook_path)
    known_codes = set(codebook["code"].astype(str))
    records: list[dict[str, Any]] = []
    unknown_codes: set[str] = set()
    for row_id, row in source.iterrows():
        answer = clean_text(row[text_col])
        if not answer:
            continue
        codes = [code for code in split_codes(row[codes_col]) if code != UNKNOWN_CODE]
        if not codes:
            continue
        invalid = [code for code in codes if code not in known_codes]
        if invalid:
            unknown_codes.update(invalid)
            continue
        context = row[context_col] if context_col else None
        records.append(
            {
                "row_id": int(row_id),
                "text": combine_text(answer, context),
                "answer": answer,
                "context": clean_text(context),
                "codes": codes,
            }
        )

    if unknown_codes:
        preview = ", ".join(sorted(unknown_codes)[:20])
        raise ValueError(f"Codes from data are absent in codebook: {preview}")
    if not records:
        raise ValueError("No labeled rows remain after filtering empty text and UNKNOWN.")
    return pd.DataFrame(records), codebook


def _can_stratify(labels: list[str], holdout_size: float) -> bool:
    if holdout_size <= 0 or len(labels) < 2:
        return False
    counts = pd.Series(labels).value_counts()
    n_holdout = math.ceil(len(labels) * holdout_size)
    return bool(not counts.empty and counts.min() >= 2 and n_holdout >= len(counts))


def _split_indices(
    frame: pd.DataFrame,
    holdout_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices = frame.index.tolist()
    if holdout_size <= 0 or len(indices) < 2:
        return indices, []
    labels = frame["codes"].apply(lambda values: str(values[0])).tolist()
    stratify = labels if _can_stratify(labels, holdout_size) else None
    try:
        from sklearn.model_selection import train_test_split

        train_indices, holdout_indices = train_test_split(
            indices,
            test_size=holdout_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        return list(train_indices), list(holdout_indices)
    except ValueError:
        shuffled = indices.copy()
        random.Random(seed).shuffle(shuffled)
        n_holdout = min(
            math.ceil(len(shuffled) * holdout_size),
            max(len(shuffled) - 1, 0),
        )
        return shuffled[n_holdout:], shuffled[:n_holdout]


def _ensure_train_label_coverage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    all_codes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_codes = {code for values in train["codes"] for code in values}
    for code in all_codes:
        if code in train_codes:
            continue
        moved = False
        for holdout in (val, test):
            candidates = holdout[holdout["codes"].apply(lambda values: code in values)]
            if candidates.empty:
                continue
            index = candidates.index[0]
            train = pd.concat([train, holdout.loc[[index]]])
            holdout.drop(index=index, inplace=True)
            train_codes.update(train.loc[index, "codes"])
            moved = True
            break
        if not moved:
            raise ValueError(f"Cannot place code {code!r} into train split.")
    return train, val, test


def split_train_val_test(
    frame: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be non-negative and sum to less than 1.")

    train_val_indices, test_indices = _split_indices(frame, test_size, seed)
    train_val = frame.loc[train_val_indices]
    relative_val_size = val_size / (1.0 - test_size) if val_size > 0 else 0.0
    train_indices, val_indices = _split_indices(train_val, relative_val_size, seed + 1)

    train = frame.loc[train_indices].copy()
    val = frame.loc[val_indices].copy()
    test = frame.loc[test_indices].copy()
    all_codes = sorted({code for values in frame["codes"] for code in values})
    train, val, test = _ensure_train_label_coverage(train, val, test, all_codes)
    return {
        "train": train.sort_index().reset_index(drop=True),
        "val": val.sort_index().reset_index(drop=True),
        "test": test.sort_index().reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assignments = []
    for name, frame in splits.items():
        saved = frame.copy()
        saved["codes"] = saved["codes"].apply(lambda values: ", ".join(values))
        saved.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8-sig")
        assignments.extend(
            {"row_id": int(row_id), "split": name}
            for row_id in frame["row_id"].tolist()
        )
    pd.DataFrame(assignments).sort_values("row_id").to_csv(
        directory / "split_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return directory
