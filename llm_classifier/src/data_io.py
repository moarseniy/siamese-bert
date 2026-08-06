from __future__ import annotations

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


def add_after_semicolon_prefix(value: Any, prefix: Any = None) -> str:
    text = clean_text(value)
    normalized_prefix = clean_text(prefix)
    if not normalized_prefix or ";" not in text:
        return text
    before, after = text.split(";", 1)
    suffix = f" {after.strip()}" if after.strip() else ""
    return f"{before.strip()}; {normalized_prefix}{suffix}"


def combine_text(
    answer: Any,
    context: Any = None,
    after_semicolon_prefix: Any = None,
) -> str:
    answer_text = add_after_semicolon_prefix(answer, after_semicolon_prefix)
    context_text = clean_text(context)
    if context_text:
        return f"Контекст: {context_text}\nОтвет: {answer_text}"
    return answer_text


def normalize_code(value: Any) -> str:
    if is_missing(value):
        return ""
    code = str(value).strip().upper().translate(_CYRILLIC_TO_LATIN)
    return re.sub(r"\s+", "", code)


def split_codes(value: Any) -> list[str]:
    if is_missing(value):
        return []
    normalized_separators = re.sub(r"[;\n\r]+", ",", str(value))
    result: list[str] = []
    seen: set[str] = set()
    for part in normalized_separators.split(","):
        code = normalize_code(part)
        if code and code not in seen:
            seen.add(code)
            result.append(code)
    return result


def parse_code_sentiments(value: Any) -> list[tuple[str, int]]:
    """Parse inline labels such as ``E1:1, A3:0``."""
    if is_missing(value):
        return []
    normalized_separators = re.sub(r"[;\n\r]+", ",", str(value))
    labels: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for raw_item in normalized_separators.split(","):
        item = raw_item.strip()
        if not item or normalize_code(item) == UNKNOWN_CODE:
            continue
        match = re.match(r"^(.+?)\s*[:=|]\s*([012])\s*$", item)
        if not match:
            raise ValueError(
                f"Invalid code/sentiment label {item!r}; expected format E1:1."
            )
        code = normalize_code(match.group(1))
        sentiment = int(match.group(2))
        label = (code, sentiment)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


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
                "order": len(records),
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
    duplicates = frame["code"].value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        raise ValueError(f"Duplicate codes in codebook: {', '.join(duplicates.index)}")

    category_counts = frame.groupby("parent_code")["parent_name"].nunique()
    inconsistent = category_counts[category_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Different Категория values for parent codes: "
            + ", ".join(inconsistent.index)
        )
    return frame[
        ["order", "code", "name", "parent_code", "parent_name", "is_parent"]
    ]


def assignable_codes(codebook: pd.DataFrame) -> list[str]:
    leaves = codebook.loc[~codebook["is_parent"].astype(bool), "code"].astype(str).tolist()
    if leaves:
        return leaves
    return codebook["code"].astype(str).tolist()


def render_codebook(codebook: pd.DataFrame) -> str:
    rows = codebook.sort_values("order")
    return "\n".join(
        f"{row.code}. Категория: {row.parent_name}. "
        f"Подкатегория: {row.name}"
        for row in rows.itertuples(index=False)
    )


def read_table(path: str | Path, csv_sep: str | None = None) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")
    suffix = source_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(source_path, engine="openpyxl")
    if suffix == ".csv":
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
    suffix = output_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        frame.to_excel(output_path, index=False, engine="openpyxl")
    elif suffix == ".csv":
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("Output file must be .xlsx, .xlsm or .csv.")
    return output_path
