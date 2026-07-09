from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import unique_preserve_order

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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_code(code: str) -> str:
    if _is_missing(code):
        return ""
    value = str(code).strip().upper().translate(_CYRILLIC_TO_LATIN)
    return re.sub(r"\s+", "", value)


def split_codes(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [item for item in value]
    else:
        normalized_separators = re.sub(r"[;\n\r]+", ",", str(value))
        parts = normalized_separators.split(",")
    codes = [normalize_code(part) for part in parts]
    return unique_preserve_order([code for code in codes if code])


def parent_code(code: str) -> str:
    normalized = normalize_code(code)
    match = re.match(r"^([A-Z]+)", normalized)
    if not match:
        return normalized
    return match.group(1)


def parse_codebook(txt_path: str | Path) -> pd.DataFrame:
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Codebook file not found: {path}")

    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([^.]+?)\.\s*(.+?)\s*$", line)
        if not match:
            raise ValueError(f"Invalid codebook line {line_number}: {raw_line!r}")
        code = normalize_code(match.group(1))
        name = match.group(2).strip()
        if not code or not name:
            raise ValueError(f"Invalid codebook line {line_number}: {raw_line!r}")
        records.append({"code": code, "name": name})

    if not records:
        raise ValueError(f"Codebook is empty: {path}")

    code_counts = pd.Series([record["code"] for record in records]).value_counts()
    duplicates = code_counts[code_counts > 1]
    if not duplicates.empty:
        dupes = ", ".join(duplicates.index.tolist())
        raise ValueError(f"Duplicate codes in codebook: {dupes}")

    name_by_code = {record["code"]: record["name"] for record in records}
    rows: list[dict[str, Any]] = []
    for record in records:
        code = record["code"]
        parent = parent_code(code)
        rows.append(
            {
                "code": code,
                "name": record["name"],
                "parent_code": parent,
                "parent_name": name_by_code.get(parent, record["name"] if parent == code else ""),
                "is_parent": parent == code,
            }
        )
    return pd.DataFrame(rows, columns=["code", "name", "parent_code", "parent_name", "is_parent"])


def _require_columns(frame: pd.DataFrame, columns: list[str], source: str | Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        existing = ", ".join(map(str, frame.columns.tolist()))
        raise ValueError(f"Missing columns in {source}: {missing}. Existing columns: {existing}")


def load_train_data(
    xlsx_path: str | Path,
    txt_path: str | Path,
    text_col: str = TEXT_COL_DEFAULT,
    codes_col: str = CODES_COL_DEFAULT,
) -> pd.DataFrame:
    xlsx = Path(xlsx_path)
    if not xlsx.exists():
        raise FileNotFoundError(f"Training Excel file not found: {xlsx}")

    codebook = parse_codebook(txt_path)
    codebook_by_code = codebook.set_index("code").to_dict(orient="index")

    source = pd.read_excel(xlsx, engine="openpyxl")
    _require_columns(source, [text_col, codes_col], xlsx)

    records: list[dict[str, Any]] = []
    missing_codes: set[str] = set()
    for row_id, row in source.iterrows():
        text_value = row[text_col]
        if _is_missing(text_value):
            continue
        text = str(text_value).strip()
        if not text:
            continue

        codes = [code for code in split_codes(row[codes_col]) if code != UNKNOWN_CODE]
        if not codes:
            continue

        unknown_in_codebook = [code for code in codes if code not in codebook_by_code]
        if unknown_in_codebook:
            missing_codes.update(unknown_in_codebook)
            continue

        codes_str = ", ".join(codes)
        for code in codes:
            codebook_row = codebook_by_code[code]
            records.append(
                {
                    "row_id": int(row_id),
                    "text": text,
                    "codes": codes_str,
                    "code": code,
                    "code_name": codebook_row["name"],
                    "parent_code": codebook_row["parent_code"],
                    "parent_name": codebook_row["parent_name"],
                }
            )

    if missing_codes:
        preview = ", ".join(sorted(missing_codes)[:20])
        raise ValueError(f"Codes from training data are absent in codebook: {preview}")

    columns = ["row_id", "text", "codes", "code", "code_name", "parent_code", "parent_name"]
    if not records:
        raise ValueError("No training rows left after dropping empty texts/codes and UNKNOWN labels.")
    return pd.DataFrame(records, columns=columns)
