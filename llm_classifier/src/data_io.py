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


def combine_text(answer: Any, context: Any = None) -> str:
    answer_text = clean_text(answer)
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


def parent_code(code: str) -> str:
    match = re.match(r"^([A-Z]+)", normalize_code(code))
    return match.group(1) if match else normalize_code(code)


def parse_codebook(path: str | Path) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Codebook file not found: {source_path}")

    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        source_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
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
        records.append(
            {
                "order": len(records),
                "code": code,
                "name": name,
                "parent_code": parent_code(code),
                "is_parent": parent_code(code) == code,
            }
        )

    if not records:
        raise ValueError(f"Codebook is empty: {source_path}")
    frame = pd.DataFrame(records)
    duplicates = frame["code"].value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        raise ValueError(f"Duplicate codes in codebook: {', '.join(duplicates.index)}")

    name_by_code = frame.set_index("code")["name"].to_dict()
    frame["parent_name"] = frame["parent_code"].map(name_by_code).fillna("")
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
    return "\n".join(f"{row.code}. {row.name}" for row in rows.itertuples(index=False))


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
