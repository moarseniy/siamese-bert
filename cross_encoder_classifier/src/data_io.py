from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_COL_DEFAULT = "Ответ"
CODES_COL_DEFAULT = "Коды_новые"
SENTIMENTS_COL_DEFAULT = "Тональности"
UNKNOWN_CODE = "UNKNOWN"

MODEL_CLASS_NAMES = {
    0: "absent",
    1: "neutral",
    2: "positive",
    3: "negative",
}
SENTIMENT_NAMES = {0: "нейтральная", 1: "позитивная", 2: "негативная"}
MODEL_CLASS_BY_SENTIMENT = {0: 1, 1: 2, 2: 3}
SENTIMENT_BY_MODEL_CLASS = {
    value: key for key, value in MODEL_CLASS_BY_SENTIMENT.items()
}


class ConflictingSentimentsError(ValueError):
    """Raised when one response assigns different sentiments to one code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Code {code!r} has conflicting sentiments in one row.")


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
_SENTIMENT_ALIASES = {
    "0": 0,
    "0.0": 0,
    "neutral": 0,
    "нейтральная": 0,
    "нейтральный": 0,
    "нейтрально": 0,
    "1": 1,
    "1.0": 1,
    "positive": 1,
    "позитивная": 1,
    "позитивный": 1,
    "позитивно": 1,
    "2": 2,
    "2.0": 2,
    "negative": 2,
    "негативная": 2,
    "негативный": 2,
    "негативно": 2,
}


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


def _split_items(value: Any) -> list[str]:
    if is_missing(value):
        return []
    return [item.strip() for item in re.split(r"[;,\n\r]+", str(value)) if item.strip()]


def split_codes(value: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in _split_items(value):
        code = normalize_code(item)
        if code and code not in seen:
            seen.add(code)
            result.append(code)
    return result


def parse_sentiment(value: Any) -> int:
    key = clean_text(value).casefold()
    if key not in _SENTIMENT_ALIASES:
        raise ValueError(
            f"Unknown sentiment {value!r}; expected 0/1/2 or neutral/positive/negative."
        )
    return _SENTIMENT_ALIASES[key]


def parse_annotations(
    codes_value: Any,
    sentiments_value: Any = None,
) -> list[tuple[str, int]]:
    """Return unique ``(code, raw_sentiment)`` annotations for one response."""
    code_items = _split_items(codes_value)
    if not code_items or all(
        normalize_code(item) == UNKNOWN_CODE for item in code_items
    ):
        return []

    sentiment_items = _split_items(sentiments_value)
    parsed: list[tuple[str, int]] = []
    if sentiment_items:
        codes = [normalize_code(item) for item in code_items]
        codes = [code for code in codes if code != UNKNOWN_CODE]
        if len(codes) != len(sentiment_items):
            raise ValueError(
                "Codes and sentiments must have equal item counts: "
                f"{len(codes)} != {len(sentiment_items)}."
            )
        parsed = [
            (code, parse_sentiment(sentiment))
            for code, sentiment in zip(codes, sentiment_items)
        ]
    else:
        for item in code_items:
            if normalize_code(item) == UNKNOWN_CODE:
                continue
            match = re.match(r"^(.+?)\s*[:=|]\s*([^:=|]+?)\s*$", item)
            if not match:
                raise ValueError(
                    "Sentiment is missing. Add a sentiments column or use inline "
                    "annotations such as A1:2, B2:0."
                )
            parsed.append(
                (normalize_code(match.group(1)), parse_sentiment(match.group(2)))
            )

    unique: dict[str, int] = {}
    for code, sentiment in parsed:
        if not code:
            raise ValueError("Annotation contains an empty code.")
        if code in unique and unique[code] != sentiment:
            raise ConflictingSentimentsError(code)
        unique[code] = sentiment
    return list(unique.items())


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
    frame["description"] = frame.apply(
        lambda row: (
            f"{row['code']}. Категория: {row['parent_name']}. "
            f"Подкатегория: {row['name']}"
        ),
        axis=1,
    )
    return frame[
        ["code", "name", "description", "parent_code", "parent_name", "is_parent"]
    ]


def leaf_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    leaves = codebook.loc[~codebook["is_parent"].astype(bool)].copy()
    if leaves.empty:
        raise ValueError("Codebook has no leaf subcategories.")
    return leaves.reset_index(drop=True)


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
    sentiments_col: str | None = SENTIMENTS_COL_DEFAULT,
    context_col: str | None = None,
    csv_sep: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = read_table(input_path, csv_sep=csv_sep)
    required = [text_col, codes_col]
    if context_col:
        required.append(context_col)
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Existing columns: {list(source.columns)}"
        )

    codebook = parse_codebook(codebook_path)
    leaves = leaf_codebook(codebook)
    known_leaves = set(leaves["code"].astype(str))
    sentiment_column_exists = bool(sentiments_col and sentiments_col in source.columns)
    records: list[dict[str, Any]] = []
    unknown_codes: set[str] = set()
    conflicting_rows: list[int] = []
    empty_text_rows = 0
    for row_id, row in source.iterrows():
        answer = clean_text(row[text_col])
        if not answer:
            empty_text_rows += 1
            continue
        sentiment_value = row[sentiments_col] if sentiment_column_exists else None
        try:
            annotations = parse_annotations(row[codes_col], sentiment_value)
        except ConflictingSentimentsError:
            conflicting_rows.append(int(row_id) + 2)
            continue
        except ValueError as exc:
            raise ValueError(
                f"Invalid annotations at source row {row_id + 2}: {exc}"
            ) from exc
        invalid = [code for code, _ in annotations if code not in known_leaves]
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
                "codes": [code for code, _ in annotations],
                "annotations": annotations,
            }
        )

    if unknown_codes:
        preview = ", ".join(sorted(unknown_codes)[:20])
        raise ValueError(f"Codes are absent from leaf subcategories: {preview}")
    if not records:
        raise ValueError(
            "No usable responses remain after loading the training data "
            f"(conflicting sentiment rows skipped: {len(conflicting_rows)})."
        )
    data = pd.DataFrame(records)
    data.attrs["load_report"] = {
        "input_rows": int(len(source)),
        "loaded_rows": int(len(data)),
        "empty_text_rows": int(empty_text_rows),
        "skipped_conflicting_sentiment_rows": int(len(conflicting_rows)),
        "skipped_conflicting_sentiment_source_rows": conflicting_rows,
    }
    return data, codebook


def _primary_split_label(annotations: list[tuple[str, int]]) -> str:
    if not annotations:
        return UNKNOWN_CODE
    code, sentiment = annotations[0]
    return f"{code}:{sentiment}"


def _can_stratify(labels: list[str], holdout_size: float) -> bool:
    if holdout_size <= 0 or len(labels) < 2:
        return False
    counts = pd.Series(labels).value_counts()
    n_holdout = math.ceil(len(labels) * holdout_size)
    return bool(not counts.empty and counts.min() >= 2 and n_holdout >= len(counts))


def _split_indices(
    frame: pd.DataFrame, holdout_size: float, seed: int
) -> tuple[list[int], list[int]]:
    indices = frame.index.tolist()
    if holdout_size <= 0 or len(indices) < 2:
        return indices, []
    labels = frame["annotations"].apply(_primary_split_label).tolist()
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
            math.ceil(len(shuffled) * holdout_size), max(len(shuffled) - 1, 0)
        )
        return shuffled[n_holdout:], shuffled[:n_holdout]


def _ensure_train_coverage(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_codes = {
        code
        for annotations in pd.concat([train, val, test])["annotations"]
        for code, _ in annotations
    }
    required_sentiments = {
        sentiment
        for annotations in pd.concat([train, val, test])["annotations"]
        for _, sentiment in annotations
    }

    def coverage(frame: pd.DataFrame) -> tuple[set[str], set[int]]:
        annotations = [item for values in frame["annotations"] for item in values]
        return {code for code, _ in annotations}, {
            sentiment for _, sentiment in annotations
        }

    for kind, required in (
        ("code", required_codes),
        ("sentiment", required_sentiments),
    ):
        current_codes, current_sentiments = coverage(train)
        current: set[Any] = current_codes if kind == "code" else current_sentiments
        for missing_value in sorted(required):
            if missing_value in current:
                continue
            moved = False
            for holdout in (val, test):
                mask = holdout["annotations"].apply(
                    lambda values: any(
                        (code if kind == "code" else sentiment) == missing_value
                        for code, sentiment in values
                    )
                )
                candidates = holdout[mask]
                if candidates.empty:
                    continue
                index = candidates.index[0]
                train = pd.concat([train, holdout.loc[[index]]])
                holdout.drop(index=index, inplace=True)
                current_codes, current_sentiments = coverage(train)
                current = current_codes if kind == "code" else current_sentiments
                moved = True
                break
            if not moved:
                raise ValueError(
                    f"Cannot place {kind} {missing_value!r} into train split."
                )
    return train, val, test


def split_train_val_test(
    frame: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1:
        raise ValueError(
            "val_size and test_size must be non-negative and sum to less than 1."
        )
    train_val_indices, test_indices = _split_indices(frame, test_size, seed)
    train_val = frame.loc[train_val_indices]
    relative_val = val_size / (1.0 - test_size) if val_size > 0 else 0.0
    train_indices, val_indices = _split_indices(train_val, relative_val, seed + 1)
    train = frame.loc[train_indices].copy()
    val = frame.loc[val_indices].copy()
    test = frame.loc[test_indices].copy()
    train, val, test = _ensure_train_coverage(train, val, test)
    return {
        "train": train.sort_index().reset_index(drop=True),
        "val": val.sort_index().reset_index(drop=True),
        "test": test.sort_index().reset_index(drop=True),
    }


def build_pairs(
    responses: pd.DataFrame,
    codebook: pd.DataFrame,
    negative_ratio: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Expand responses into answer/code pairs; ``None`` keeps every negative."""
    if negative_ratio is not None and negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative or None.")
    leaves = leaf_codebook(codebook)
    leaf_records = leaves[["code", "name", "description", "parent_code"]].to_dict(
        "records"
    )
    columns = [
        "response_position",
        "row_id",
        "text",
        "answer",
        "context",
        "code",
        "code_name",
        "code_description",
        "label",
        "sentiment",
    ]
    rows: list[dict[str, Any]] = []
    for response_position, response in responses.reset_index(drop=True).iterrows():
        positive = dict(response["annotations"])
        candidates = [
            record for record in leaf_records if record["code"] not in positive
        ]
        if negative_ratio is None:
            selected = leaf_records
        else:
            target = math.ceil(negative_ratio * max(len(positive), 1))
            target = min(target, len(candidates))
            rng = random.Random(seed * 1_000_003 + int(response["row_id"]))
            positive_parents = {
                record["parent_code"]
                for record in leaf_records
                if record["code"] in positive
            }
            hard_candidates = [
                record
                for record in candidates
                if record["parent_code"] in positive_parents
            ]
            rng.shuffle(hard_candidates)
            hard_count = min(len(hard_candidates), math.ceil(target / 2))
            sampled = hard_candidates[:hard_count]
            sampled_codes = {record["code"] for record in sampled}
            remaining = [
                record for record in candidates if record["code"] not in sampled_codes
            ]
            sampled.extend(rng.sample(remaining, target - hard_count))
            candidates = sampled
            selected = [record for record in leaf_records if record["code"] in positive]
            selected.extend(candidates)
        for record in selected:
            raw_sentiment = positive.get(record["code"])
            label = (
                MODEL_CLASS_BY_SENTIMENT[raw_sentiment]
                if raw_sentiment is not None
                else 0
            )
            rows.append(
                {
                    "response_position": response_position,
                    "row_id": int(response["row_id"]),
                    "text": str(response["text"]),
                    "answer": str(response["answer"]),
                    "context": str(response["context"]),
                    "code": record["code"],
                    "code_name": record["name"],
                    "code_description": record["description"],
                    "label": label,
                    "sentiment": raw_sentiment,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def annotations_to_text(annotations: list[tuple[str, int]]) -> tuple[str, str, str]:
    codes = ", ".join(code for code, _ in annotations) or UNKNOWN_CODE
    sentiments = ", ".join(str(sentiment) for _, sentiment in annotations)
    combined = ", ".join(f"{code}:{sentiment}" for code, sentiment in annotations)
    return codes, sentiments, combined or UNKNOWN_CODE


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assignments: list[dict[str, Any]] = []
    for name, frame in splits.items():
        saved = frame.drop(columns=["annotations"]).copy()
        annotation_text = frame["annotations"].apply(annotations_to_text)
        saved["codes"] = annotation_text.apply(lambda item: item[0])
        saved["sentiments"] = annotation_text.apply(lambda item: item[1])
        saved["code_sentiments"] = annotation_text.apply(lambda item: item[2])
        saved.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8-sig")
        assignments.extend(
            {"row_id": int(row_id), "split": name}
            for row_id in frame["row_id"].tolist()
        )
    pd.DataFrame(assignments).sort_values("row_id").to_csv(
        directory / "split_assignments.csv", index=False, encoding="utf-8-sig"
    )
    return directory
