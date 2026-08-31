from __future__ import annotations

import asyncio
import csv
import io
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from tokenizers import Tokenizer


APP_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "config.yaml").is_file()
)
SERVICE_CONFIG_PATH = APP_ROOT / "config.yaml"
SETTINGS = yaml.safe_load(SERVICE_CONFIG_PATH.read_text(encoding="utf-8"))
VLLM_URL = str(SETTINGS["vllm_url"]).rstrip("/")
VLLM_API_KEY = str(SETTINGS.get("vllm_api_key") or "").strip()
SERVED_MODEL_NAME = str(SETTINGS.get("served_model_name") or "").strip()
PAIR_BATCH_SIZE = int(SETTINGS.get("pair_batch_size", 512))
REQUEST_TIMEOUT = float(SETTINGS.get("request_timeout", 600))
DEFAULT_CODEBOOK = str(SETTINGS.get("default_codebook", "default"))
CODEBOOK_PATHS = {
    str(name): APP_ROOT / path
    for name, path in SETTINGS.get(
        "codebooks", {"default": "artifacts/codebook.csv"}
    ).items()
}
TOKENIZER_PATH = APP_ROOT / SETTINGS.get("tokenizer_path", "artifacts/tokenizer.json")
CONFIG_PATH = APP_ROOT / SETTINGS.get(
    "classifier_config_path", "artifacts/classifier_config.json"
)
SENTIMENT_NAMES = ("neutral", "positive", "negative")


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def load_codebook(
    path: Path, description_format: object
) -> list[tuple[str, str, str, str]]:
    source = path.read_text(encoding="utf-8-sig")
    try:
        delimiter = csv.Sniffer().sniff(source[:4096], delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    rows = csv.DictReader(io.StringIO(source), delimiter=delimiter)
    required = {"Код", "Категория", "Подкатегория"}
    if missing := required - set(rows.fieldnames or []):
        raise ValueError(f"Missing codebook columns: {sorted(missing)}")

    codebook: list[tuple[str, str, str, str]] = []
    for row_number, row in enumerate(rows, 2):
        code = re.sub(r"\s+", "", text(row["Код"]).upper())
        category = text(row["Категория"])
        subcategory = text(row["Подкатегория"])
        if not code and not category and not subcategory:
            continue
        if not code or not category or not subcategory:
            raise ValueError(f"Invalid codebook row {row_number}")
        description = f"Категория: {category}. Подкатегория: {subcategory}"
        if description_format in {
            "category_subcategory_v1",
            "code_category_subcategory_v1",
        }:
            description = f"{code}. {description}"
        codebook.append((code, category, subcategory, description))
    if not codebook:
        raise ValueError(f"Codebook is empty: {path}")
    return codebook


def load_assets() -> tuple[
    dict[str, list[tuple[str, str, str, str]]], Tokenizer, dict[str, object]
]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    description_format = config.get(
        "code_description_format", "category_subcategory_v2"
    )
    if not CODEBOOK_PATHS or DEFAULT_CODEBOOK not in CODEBOOK_PATHS:
        raise ValueError("default_codebook must exist in codebooks")
    codebooks = {
        name: load_codebook(path, description_format)
        for name, path in CODEBOOK_PATHS.items()
    }

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer.no_padding()
    tokenizer.enable_truncation(
        max_length=int(config.get("max_length", 256)), strategy="longest_first"
    )
    return codebooks, tokenizer, config


def add_prefix(value: object, prefix: str) -> str:
    value = text(value)
    if not prefix or ";" not in value:
        return value
    before, after = value.split(";", 1)
    return f"{before.strip()}; {prefix} {after.strip()}".strip()


async def classify(app: FastAPI, token_ids: list[list[int]]) -> list[list[float]]:
    response = await app.state.client.post(
        f"{VLLM_URL}/classify",
        json={"model": app.state.model, "input": token_ids, "use_activation": True},
    )
    if not response.is_success:
        raise RuntimeError(f"vLLM HTTP {response.status_code}: {response.text[:500]}")
    try:
        items = sorted(response.json()["data"], key=lambda item: item["index"])
        probabilities = [list(map(float, item["probs"])) for item in items]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Invalid response from vLLM /classify") from exc
    if len(probabilities) != len(token_ids) or any(
        len(values) != 4 for values in probabilities
    ):
        raise RuntimeError("vLLM returned an unexpected number of predictions")
    return probabilities


def decode(
    probabilities: list[list[float]],
    codebook: list[tuple[str, str, str, str]],
    threshold: float,
    max_labels: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, tuple[str, str, str, str], int]] = []
    for values, entry in zip(probabilities, codebook, strict=True):
        presence = 1.0 - values[0]
        if presence >= threshold:
            sentiment = max(range(3), key=lambda index: values[index + 1])
            candidates.append((presence, entry, sentiment))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "code": entry[0],
            "category": entry[1],
            "subcategory": entry[2],
            "sentiment": sentiment,
            "sentiment_name": SENTIMENT_NAMES[sentiment],
            "confidence": round(presence, 6),
        }
        for presence, entry, sentiment in candidates[:max_labels]
    ]


async def predict_answers(
    app: FastAPI,
    answers: list[object],
    codebook: list[tuple[str, str, str, str]],
) -> list[list[dict[str, Any]]]:
    config = app.state.config
    results: list[list[dict[str, Any]]] = [[] for _ in answers]
    rows_per_group = max(1, PAIR_BATCH_SIZE // len(codebook))
    prefix = text(config.get("after_semicolon_prefix", ""))

    for start in range(0, len(answers), rows_per_group):
        pairs: list[tuple[str, str]] = []
        active_rows: list[int] = []
        for row, answer in enumerate(answers[start : start + rows_per_group]):
            answer = add_prefix(answer, prefix)
            if answer:
                active_rows.append(row)
                pairs.extend((answer, entry[3]) for entry in codebook)

        probabilities: list[list[float]] = []
        for offset in range(0, len(pairs), PAIR_BATCH_SIZE):
            encodings = app.state.tokenizer.encode_batch(
                pairs[offset : offset + PAIR_BATCH_SIZE], add_special_tokens=True
            )
            probabilities.extend(
                await classify(app, [encoding.ids for encoding in encodings])
            )

        for position, row in enumerate(active_rows):
            left = position * len(codebook)
            results[start + row] = decode(
                probabilities[left : left + len(codebook)],
                codebook,
                float(config["threshold"]),
                int(config.get("max_labels", 6)),
            )
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    codebooks, tokenizer, config = load_assets()
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
    client = httpx.AsyncClient(headers=headers, timeout=REQUEST_TIMEOUT)
    response = await client.get(f"{VLLM_URL}/v1/models")
    response.raise_for_status()
    models = response.json().get("data", [])
    model = SERVED_MODEL_NAME or (models[0].get("id") if models else "")
    if not model:
        raise RuntimeError("vLLM returned no served model")

    app.state.client = client
    app.state.model = model
    app.state.codebooks = codebooks
    app.state.tokenizer = tokenizer
    app.state.config = config
    app.state.lock = asyncio.Lock()
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@app.post("/predict")
async def predict(
    records: list[dict[str, Any]], codebook: str | None = None
) -> list[dict[str, Any]]:
    codebook_name = codebook or DEFAULT_CODEBOOK
    selected_codebook = app.state.codebooks.get(codebook_name)
    if selected_codebook is None:
        raise HTTPException(404, f"Unknown codebook: {codebook_name}")
    missing = [index for index, record in enumerate(records) if "RESPONSE" not in record]
    if missing:
        raise HTTPException(422, f"RESPONSE is missing in items: {missing}")
    try:
        async with app.state.lock:
            predictions = await predict_answers(
                app,
                [record["RESPONSE"] for record in records],
                selected_codebook,
            )
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(502, str(exc)) from exc
    return [
        {**record, "PREDICTIONS": prediction}
        for record, prediction in zip(records, predictions, strict=True)
    ]
