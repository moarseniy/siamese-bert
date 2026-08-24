from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from .codebook import load_codebook
from .config import ClassifierConfig, Settings, validate_bge_reranker_v2_m3
from .excel_io import (
    ExcelInputError,
    build_result_zip,
    load_excel,
    safe_result_name,
    write_predictions,
)
from .pair_tokenizer import PairTokenizer
from .predictor import SurveyPredictor
from .vllm_client import VLLMClient, VLLMError


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    classifier_config = ClassifierConfig.load(
        settings.classifier_config_path,
        threshold_override=settings.threshold_override,
        max_labels_override=settings.max_labels_override,
    )
    validate_bge_reranker_v2_m3(settings.model_config_path)
    codebook = load_codebook(
        settings.codebook_path,
        classifier_config.code_description_format,
    )
    tokenizer = PairTokenizer(
        settings.tokenizer_path,
        classifier_config.max_length,
    )
    client = VLLMClient(
        settings.vllm_base_url,
        settings.vllm_model,
        settings.request_timeout_seconds,
        api_key=settings.vllm_api_key,
    )
    await client.wait_until_ready(settings.startup_timeout_seconds)
    app.state.settings = settings
    app.state.predictor = SurveyPredictor(
        codebook=codebook,
        tokenizer=tokenizer,
        client=client,
        threshold=classifier_config.threshold,
        max_labels=classifier_config.max_labels,
        pair_batch_size=settings.pair_batch_size,
        after_semicolon_prefix=classifier_config.after_semicolon_prefix,
    )
    app.state.codebook_bytes = settings.codebook_path.read_bytes()
    app.state.request_semaphore = asyncio.Semaphore(1)
    try:
        yield
    finally:
        await client.close()


app = FastAPI(
    title="Survey cross-encoder Excel inference",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.post("/predict", response_class=Response)
async def predict(file: UploadFile = File(...)) -> Response:
    settings: Settings = app.state.settings
    filename = file.filename or "responses.xlsx"
    if Path(filename).suffix.lower() != ".xlsx":
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted.")
    content = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    try:
        async with app.state.request_semaphore:
            document = load_excel(
                content,
                text_column=settings.text_column,
                output_column=settings.output_column,
                sheet_name=settings.sheet_name,
                max_rows=settings.max_rows,
                max_uncompressed_bytes=settings.max_uncompressed_xlsx_bytes,
            )
            predictions = await app.state.predictor.predict(document.answers)
            result_name = safe_result_name(filename)
            result_xlsx = write_predictions(document, predictions)
            archive = build_result_zip(
                result_xlsx,
                result_name,
                app.state.codebook_bytes,
            )
    except ExcelInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VLLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    archive_name = f"{Path(result_name).stem}.zip"
    disposition = f"attachment; filename*=UTF-8''{quote(archive_name)}"
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition": disposition},
    )
