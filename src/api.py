"""HTTP API для распознавания текста с документов."""

import asyncio
import io
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.ocr_engine import OCREngine
from src.preprocessor import enhance_for_ocr, preprocess_image

ocr_engine: OCREngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_engine
    print("Загрузка моделей PaddleOCR (GPU)...")
    ocr_engine = OCREngine(use_gpu=True)
    print("Все модели загружены, сервер готов к работе.")
    yield


app = FastAPI(title="Document OCR API", lifespan=lifespan)


def _save_to_tmp(contents: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        return tmp.name


def _run_ocr(tmp_path: str | None, image_data: bytes | None, enhance: bool, min_score: float) -> list[dict]:
    """Синхронная функция OCR — выполняется в thread pool."""
    if tmp_path:
        pages = ocr_engine.recognize_file(tmp_path)
    else:
        image = Image.open(io.BytesIO(image_data))
        processed = preprocess_image(image)
        if enhance:
            processed = enhance_for_ocr(processed)
        result = ocr_engine.recognize_image(processed)
        pages = [result]

    output = []
    for page_num, result in enumerate(pages, start=1):
        filtered = result.filtered(min_score)
        output.append({
            "page": page_num,
            "texts": filtered.texts,
            "scores": [round(s, 4) for s in filtered.scores],
        })
    return output


@app.post("/ocr")
async def perform_ocr(
    file: UploadFile = File(...),
    min_score: float = Query(0.5, ge=0.0, le=1.0),
    enhance: bool = Query(True, description="Применить предобработку (CLAHE + бинаризация)"),
):
    """Распознать текст через PP-OCRv5."""
    contents = await file.read()
    suffix = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""

    if suffix == "pdf":
        tmp_path = _save_to_tmp(contents, ".pdf")
        output = await asyncio.to_thread(_run_ocr, tmp_path, None, enhance, min_score)
    else:
        output = await asyncio.to_thread(_run_ocr, None, contents, enhance, min_score)

    return JSONResponse(content={"filename": file.filename, "pages": output})


@app.get("/health")
async def health():
    return {"status": "ok"}
