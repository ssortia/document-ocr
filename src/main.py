"""CLI для распознавания текста с PDF-документов и изображений."""

import argparse
import sys
from pathlib import Path

from src.ocr_engine import OCREngine

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"


def process_file(file_path: Path, engine: OCREngine, min_score: float) -> None:
    """Обрабатывает файл через PP-OCRv5."""
    print(f"\n{'=' * 60}")
    print(f"Файл: {file_path.name}")
    print(f"{'=' * 60}")

    pages = engine.recognize_file(file_path)

    for page_num, result in enumerate(pages, start=1):
        print(f"\n--- Страница {page_num} ---")
        filtered = result.filtered(min_score)

        if not filtered.texts:
            print("(текст не распознан)")
            continue

        for text, score in zip(filtered.texts, filtered.scores):
            print(f"  [{score:.2f}] {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Распознавание текста с документов")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Пути к PDF/изображениям. Если не указаны — обрабатывается папка documents/",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Минимальный confidence score (по умолчанию 0.5)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Использовать GPU для распознавания",
    )
    args = parser.parse_args()

    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not files:
        print("Файлы не найдены.", file=sys.stderr)
        sys.exit(1)

    engine = OCREngine(use_gpu=args.gpu)
    for file_path in files:
        if not file_path.exists():
            print(f"Файл не найден: {file_path}", file=sys.stderr)
            continue
        process_file(file_path, engine, min_score=args.min_score)


if __name__ == "__main__":
    main()
