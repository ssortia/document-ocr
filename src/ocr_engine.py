from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR

# Латинские символы → кириллические визуальные двойники
_LATIN_TO_CYRILLIC = str.maketrans(
    "ABCEHKMOPTXYaceiopuxy",
    "АВСЕНКМОРТХУасеіорцху",
)


def latin_to_cyrillic(text: str) -> str:
    """Заменяет латинские символы-двойники на кириллические.

    Оставляет нетронутыми цифры, знаки препинания и символы без аналогов.
    """
    return text.translate(_LATIN_TO_CYRILLIC)


@dataclass
class OCRResult:
    """Результат распознавания одной страницы."""

    texts: list[str]
    scores: list[float]

    def filtered(self, min_score: float = 0.5) -> OCRResult:
        """Возвращает результат, отфильтрованный по минимальному confidence."""
        pairs = [(t, s) for t, s in zip(self.texts, self.scores) if s >= min_score]
        if not pairs:
            return OCRResult(texts=[], scores=[])
        texts, scores = zip(*pairs)
        return OCRResult(texts=list(texts), scores=list(scores))


class OCREngine:
    """Обёртка над PaddleOCR для распознавания текста."""

    def __init__(self, lang: str = "ru", use_gpu: bool = True) -> None:
        device = "gpu:0" if use_gpu else "cpu"
        self._ocr = PaddleOCR(
            lang=lang,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize_file(self, file_path: str | Path) -> list[OCRResult]:
        """Распознаёт текст из файла (PDF или изображение).

        PaddleOCR нативно поддерживает PDF — каждая страница
        возвращается отдельным результатом.
        """
        results = self._ocr.predict(str(file_path))
        return [self._parse_result(r) for r in results]

    def recognize_image(self, image: np.ndarray) -> OCRResult:
        """Распознаёт текст на numpy-изображении (после предобработки)."""
        results = self._ocr.predict(image)
        if results:
            return self._parse_result(results[0])
        return OCRResult(texts=[], scores=[])

    @staticmethod
    def _parse_result(result: dict | list) -> OCRResult:
        """Извлекает тексты и scores из результата PaddleOCR."""
        if isinstance(result, dict):
            data = result
        elif isinstance(result, list) and len(result) > 0:
            data = result[0] if isinstance(result[0], dict) else {}
        else:
            return OCRResult(texts=[], scores=[])

        texts = [latin_to_cyrillic(t) for t in data.get("rec_texts", [])]
        scores = data.get("rec_scores", [])
        return OCRResult(texts=texts, scores=scores)
