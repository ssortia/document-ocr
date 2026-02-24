import cv2
import numpy as np
from PIL import Image


def preprocess_image(
    image: Image.Image,
    compress_scale: float = 1.0,
) -> np.ndarray:
    """Подготовка изображения для OCR: конвертация в RGB и опциональное сжатие.

    Args:
        image: исходное PIL-изображение.
        compress_scale: коэффициент масштабирования (1.0 = без сжатия).
    """
    image = image.convert("RGB")

    if compress_scale < 1.0:
        width, height = image.size
        new_size = (int(width * compress_scale), int(height * compress_scale))
        image = image.resize(new_size, Image.LANCZOS)

    return np.array(image)


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Улучшение изображения для OCR: контраст, шумоподавление, бинаризация.

    Пайплайн:
        1. Grayscale
        2. CLAHE (адаптивное выравнивание гистограммы)
        3. Гауссово размытие (шумоподавление)
        4. Бинаризация Otsu
        5. Возврат в 3-канальный формат (PaddleOCR ожидает RGB)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
