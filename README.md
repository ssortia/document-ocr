# document-ocr

Тестовый проект для распознавания рукописного текста с документов с использованием [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## Стек

- Python 3.10
- [PaddlePaddle GPU](https://github.com/PaddlePaddle/PaddlePaddle) 3.2.1 (CUDA 12.6)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv5
- FastAPI + Uvicorn
- Docker (база: `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`)

## Запуск

```bash
# Сборка и запуск сервиса в фоне
docker compose build
docker compose up -d

# Распознавание через API
curl -X POST http://localhost:8000/ocr -F "file=@documents/test1.png"
curl -X POST "http://localhost:8000/ocr?min_score=1" -F "file=@documents/test1.png"

# Без предобработки изображения
curl -X POST "http://localhost:8000/ocr?enhance=false" -F "file=@documents/test1.png"

# Остановка
docker compose down
```

## API

`POST /ocr` — распознать текст из PDF или изображения.

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `file` | multipart | — | PDF или изображение |
| `min_score` | float | `0.5` | Минимальный confidence score (0.0–1.0) |
| `enhance` | bool | `true` | CLAHE + бинаризация Otsu перед распознаванием |

Ответ:
```json
{
   "filename":"test1.png",
   "pages":[
      {
         "page":1,
         "texts":[
            "Дворец Н. Д. Алфераки — памятник архитектуры федерального значения в городе Таганроге Ростовской",
            "области. В нём находится музей — филиал Таганрогского государственного литературного и историко-",
            "архитектурного музея-заповедника.",
            "Здание построено в 1848 году. Принадлежало кфупному таганрогсхому домовладельцу Н. Д. Алфераки.",
            "Автором проекта был профессор Петербурской академии художеств, архитектор Андрей Иванович",
            "Штакеншнейдер. Князь М. Г. Голицын, побывавший в Таганроге в 1857 году, писал:"
         ],
         "scores":[
            0.9902,
            0.9917,
            0.9967,
            0.982,
            0.9923,
            0.9882
         ]
      }
   ]
}
```
