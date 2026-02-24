FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3-pip \
        libgomp1 libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY paddlepaddle_gpu-3.2.1-cp310-cp310-linux_x86_64.whl .
RUN pip3 install --no-cache-dir paddlepaddle_gpu-3.2.1-cp310-cp310-linux_x86_64.whl && \
    rm paddlepaddle_gpu-3.2.1-cp310-cp310-linux_x86_64.whl

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ src/

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
