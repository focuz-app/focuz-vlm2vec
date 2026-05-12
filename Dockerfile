FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-server.txt ./
RUN grep -Ev '^(flash-attn|torch|torchvision)([[:space:]]|$)' requirements.txt > /tmp/requirements-base.txt \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements-base.txt \
    && python -m pip install --no-build-isolation flash-attn \
    && python -m pip install -r requirements-server.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

