FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
        tesseract-ocr-eng \
        poppler-utils \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/prefetch_embed.py /prefetch_embed.py
RUN chmod +x /entrypoint.sh \
    && pip install --no-cache-dir -e ".[chat]"

ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
ARG PREFETCH_EMBED=1
ENV EMBED_MODEL=${EMBED_MODEL} \
    DATA_DIR=/data \
    INCOMING_DIR=/incoming \
    PYTHONUNBUFFERED=1 \
    CHAT_API_HOST=0.0.0.0 \
    CHAT_API_PORT=8000 \
    HF_HUB_DISABLE_TELEMETRY=1

# Bake FastEmbed weights into the image. First build is slow; PREFETCH_EMBED=0 to skip.
RUN if [ "$PREFETCH_EMBED" = "1" ]; then python /prefetch_embed.py; fi

VOLUME ["/data", "/incoming"]
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["rag-chat", "serve", "--host", "0.0.0.0", "--port", "8000"]
