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
RUN chmod +x /entrypoint.sh \
    && pip install --no-cache-dir -e ".[chat]"

ARG EMBED_MODEL=intfloat/multilingual-e5-large
ARG PREFETCH_EMBED=0
ENV EMBED_MODEL=${EMBED_MODEL} \
    DATA_DIR=/data \
    INCOMING_DIR=/incoming \
    PYTHONUNBUFFERED=1 \
    CHAT_API_HOST=0.0.0.0 \
    CHAT_API_PORT=8000

# Bake the embedding model into the image for air-gapped / client pulls.
# First build with PREFETCH_EMBED=1 is slow; skip with 0 for local iteration.
RUN if [ "$PREFETCH_EMBED" = "1" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${EMBED_MODEL}')"; \
    fi

VOLUME ["/data", "/incoming"]
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["rag-chat", "serve", "--host", "0.0.0.0", "--port", "8000"]
