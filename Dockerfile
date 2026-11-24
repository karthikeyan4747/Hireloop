# HireLoop remote MCP server (streamable-http) for AWS App Runner.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HIRELOOP_TRANSPORT=streamable-http \
    HIRELOOP_HOST=0.0.0.0 \
    HIRELOOP_PORT=8000 \
    HIRELOOP_DB_PATH=/data/hireloop.db \
    USE_BEDROCK=true

WORKDIR /app
COPY pyproject.toml README.md ./
COPY server ./server
# [aws] adds boto3. Local fastembed is left out: Titan embeddings replace it and the image stays small.
RUN pip install ".[aws]"

RUN useradd --create-home app && mkdir /data && chown app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('HIRELOOP_PORT','8000'), timeout=2)"

# Secrets (GROQ_API_KEY, HIRELOOP_AUTH_TOKEN, ...) are injected as environment variables at runtime.
CMD ["python", "-m", "server.main"]
