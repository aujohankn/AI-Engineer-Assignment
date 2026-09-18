FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip && pip install .

RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /data/jobs && chown -R app:app /data/jobs /app

USER app
EXPOSE 8000
VOLUME ["/data/jobs"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
