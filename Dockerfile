FROM node:22-alpine AS web-build
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SMALLFORGE_DATA_DIR=/app/data \
    SMALLFORGE_RESULTS_DIR=/app/results
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY --from=web-build /build/frontend/dist ./frontend/dist
RUN pip install --no-cache-dir -e .
RUN useradd --create-home --uid 10001 smallforge && \
    mkdir -p /app/data /app/results && \
    chown -R smallforge:smallforge /app
USER smallforge
EXPOSE 8000
CMD ["uvicorn", "smallforge.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
