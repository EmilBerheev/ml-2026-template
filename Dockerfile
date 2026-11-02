# syntax=docker/dockerfile:1
# Этап 5: образ сервиса. Сборка и запуск:
#   docker build -t mlproject:latest .
#   docker run --rm -p 8000:8000 mlproject:latest
#   curl http://127.0.0.1:8000/health

FROM python:3.14-slim

# LightGBM использует OpenMP: в slim-образе нет libgomp, без неё модель не загрузится
# (ошибка «libgomp.so.1: cannot open shared object file» при старте сервиса)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# uv из официального образа Astral: тот же lock-файл, что и у вас локально
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# 1) Зависимости отдельным слоем: пересобираются, только когда меняется uv.lock.
#    Если лучшая модель — PyTorch, добавьте --group torch (или экспортируйте её в ONNX).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# 2) Код, конфиги и обученная модель
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib

# TODO (этап 5): контейнер не должен работать от root — создайте пользователя и переключитесь
#   на него инструкцией USER.
# TODO (этап 5): добавьте HEALTHCHECK, который дёргает /health (curl в slim-образе нет,
#   используйте python -c "import urllib.request; ...").

EXPOSE 8000
CMD ["uvicorn", "mlproject.api:app", "--host", "0.0.0.0", "--port", "8000"]
