"""Этап 5: сервис предсказаний на FastAPI (вопрос 34).

    uv run uvicorn mlproject.api:app --reload --port 8000
    открыть http://127.0.0.1:8000/docs

Контракт, который проверяет грейдер:
    GET  /health  → 200 {"status": "ok", "model_loaded": true, "model_version": "..."}
    POST /predict → 200 {"prediction": ..., "probability": ..., "model_version": "..."}
                    422 на неверный вход (это делает Pydantic)

Поля запроса описаны в классе Features — перепишите его под свой датасет.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("mlproject.api")


class Features(BaseModel):
    """Один клиент. Ограничения полей = контракт данных из configs/schema.yaml."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{
            "tenure_months": 5, "monthly_charges": 89.9, "total_charges": 450.0,
            "contract": "month-to-month", "payment_method": "e_wallet",
            "internet_service": "fiber", "region": "south", "support_calls": 3,
            "num_services": 2,
        }]},
    )

    tenure_months: int = Field(ge=0, le=240, description="Сколько месяцев клиент с нами")
    monthly_charges: float = Field(ge=0, le=1000, description="Платёж в месяц")
    total_charges: float | None = Field(default=None, ge=0, description="Сумма всех платежей")
    contract: Literal["month-to-month", "one-year", "two-year"]
    payment_method: Literal["card", "bank_transfer", "e_wallet", "cash"]
    internet_service: Literal["dsl", "fiber", "none"]
    region: Literal["center", "north", "south", "east", "west"]
    support_calls: int = Field(ge=0, le=100)
    num_services: int = Field(ge=0, le=20)


class Prediction(BaseModel):
    prediction: int | float | str
    probability: float | None = Field(default=None, description="Вероятность положительного класса")
    model_version: str


def _model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", "models/model.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = _model_path()
    app.state.model, app.state.meta = None, {}
    if path.exists():
        app.state.model = joblib.load(path)
        meta_path = path.with_name("model_meta.json")
        if meta_path.exists():
            app.state.meta = json.loads(meta_path.read_text("utf-8"))
        log.info("модель загружена: %s", path)
    else:
        log.error("нет файла модели %s — сначала обучите: python -m mlproject.train", path)
    yield


app = FastAPI(title="ML-сервис учебного проекта", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


@app.get("/health")
def health() -> dict:
    loaded = app.state.model is not None
    if not loaded:
        raise HTTPException(status_code=503, detail="модель не загружена")
    return {"status": "ok", "model_loaded": loaded,
            "model_version": app.state.meta.get("model_version", "unknown")}


@app.post("/predict", response_model=Prediction)
def predict(item: Features) -> Prediction:
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="модель не загружена")
    start = time.perf_counter()
    row = pd.DataFrame([item.model_dump()])
    # пустое необязательное поле (None) превращаем в NaN — его заполнит импьютер из пайплайна
    row = row.astype({c: float for c in row.columns if row[c].isna().all()})
    meta = app.state.meta
    probability = None
    if hasattr(model, "predict_proba") and meta.get("task", "binary") == "binary":
        proba = model.predict_proba(row)[0]
        probability = float(proba[1])
        threshold = meta.get("decision_threshold") or 0.5
        label = model.classes_[1] if probability >= threshold else model.classes_[0]
    else:
        label = model.predict(row)[0]
    label = label.item() if hasattr(label, "item") else label
    result = Prediction(prediction=label, probability=probability,
                        model_version=meta.get("model_version", "unknown"))
    _log_prediction(item, result, (time.perf_counter() - start) * 1000)
    return result


def _log_prediction(item: Features, result: Prediction, latency_ms: float) -> None:
    """Журнал для мониторинга (этап 6): признаки без персональных данных, ответ, задержка."""
    path = os.environ.get("PREDICTION_LOG")
    if not path:
        return
    record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **item.model_dump(),
              "prediction": result.prediction, "probability": result.probability,
              "latency_ms": round(latency_ms, 2)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
