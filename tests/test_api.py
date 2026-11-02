"""Тесты сервиса (этап 5). Запуск: uv run pytest -m api

Добавьте минимум два своих теста: 422 на неверный тип или значение вне диапазона,
422 на лишнее поле, необязательное поле без значения, заголовок X-Process-Time-Ms,
одинаковый ответ на одинаковый запрос.
"""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

EXAMPLE = {
    "tenure_months": 5, "monthly_charges": 89.9, "total_charges": 450.0,
    "contract": "month-to-month", "payment_method": "e_wallet", "internet_service": "fiber",
    "region": "south", "support_calls": 3, "num_services": 2,
}


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from mlproject.api import app

    with TestClient(app) as c:  # with — чтобы отработал lifespan и загрузилась модель
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_probability(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"]
