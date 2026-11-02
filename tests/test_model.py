"""Тесты модели (этапы 3–4). Запуск: uv run pytest -m model

Шаблонные проверки — минимальные. Добавьте минимум две свои, например:
  * направленный тест: рост support_calls не уменьшает вероятность оттока;
  * качество обученной модели из models/ на sample не ниже порога из этапа 1;
  * качество на важном срезе (например, contract=two-year) не хуже заданного.
"""

import numpy as np
import pytest

from mlproject import data as data_mod
from mlproject.metrics import compute_all
from mlproject.train import predict

pytestmark = pytest.mark.model


def test_probabilities_are_valid(smoke_model, splits, cfg):
    X, _ = data_mod.xy(splits["test"], cfg)
    proba = smoke_model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_smoke_model_beats_dummy(smoke_model, splits, cfg):
    X, y = data_mod.xy(splits["test"], cfg)
    pred, proba = predict(smoke_model, X, cfg)
    score = compute_all(cfg["task"]["type"], y, pred, proba)[cfg["metric"]["primary"]]
    assert score > 0.6, f"{cfg['metric']['primary']}={score:.3f} — модель почти не учится"


def test_prediction_ignores_id(smoke_model, splits, cfg):
    X, _ = data_mod.xy(splits["test"].head(50), cfg)
    shuffled = X.copy()
    id_col = cfg["data"]["id_column"]
    shuffled[id_col] = shuffled[id_col].sample(frac=1, random_state=0).to_numpy()
    np.testing.assert_allclose(smoke_model.predict_proba(X), smoke_model.predict_proba(shuffled))
