"""Общие фикстуры. Всё считается на data/sample.csv — тесты должны идти секунды, а не минуты."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlproject import data as data_mod  # noqa: E402
from mlproject.config import load_config, load_schema  # noqa: E402
from mlproject.data import read_table  # noqa: E402
from mlproject.features import make_pipeline  # noqa: E402
from mlproject.models import make_boosting  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(smoke=True)


@pytest.fixture(scope="session")
def schema():
    return load_schema()


@pytest.fixture(scope="session")
def sample_df(cfg):
    """Сырая выборка ровно в том виде, в каком она лежит в репозитории."""
    return read_table(Path(cfg["_root"]) / cfg["data"]["sample_path"])


@pytest.fixture(scope="session")
def splits(cfg, sample_df):
    df, _ = data_mod.clean(sample_df, cfg)
    return data_mod.split(df, cfg)


@pytest.fixture(scope="session")
def smoke_model(cfg, splits):
    """Маленькая модель на train-части sample: проверяем код обучения, а не рекорд качества."""
    X, y = data_mod.xy(splits["train"], cfg)
    params = {**cfg["models"]["boosting"], "n_estimators": 60}
    return make_pipeline(cfg, make_boosting(cfg["task"]["type"], params, 42)).fit(X, y)


@pytest.fixture(scope="session")
def model_path(cfg, smoke_model, tmp_path_factory):
    """Обученная модель из models/, а если её ещё нет — smoke-модель во временной папке."""
    committed = Path(cfg["_root"]) / "models" / "model.joblib"
    if committed.exists():
        return committed
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(smoke_model, path)
    return path
