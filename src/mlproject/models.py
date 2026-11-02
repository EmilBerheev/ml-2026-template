"""Фабрики моделей этапа 3: dummy → baseline → градиентный бустинг → нейросеть."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from mlproject.nn import make_nn


def make_dummy(task: str):
    """Нижняя планка: «всегда самый частый класс» или «всегда среднее»."""
    return DummyRegressor() if task == "regression" else DummyClassifier(strategy="prior")


def make_baseline(task: str, params: dict[str, Any], seed: int):
    """Простая, но честная модель: логистическая регрессия / Ridge на тех же признаках."""
    if task == "regression":
        return Ridge(alpha=1.0 / params.get("C", 1.0), random_state=seed)
    return LogisticRegression(
        C=params.get("C", 1.0), max_iter=2000, class_weight=params.get("class_weight"),
        random_state=seed,
    )


def make_boosting(task: str, params: dict[str, Any], seed: int):
    """LightGBM по умолчанию. XGBoost/CatBoost подключаются так же (uv add xgboost)."""
    library = params.get("library", "lightgbm")
    if library != "lightgbm":
        raise ValueError(f"добавьте поддержку {library} в models.make_boosting")
    from lightgbm import LGBMClassifier, LGBMRegressor

    kw = {k: v for k, v in params.items() if k not in ("library",)}
    kw.setdefault("n_estimators", 400)
    cls = LGBMRegressor if task == "regression" else LGBMClassifier
    if task == "regression":
        kw.pop("class_weight", None)
    return cls(random_state=seed, verbose=-1, n_jobs=-1, **kw)


def boosting_space(trial) -> dict[str, Any]:
    """Пространство поиска Optuna для LightGBM (вопрос 19)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def make_nn_model(task: str, params: dict[str, Any], seed: int):
    return make_nn(task, params, seed)
