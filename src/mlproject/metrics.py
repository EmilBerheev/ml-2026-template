"""Метрики, бутстреп-интервалы и калибровка (вопросы 9 и 29)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn import metrics as skm

HIGHER_IS_BETTER = {
    "roc_auc": True, "pr_auc": True, "f1": True, "f1_macro": True, "accuracy": True,
    "balanced_accuracy": True, "recall": True, "precision": True, "r2": True,
    "log_loss": False, "brier": False, "mae": False, "rmse": False, "mape": False,
}
TASK_METRICS = {
    "binary": ["roc_auc", "pr_auc", "f1", "recall", "precision", "balanced_accuracy",
               "accuracy", "log_loss", "brier"],
    "multiclass": ["f1_macro", "accuracy", "balanced_accuracy", "roc_auc", "log_loss"],
    "regression": ["mae", "rmse", "r2", "mape"],
}


def _proba_pos(proba: np.ndarray) -> np.ndarray:
    return proba[:, 1] if proba.ndim == 2 else proba


def metric_fn(name: str, task: str) -> Callable[[np.ndarray, np.ndarray, Any], float]:
    """Функция (y_true, y_pred, proba) -> число."""
    if task == "regression":
        return {
            "mae": lambda y, p, _: skm.mean_absolute_error(y, p),
            "rmse": lambda y, p, _: skm.root_mean_squared_error(y, p),
            "r2": lambda y, p, _: skm.r2_score(y, p),
            "mape": lambda y, p, _: skm.mean_absolute_percentage_error(y, p),
        }[name]
    if task == "binary":
        return {
            "roc_auc": lambda y, p, pr: skm.roc_auc_score(y, _proba_pos(pr)),
            "pr_auc": lambda y, p, pr: skm.average_precision_score(y, _proba_pos(pr)),
            "f1": lambda y, p, _: skm.f1_score(y, p, zero_division=0),
            "recall": lambda y, p, _: skm.recall_score(y, p, zero_division=0),
            "precision": lambda y, p, _: skm.precision_score(y, p, zero_division=0),
            "balanced_accuracy": lambda y, p, _: skm.balanced_accuracy_score(y, p),
            "accuracy": lambda y, p, _: skm.accuracy_score(y, p),
            "log_loss": lambda y, p, pr: skm.log_loss(y, _proba_pos(pr), labels=[0, 1]),
            "brier": lambda y, p, pr: skm.brier_score_loss(y, _proba_pos(pr)),
        }[name]
    return {
        "f1_macro": lambda y, p, _: skm.f1_score(y, p, average="macro", zero_division=0),
        "f1": lambda y, p, _: skm.f1_score(y, p, average="macro", zero_division=0),
        "accuracy": lambda y, p, _: skm.accuracy_score(y, p),
        "balanced_accuracy": lambda y, p, _: skm.balanced_accuracy_score(y, p),
        "roc_auc": lambda y, p, pr: skm.roc_auc_score(y, pr, multi_class="ovr"),
        "log_loss": lambda y, p, pr: skm.log_loss(y, pr),
    }[name]


def compute_all(task: str, y_true, y_pred, proba=None) -> dict[str, float]:
    out = {}
    for name in TASK_METRICS[task]:
        try:
            out[name] = round(float(metric_fn(name, task)(y_true, y_pred, proba)), 6)
        except (ValueError, TypeError):  # например, ROC-AUC без вероятностей
            continue
    return out


def bootstrap_ci(
    name: str, task: str, y_true, y_pred, proba=None,
    n: int = 1000, level: float = 0.95, seed: int = 42,
) -> dict[str, float]:
    """Перцентильный бутстреп: пересэмплируем test с возвращением и пересчитываем метрику."""
    fn = metric_fn(name, task)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    proba = None if proba is None else np.asarray(proba)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            values.append(fn(y_true[idx], y_pred[idx], None if proba is None else proba[idx]))
        except ValueError:  # в выборку попал один класс
            continue
    alpha = (1 - level) / 2
    return {
        "value": round(float(fn(y_true, y_pred, proba)), 6),
        "ci_low": round(float(np.quantile(values, alpha)), 6),
        "ci_high": round(float(np.quantile(values, 1 - alpha)), 6),
        "n_bootstrap": len(values),
        "level": level,
    }


def _take(outputs: tuple, idx: np.ndarray) -> tuple:
    pred, proba = outputs
    return np.asarray(pred)[idx], None if proba is None else np.asarray(proba)[idx]


def paired_delta_ci(
    name: str, task: str, y_true, a: tuple, b: tuple,
    n: int = 1000, level: float = 0.95, seed: int = 42,
) -> dict[str, float]:
    """Разница метрик моделей a и b на одних и тех же бутстреп-выборках (парный бутстреп)."""
    fn = metric_fn(name, task)
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            va = fn(y_true[idx], *_take(a, idx))
            vb = fn(y_true[idx], *_take(b, idx))
        except ValueError:
            continue
        deltas.append(va - vb)
    alpha = (1 - level) / 2
    return {
        "delta": round(float(np.mean(deltas)), 6),
        "ci_low": round(float(np.quantile(deltas, alpha)), 6),
        "ci_high": round(float(np.quantile(deltas, 1 - alpha)), 6),
        "level": level,
    }


def expected_calibration_error(y_true, proba_pos, n_bins: int = 10) -> float:
    """ECE: средняя по корзинам разница «уверенность − частота», с весом по размеру корзины."""
    y_true, proba_pos = np.asarray(y_true), np.asarray(proba_pos)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (proba_pos >= lo) & (proba_pos < hi if hi < 1 else proba_pos <= hi)
        if mask.any():
            ece += mask.mean() * abs(proba_pos[mask].mean() - y_true[mask].mean())
    return round(float(ece), 6)
