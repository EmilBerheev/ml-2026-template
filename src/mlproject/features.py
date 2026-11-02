"""Признаки и препроцессинг. Одна и та же функция работает при обучении и в сервисе,
поэтому расхождения train/serve (training-serving skew) не возникает."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Доменные признаки. Добавляйте свои сюда и перечисляйте их в config.yaml → features."""
    out = df.copy()
    if {"total_charges", "tenure_months"} <= set(out.columns):
        out["charges_per_month"] = out["total_charges"] / np.maximum(out["tenure_months"], 1)
    return out


def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    f = cfg["features"]
    blocks = []
    if f.get("numeric"):
        blocks.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            list(f["numeric"]),
        ))
    if f.get("categorical"):
        blocks.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                         sparse_output=False)),
            ]),
            list(f["categorical"]),
        ))
    if f.get("text"):
        blocks.append(("text", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2)), f["text"]))
    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=True)


def make_pipeline(cfg: dict[str, Any], estimator: Any) -> Pipeline:
    """Признаки → препроцессинг → модель. Сохраняется целиком в models/model.joblib."""
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor(cfg)),
        ("model", estimator),
    ])


def feature_names(pipeline: Pipeline) -> list[str]:
    try:
        return [str(n) for n in pipeline.named_steps["prep"].get_feature_names_out()]
    except Exception:  # у некоторых трансформеров нет имён признаков
        return []
