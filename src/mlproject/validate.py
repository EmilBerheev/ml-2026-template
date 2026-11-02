"""Проверка таблицы по контракту configs/schema.yaml.

Каждая функция возвращает список ошибок (пустой — всё хорошо). Так их удобно вызывать
и из pytest, и из кода перед обучением.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

NUMERIC = {"int", "float"}


def check_columns(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    expected = set(schema["columns"])
    actual = set(df.columns)
    errors = []
    if missing := sorted(expected - actual):
        errors.append(f"нет колонок: {missing}")
    if extra := sorted(actual - expected):
        errors.append(f"колонки не описаны в schema.yaml: {extra}")
    return errors


def check_types(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    errors = []
    for col, rule in schema["columns"].items():
        if col not in df:
            continue
        s = df[col].dropna()
        dtype = rule.get("dtype")
        if dtype in NUMERIC and not pd.api.types.is_numeric_dtype(s):
            errors.append(f"{col}: ожидали число, получили {df[col].dtype}")
        elif dtype == "int" and len(s) and not (s == s.round()).all():
            errors.append(f"{col}: ожидали целые числа")
        elif dtype == "date" and pd.to_datetime(s, errors="coerce").isna().any():
            errors.append(f"{col}: есть значения, которые не разбираются как дата")
    return errors


def check_nulls(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    errors = []
    for col, rule in schema["columns"].items():
        if col not in df:
            continue
        share = float(df[col].isna().mean())
        if rule.get("nullable", True) is False and share > 0:
            errors.append(f"{col}: пропуски недопустимы, а их {share:.1%}")
        limit = rule.get("max_missing")
        if limit is not None and share > limit:
            errors.append(f"{col}: пропусков {share:.1%}, допустимо до {limit:.0%}")
    return errors


def check_unique(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    errors = []
    for col, rule in schema["columns"].items():
        if col in df and rule.get("unique") and df[col].duplicated().any():
            errors.append(f"{col}: {int(df[col].duplicated().sum())} повторов")
    return errors


def check_ranges(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    errors = []
    for col, rule in schema["columns"].items():
        if col not in df:
            continue
        s = df[col].dropna()
        if rule.get("dtype") == "date":
            s = pd.to_datetime(s, errors="coerce").dropna()
            lo = pd.Timestamp(rule["min"]) if "min" in rule else None
            hi = pd.Timestamp(rule["max"]) if "max" in rule else None
        else:
            lo, hi = rule.get("min"), rule.get("max")
        if lo is not None and len(s) and s.min() < lo:
            errors.append(f"{col}: минимум {s.min()} меньше {lo}")
        if hi is not None and len(s) and s.max() > hi:
            errors.append(f"{col}: максимум {s.max()} больше {hi}")
        allowed = rule.get("allowed")
        if allowed is not None:
            bad = sorted(set(s.unique()) - set(allowed), key=str)
            if bad:
                errors.append(f"{col}: недопустимые значения {bad[:5]}")
    return errors


def check_table(df: pd.DataFrame, schema: dict[str, Any], target: str | None = None) -> list[str]:
    errors = []
    table = schema.get("table", {})
    if len(df) < table.get("min_rows", 0):
        errors.append(f"строк {len(df)}, нужно не меньше {table['min_rows']}")
    share_rule = table.get("target_share")
    if share_rule and target and target in df:
        share = float(df[target].mean())
        if not share_rule["min"] <= share <= share_rule["max"]:
            errors.append(f"доля положительного класса {share:.3f} вне [{share_rule['min']}, "
                          f"{share_rule['max']}]")
    return errors


def validate(df: pd.DataFrame, schema: dict[str, Any], target: str | None = None) -> list[str]:
    return [
        *check_columns(df, schema),
        *check_types(df, schema),
        *check_nulls(df, schema),
        *check_unique(df, schema),
        *check_ranges(df, schema),
        *check_table(df, schema, target),
    ]
