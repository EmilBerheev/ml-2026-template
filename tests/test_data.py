"""Тесты данных (этап 2). Запуск: uv run pytest -m data

Шаблонные проверки идут по контракту configs/schema.yaml. Добавьте минимум две свои —
про ваш домен: например, «сумма платежей не меньше месячного платежа»,
«дата регистрации не в будущем», «доля класса в train и test отличается не больше чем на 3 п.п.».
"""

import pytest

from mlproject import validate
from mlproject.data import leakage_report

pytestmark = pytest.mark.data


def test_columns_match_schema(sample_df, schema):
    assert validate.check_columns(sample_df, schema) == []


def test_types_match_schema(sample_df, schema):
    assert validate.check_types(sample_df, schema) == []


def test_no_unexpected_missing(sample_df, schema):
    assert validate.check_nulls(sample_df, schema) == []


def test_ids_are_unique(sample_df, schema):
    assert validate.check_unique(sample_df, schema) == []


def test_values_in_allowed_ranges(sample_df, schema):
    assert validate.check_ranges(sample_df, schema) == []


def test_splits_do_not_leak(splits, cfg):
    report = leakage_report(splits, cfg)
    assert report["ok"], report
