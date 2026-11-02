"""Общие проверки, которые встречаются в нескольких этапах."""

from __future__ import annotations

import re

from grader import md
from grader.core import Ctx, Outcome, fail, ok, part

# Тесты, которые уже лежат в шаблоне. Свои тесты — всё, что не из этого списка.
TEMPLATE_TESTS = {
    "data": {"test_columns_match_schema", "test_types_match_schema", "test_no_unexpected_missing",
             "test_ids_are_unique", "test_values_in_allowed_ranges", "test_splits_do_not_leak"},
    "model": {"test_probabilities_are_valid", "test_smoke_model_beats_dummy",
              "test_prediction_ignores_id"},
    "api": {"test_health", "test_predict_returns_probability"},
}

METRICS_BY_TASK = {
    "binary": {"roc_auc", "pr_auc", "f1", "recall", "precision", "balanced_accuracy", "accuracy",
               "log_loss", "brier"},
    "multiclass": {"f1_macro", "f1", "accuracy", "balanced_accuracy", "roc_auc", "log_loss"},
    "regression": {"mae", "rmse", "r2", "mape"},
    # LLM-трек (RAG): метрики из evals
    "generation": {"pass_rate", "answer_accuracy", "faithfulness", "answer_relevancy",
                   "context_precision", "hit_rate"},
}
LOWER_IS_BETTER = {"mae", "rmse", "mape", "log_loss", "brier"}


def readme_section(ctx: Ctx, name: str) -> str | None:
    return md.find(md.sections(ctx.text("README.md")), name)


def readme_filled(ctx: Ctx, name: str, min_words: int = 40) -> Outcome:
    good, why = md.filled(readme_section(ctx, name), min_words)
    return ok() if good else fail(f"README.md, раздел «{name}»: {why}")


def doc_sections(ctx: Ctx, rel: str, names: list[str], min_words: int = 20) -> Outcome:
    text = ctx.text(rel)
    if text is None:
        return fail(f"нет файла {rel}")
    secs = md.sections(text)
    bad = []
    for name in names:
        good, why = md.filled(md.find(secs, name), min_words)
        if not good:
            bad.append(f"«{name}» — {why}")
    done = len(names) - len(bad)
    if not bad:
        return ok()
    return part(done / len(names), f"{rel}: " + "; ".join(bad))


def tests_check(ctx: Ctx, group: str, min_own: int = 2) -> Outcome:
    res = ctx.tests()
    if isinstance(res, str):
        return fail(res)
    cases = res.get(group, [])
    if not cases:
        return fail(f"не нашёл тестов в tests/test_{group}*.py")
    failed = [c for c in cases if c.outcome == "failed"]
    if failed:
        names = ", ".join(f"{c.name} ({c.message})" if c.message else c.name for c in failed[:4])
        return fail(f"падают тесты: {names}")
    template = TEMPLATE_TESTS.get(group, set())
    own = {c.base for c in cases if c.base not in template and c.outcome == "passed"}
    if len(own) >= min_own:
        return ok()
    hint = (f"все {len(cases)} тестов проходят, но своих (не из шаблона) {len(own)} из "
            f"{min_own}: допишите тесты в tests/test_{group}.py")
    return part(len(own) / min_own, hint)


def higher_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def number(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    return float(m.group(0).replace(",", ".")) if m else None
