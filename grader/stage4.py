"""Этап 4. Оценка и интерпретация (вопросы 9, 20, 29)."""

from __future__ import annotations

import csv

from grader import md
from grader.common import doc_sections, tests_check
from grader.core import Check, Ctx, Outcome, fail, ok, part

KEY = "stage4"
TITLE = "Этап 4. Оценка и интерпретация"
QUESTIONS = "9, 20, 29"
EVAL = "reports/evaluation.json"
CARD = "docs/model_card.md"
CARD_SECTIONS = ["Назначение", "Данные", "Метрики", "Интерпретация", "Ограничения",
                 "Этические риски", "Как использовать"]


def _eval(ctx: Ctx) -> dict | None:
    ev = ctx.json(EVAL)
    return None if not ev or ev.get("__invalid__") else ev


def _rows(ctx: Ctx, rel: str) -> list[dict]:
    p = ctx.path(rel)
    if not p.exists():
        return []
    with open(p, encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def _ci_ok(rec: dict | None) -> bool:
    if not isinstance(rec, dict):
        return False
    try:
        v, lo, hi = float(rec["value"]), float(rec["ci_low"]), float(rec["ci_high"])
    except (KeyError, TypeError, ValueError):
        return False
    return lo <= v <= hi and lo < hi and int(rec.get("n_bootstrap", 0)) >= 200


def c_ci(ctx: Ctx) -> Outcome:
    ev = _eval(ctx)
    if ev is None:
        return fail(f"нет {EVAL}: запустите uv run python -m mlproject.evaluate")
    metrics = ctx.json("reports/metrics.json") or {}
    best, base = metrics.get("best", ev.get("best")), metrics.get("baseline", "baseline")
    models = ev.get("models", {})
    score, hints = 0.0, []
    if _ci_ok(models.get(best)):
        score += 0.6
    else:
        hints.append(f"нет корректного ДИ для лучшей модели {best} (value, ci_low, ci_high, "
                     "n_bootstrap ≥ 200)")
    if _ci_ok(models.get(base)):
        score += 0.2
    else:
        hints.append(f"нет ДИ для baseline {base}")
    d = ev.get("delta_best_vs_baseline") or {}
    if all(k in d for k in ("delta", "ci_low", "ci_high")):
        score += 0.2
    else:
        hints.append("нет интервала для разницы best − baseline (парный бутстреп)")
    return part(score, "; ".join(hints))


def c_calibration(ctx: Ctx) -> Outcome:
    if ctx.track == "llm":
        ev = ctx.json("reports/llm_eval.json") or {}
        cats = {c.get("category") for c in ev.get("cases", []) if isinstance(c, dict)}
        need = {"injection", "out_of_scope"}
        if len(cats) >= 3 and cats & need:
            return ok()
        return fail("LLM-трек: в llm_eval.json нужны ≥ 3 категории кейсов, среди них injection "
                    "или out_of_scope")
    ev = _eval(ctx)
    if ev is None:
        return fail(f"нет {EVAL}")
    figs = ctx.path("reports/figures")
    if ctx.task == "regression":
        if (figs / "residuals.png").exists():
            return ok()
        return fail("для регрессии нужен график остатков reports/figures/residuals.png")
    cal = ev.get("calibration") or {}
    hints = []
    if cal.get("ece") is None:
        hints.append("в evaluation.json нет calibration.ece")
    if ctx.task == "binary" and cal.get("brier") is None:
        hints.append("нет calibration.brier")
    if not (figs / "calibration.png").exists():
        hints.append("нет reports/figures/calibration.png")
    return ok() if not hints else part(1 - len(hints) / 3, "; ".join(hints))


def c_importance(ctx: Ctx) -> Outcome:
    if ctx.track == "llm":
        ev = ctx.json("reports/llm_eval.json") or {}
        summary = {k.lower() for k in (ev.get("summary") or {})}
        if any(k.startswith(("hit_rate", "recall@", "context_precision", "context_recall", "mrr"))
               for k in summary):
            return ok()
        return fail("LLM-трек: в llm_eval.json → summary нужна метрика поиска (hit_rate@k, "
                    "context_precision, MRR)")
    rows = _rows(ctx, "reports/feature_importance.csv")
    hints = []
    if len(rows) < 5:
        hints.append(f"в reports/feature_importance.csv {len(rows)} строк, нужно от 5")
    elif not {"feature", "importance"} <= set(rows[0]):
        hints.append("в feature_importance.csv нужны колонки feature и importance")
    if not ctx.path("reports/figures/feature_importance.png").exists():
        hints.append("нет reports/figures/feature_importance.png")
    return ok() if not hints else part(0.5 if len(hints) == 1 else 0, "; ".join(hints))


def c_errors(ctx: Ctx) -> Outcome:
    rows = _rows(ctx, "reports/errors.csv")
    score, hints = 0.0, []
    if len(rows) >= 20:
        score += 0.5
    else:
        hints.append(f"в reports/errors.csv {len(rows)} ошибок, разберите от 20")
    good, why = md.filled(md.find(md.sections(ctx.text(CARD)), "Анализ ошибок"), 40)
    if good:
        score += 0.5
    else:
        hints.append(f"{CARD}, «Анализ ошибок»: {why}")
    return part(score, "; ".join(hints))


def c_card(ctx: Ctx) -> Outcome:
    out = doc_sections(ctx, CARD, CARD_SECTIONS, min_words=15)
    text = ctx.text(CARD) or ""
    if "auto:metrics:start" in text and "| Модель |" not in text:
        return part(min(out.fraction, 0.8), (out.hint + "; " if out.hint else "") +
                    "блок метрик пуст — запустите mlproject.evaluate")
    return out


def c_tests(ctx: Ctx) -> Outcome:
    return tests_check(ctx, "model", 2)


CHECKS = [
    Check("4.1", "Метрики с доверительными интервалами", 10, c_ci),
    Check("4.2", "Калибровка: Brier, ECE, график (LLM: категории evals)", 8, c_calibration),
    Check("4.3", "Интерпретация: SHAP/permutation (LLM: метрики поиска)", 8, c_importance),
    Check("4.4", "Анализ ошибок: ≥ 20 примеров и выводы", 8, c_errors),
    Check("4.5", "Model card: разделы заполнены", 14, c_card),
    Check("4.6", "Model-тесты: свои (≥ 2) и все проходят", 12, c_tests),
]
