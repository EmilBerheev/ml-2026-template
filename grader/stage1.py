"""Этап 1. Постановка задачи (вопросы 3–4)."""

from __future__ import annotations

import csv
import re

from grader import md
from grader.common import METRICS_BY_TASK, doc_sections, number, readme_filled
from grader.core import Check, Ctx, Outcome, fail, ok, part

KEY = "stage1"
TITLE = "Этап 1. Постановка задачи"
QUESTIONS = "3–4"
DOC = "docs/problem.md"
TEXT_SECTIONS = ["Бизнес-задача", "Пользователь и решение", "Признаки", "Цена ошибок",
                 "Ограничения и риски", "Жизненный цикл"]


def passport(ctx: Ctx) -> dict[str, str]:
    return md.kv(md.find(md.sections(ctx.text(DOC)), "Паспорт задачи"))


def _need_cfg(ctx: Ctx) -> None:
    if ctx.cfg is None:
        raise ValueError(ctx.cfg_error)


def c_sections(ctx: Ctx) -> Outcome:
    return doc_sections(ctx, DOC, TEXT_SECTIONS, min_words=20)


def c_task_type(ctx: Ctx) -> Outcome:
    _need_cfg(ctx)
    raw = passport(ctx).get("тип задачи", "")
    value = raw.lower()
    if not md.value_ok(raw):
        return fail(f"{DOC}, «Паспорт задачи»: заполните «Тип задачи» (binary, multiclass, "
                    "regression; в LLM-треке — generation)")
    if value.split()[0] not in METRICS_BY_TASK:
        return fail(f"«Тип задачи: {value}» — используйте binary, multiclass, regression "
                    "или generation")
    if value.split()[0] != ctx.task:
        return fail(f"в паспорте {value}, а в configs/config.yaml task.type={ctx.task}")
    return ok()


def c_target(ctx: Ctx) -> Outcome:
    _need_cfg(ctx)
    value = passport(ctx).get("целевая переменная", "")
    target = ctx.cfg_get("data", "target")
    if not md.value_ok(value):
        return fail(f"{DOC}: заполните «Целевая переменная» (имя колонки)")
    if target not in re.findall(r"[\w.]+", value):
        return fail(f"в паспорте «{value}», а в config.yaml data.target={target}")
    sample = ctx.path(ctx.cfg_get("data", "sample_path", default="data/sample.csv"))
    if sample.exists():
        with open(sample, encoding="utf-8", errors="replace") as fh:
            header = next(csv.reader(fh), [])
        if target not in header:
            return fail(f"колонки {target} нет в {sample.name}")
    return ok()


def c_metric(ctx: Ctx) -> Outcome:
    _need_cfg(ctx)
    pp = passport(ctx)
    primary_raw = pp.get("основная метрика", "")
    primary = primary_raw.lower()
    secondary = pp.get("вторичные метрики", "")
    cfg_metric = str(ctx.cfg_get("metric", "primary", default="")).lower()
    allowed = METRICS_BY_TASK.get(ctx.task, set())
    score, hints = 0.0, []
    if not md.value_ok(primary_raw):
        hints.append("заполните «Основная метрика»")
    elif cfg_metric not in re.findall(r"[a-z_0-9]+", primary):
        hints.append(f"основная метрика в паспорте «{primary}», в config.yaml {cfg_metric}")
    elif cfg_metric not in allowed:
        hints.append(f"{cfg_metric} не подходит для {ctx.task}: {sorted(allowed)}")
    else:
        score += 0.75
    if md.value_ok(secondary):
        score += 0.25
    else:
        hints.append("заполните «Вторичные метрики»")
    return part(score, "; ".join(hints))


def c_threshold(ctx: Ctx) -> Outcome:
    _need_cfg(ctx)
    value = passport(ctx).get("порог успеха", "")
    cfg_thr = ctx.cfg_get("metric", "threshold")
    got = number(value) if md.value_ok(value) else None
    if got is None:
        return fail(f"{DOC}: «Порог успеха» — число, например «roc_auc ≥ 0.80»")
    if cfg_thr is None or abs(float(cfg_thr) - got) > 1e-9:
        return fail(f"в паспорте {got}, а в config.yaml metric.threshold={cfg_thr}")
    return ok()


def c_baseline(ctx: Ctx) -> Outcome:
    value = passport(ctx).get("baseline", "")
    if md.value_ok(value) and md.words(value) >= 3:
        return ok()
    return fail(f"{DOC}: опишите «Baseline» — какую простую модель вы будете обгонять")


def c_scenario(ctx: Ctx) -> Outcome:
    raw = passport(ctx).get("сценарий использования", "")
    value = raw.lower()
    if md.value_ok(raw) and re.search(r"batch|online|stream|пакет|онлайн|поток|реальн", value):
        return ok()
    return fail(f"{DOC}: «Сценарий использования» — batch, online или streaming и как часто")


def c_readme_header(ctx: Ctx) -> Outcome:
    head = md.kv((ctx.text("README.md") or "").split("\n## ")[0])
    need = ["студент", "группа", "вариант"]
    missing = [k for k in need if not md.value_ok(head.get(k))]
    if not missing:
        return ok()
    return part(1 - len(missing) / len(need), "README.md, шапка: заполните " +
                ", ".join(f"«{k.capitalize()}»" for k in missing))


def c_readme(ctx: Ctx) -> Outcome:
    return readme_filled(ctx, "Этап 1", 30)


def c_ai_usage(ctx: Ctx) -> Outcome:
    text = ctx.text("AI_USAGE.md")
    if text is None:
        return fail("нет AI_USAGE.md")
    clean = md.strip_comments(text)
    # отдельная строка «Не использовал(а)» — а не фраза из инструкции шаблона
    if re.search(r"^\s*[«\"]?не\s+использовал\w*(\(а\))?[»\"]?\.?\s*$", clean, re.I | re.M):
        return ok()
    rows = [r for r in md.table_rows(clean) if any(c for c in r)]
    real = [r for r in rows if not md.PLACEHOLDER.search(" ".join(r))]
    if real:
        return ok()
    return fail("AI_USAGE.md: добавьте строку в таблицу или напишите «Не использовал(а)»")


CHECKS = [
    Check("1.1", "Разделы docs/problem.md заполнены", 12, c_sections),
    Check("1.2", "Тип задачи совпадает с config.yaml", 6, c_task_type),
    Check("1.3", "Целевая переменная есть в данных", 8, c_target),
    Check("1.4", "Основная и вторичные метрики", 8, c_metric),
    Check("1.5", "Числовой порог успеха = metric.threshold", 6, c_threshold),
    Check("1.6", "Baseline описан", 4, c_baseline),
    Check("1.7", "Сценарий использования (batch/online)", 4, c_scenario),
    Check("1.8", "Шапка README: студент, группа, вариант", 4, c_readme_header),
    Check("1.9", "README: раздел «Этап 1»", 4, c_readme),
    Check("1.10", "Журнал AI_USAGE.md", 4, c_ai_usage),
]
