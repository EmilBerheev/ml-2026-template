"""Этап 6. Мониторинг, CI для ML и итоговый отчёт (вопросы 35–36)."""

from __future__ import annotations

import json
import re
import shutil

from grader.common import doc_sections, readme_filled
from grader.core import Check, Ctx, Outcome, Skip, fail, ok, part

KEY = "stage6"
TITLE = "Этап 6. Мониторинг и итоги"
QUESTIONS = "35–36"
SUMMARY = "reports/drift_summary.json"
HTML = "reports/drift_report.html"
MON_SECTIONS = ["Что мониторим", "Симуляция сдвига", "Пороги и алерты", "Политика переобучения",
                "Журналы и персональные данные"]


def _summary(ctx: Ctx) -> dict | None:
    s = ctx.json(SUMMARY)
    return None if not s or s.get("__invalid__") else s


def c_report(ctx: Ctx) -> Outcome:
    score, hints = 0.0, []
    html = ctx.path(HTML)
    if html.exists() and html.stat().st_size > 10_000:
        page = html.read_text("utf-8", errors="replace")
        # в HTML Evidently 0.7 нет слова «evidently», зато есть виджеты и раздел про дрейф
        if "drift" in page.lower() and ("root_metric" in page or "widget" in page):
            score += 0.5
        else:
            hints.append(f"{HTML} не похож на отчёт Evidently")
    else:
        hints.append(f"нет {HTML}: uv run python -m mlproject.monitor")
    s = _summary(ctx)
    need = ("drifted_columns", "share_drifted", "columns_checked", "shift")
    if s and all(k in s for k in need):
        score += 0.5
    else:
        hints.append(f"{SUMMARY}: нужны поля {list(need)}")
    return part(score, "; ".join(hints))


def c_detected(ctx: Ctx) -> Outcome:
    s = _summary(ctx)
    if not s:
        return fail(f"нет {SUMMARY}")
    score, hints = 0.0, []
    if s.get("drifted_columns"):
        score += 0.5
    else:
        hints.append("сдвиг не обнаружен — усилите симуляцию или объясните, почему так")
    q = s.get("quality") or {}
    if all(k in q for k in ("reference", "current", "drop")):
        score += 0.5
    else:
        hints.append("нет оценки качества на сдвинутых данных (quality: reference/current/drop)")
    return part(score, "; ".join(hints))


def c_live(ctx: Ctx) -> Outcome:
    if _summary(ctx) is None:
        return fail("сначала сделайте и закоммитьте отчёт о дрейфе (проверка 6.1)")
    if ctx.track == "llm":
        raise Skip("LLM-трек: воспроизводимость мониторинга смотрит преподаватель при проверке")
    good, msg, out_dir, _ = ctx.smoke("train", timeout=300)
    if not good:
        return fail(f"smoke-обучение падает, мониторинг проверить нельзя: {msg}")
    good, msg, out_dir, _ = ctx.smoke("monitor", timeout=300)
    if not good:
        return fail(f"uv run python -m mlproject.monitor --smoke падает: {msg}")
    live = json.loads((out_dir / "reports" / "drift_summary.json").read_text("utf-8"))
    if live.get("drifted_columns"):
        return ok()
    return fail("на smoke-данных скрипт мониторинга не нашёл ни одного сдвига")


def c_doc(ctx: Ctx) -> Outcome:
    return doc_sections(ctx, "docs/monitoring.md", MON_SECTIONS, min_words=20)


def _workflows(ctx: Ctx) -> list[tuple[str, str]]:
    folder = ctx.path(".github/workflows")
    if not folder.exists():
        return []
    return [(p.name, p.read_text("utf-8", errors="replace")) for p in sorted(folder.iterdir())
            if p.suffix in (".yml", ".yaml") and p.stem != "grade"]


def c_ci(ctx: Ctx) -> Outcome:
    flows = _workflows(ctx)
    if not flows:
        return fail("нет .github/workflows/ml-ci.yml (свой CI, кроме grade.yml) — образец в "
                    "manual.md, этап 6")
    best_score, best_hints, best_name = 0.0, [], ""
    import yaml

    for name, text in flows:
        score, hints = 0.0, []
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            hints.append(f"YAML не разбирается: {str(exc)[:80]}")
            data = None
        if isinstance(data, dict):
            score += 2
            triggers = data.get("on", data.get(True))  # PyYAML читает on как True
            trig = json.dumps(triggers, default=str)
            if "push" in trig or "pull_request" in trig:
                score += 2
            else:
                hints.append("запуск on: push / pull_request")
        if "astral-sh/setup-uv" in text:
            score += 2
        else:
            hints.append("uses: astral-sh/setup-uv")
        if re.search(r"uv sync[^\n]*--(locked|frozen)", text):
            score += 2
        else:
            hints.append("uv sync --locked")
        if re.search(r"pytest", text) and (re.search(r"-m\s+['\"]?[^\n]*data", text)
                                           or re.search(r"pytest\s*(tests)?\s*$", text, re.M)
                                           or "test_data" in text):
            score += 2
        else:
            hints.append("pytest с data- и model-тестами")
        if re.search(r"mlproject\.train[^\n]*--smoke", text):
            score += 2
        else:
            hints.append("smoke-обучение python -m mlproject.train --smoke")
        if score > best_score:
            best_score, best_hints, best_name = score, hints, name
    msg = f"{best_name}: добавьте " + ", ".join(best_hints) if best_hints else ""
    return part(best_score / 12, msg)


def c_repro(ctx: Ctx) -> Outcome:
    score, hints = 0.0, []
    if ctx.exists("uv.lock"):
        score += 2
        uv = shutil.which("uv")
        if uv:
            code, out, _ = ctx.run([uv, "lock", "--check"], timeout=120)
            if code == 0:
                score += 2
            else:
                hints.append("uv.lock устарел: выполните uv lock и закоммитьте")
        else:
            score += 2  # без uv проверить нельзя — не штрафуем
    else:
        hints.append("нет uv.lock")
    if ctx.exists(".python-version"):
        score += 1
    else:
        hints.append("нет .python-version")
    if ctx.cfg_get("project", "seed") is not None:
        score += 1
    else:
        hints.append("зафиксируйте project.seed в config.yaml")
    return part(score / 6, "; ".join(hints))


def c_readme(ctx: Ctx) -> Outcome:
    parts = [("Этап 6", 40, 0.4), ("Итоги", 40, 0.3), ("Ограничения и что дальше", 25, 0.3)]
    score, hints = 0.0, []
    for name, words, weight in parts:
        res = readme_filled(ctx, name, words)
        if res.fraction:
            score += weight
        else:
            hints.append(res.hint)
    return part(score, "; ".join(hints))


def notes(ctx: Ctx) -> list[str]:
    return []


CHECKS = [
    Check("6.1", "Отчёт Evidently и drift_summary.json", 10, c_report),
    Check("6.2", "Сдвиг найден, качество на сдвиге измерено", 6, c_detected),
    Check("6.3", "Мониторинг воспроизводится (smoke)", 6, c_live),
    Check("6.4", "docs/monitoring.md: разделы заполнены", 10, c_doc),
    Check("6.5", "Свой CI для ML (.github/workflows/ml-ci.yml)", 12, c_ci),
    Check("6.6", "Воспроизводимость: uv.lock, Python, seed", 6, c_repro),
    Check("6.7", "README: «Этап 6», «Итоги», «Ограничения»", 10, c_readme),
]
