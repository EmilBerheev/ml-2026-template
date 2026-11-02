"""Этап 2. Данные и EDA (вопросы 5–10)."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import PurePosixPath

from grader import md
from grader.common import doc_sections, readme_filled, tests_check
from grader.core import Check, Ctx, Outcome, fail, ok, part

KEY = "stage2"
TITLE = "Этап 2. Данные и EDA"
QUESTIONS = "5–10"
DOC = "docs/data_card.md"
CARD_SECTIONS = ["Источник", "Лицензия", "Состав данных", "Персональные данные", "Очистка",
                 "Известные проблемы"]
# sha256 демо-выборки шаблона: если он совпал — вы ещё не подключили свой датасет
DEMO_SAMPLE_SHA256 = "112ef345c863c7a05a376995db6e9f61207535dea13571aa89d6d7f20b9ed3e1"
LICENSE_HINTS = re.compile(
    r"\bCC[- ]?(BY|0|ZERO)|creative commons|\bODbL|\bODC|\bPDDL|\bMIT\b|apache|\bBSD|\bGPL|"
    r"public domain|общественное достояние|open government|открыт\w* лиценз|"
    r"правила соревнования|competition rules|terms of use|условия использования", re.I)
DOWNLOAD_HINTS = re.compile(
    r"uv run|python |curl |wget |kaggle |huggingface-cli|hf download|load_dataset|fetch_openml|"
    r"openml|ucimlrepo|dvc pull|gdown", re.I)


def _sample(ctx: Ctx):
    rel = ctx.cfg_get("data", "sample_path", default="data/sample.csv")
    return rel, ctx.path(rel)


def c_card(ctx: Ctx) -> Outcome:
    return doc_sections(ctx, DOC, CARD_SECTIONS, min_words=15)


def c_source_license(ctx: Ctx) -> Outcome:
    secs = md.sections(ctx.text(DOC))
    src = md.strip_comments(md.find(secs, "Источник") or "")
    lic = md.strip_comments(md.find(secs, "Лицензия") or "")
    score, hints = 0.0, []
    if re.search(r"https?://|doi\.org|\bsrc/\S+\.py", src):
        score += 0.5
    else:
        hints.append("в «Источнике» нужна ссылка (URL/DOI) на датасет")
    if LICENSE_HINTS.search(lic):
        score += 0.5
    else:
        hints.append("в «Лицензии» назовите лицензию (CC BY 4.0, CC0, ODbL, правила соревнования…)")
    return part(score, "; ".join(hints))


def c_download(ctx: Ctx) -> Outcome:
    body = md.strip_comments(md.find(md.sections(ctx.text(DOC)), "Источник") or "")
    ignore = ctx.text(".gitignore") or ""
    hints = []
    if not DOWNLOAD_HINTS.search(body):
        hints.append("в «Источнике» дайте команду, которой полный датасет скачивается заново")
    full = str(ctx.cfg_get("data", "full_path", default="data/raw/dataset.csv"))
    folder = str(PurePosixPath(full).parent)
    if folder not in ignore and full not in ignore:
        hints.append(f"добавьте {folder}/ в .gitignore — полный датасет не коммитят")
    return part(1 - len(hints) / 2, "; ".join(hints))


def c_sample(ctx: Ctx) -> Outcome:
    rel, path = _sample(ctx)
    if not path.exists():
        return fail(f"нет {rel} — положите выборку до 5 МБ для CI")
    size_mb = path.stat().st_size / 2**20
    if size_mb > 5:
        return fail(f"{rel} весит {size_mb:.1f} МБ, нужно до 5 МБ")
    with open(path, encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    need = [ctx.cfg_get("data", "id_column"), ctx.cfg_get("data", "target")]
    missing = [c for c in need if c and c not in header]
    if rows < 200:
        return fail(f"в {rel} {rows} строк, нужно от 200")
    if missing:
        return fail(f"в {rel} нет колонок {missing} из config.yaml")
    return ok()


def c_schema(ctx: Ctx) -> Outcome:
    schema = ctx.yaml("configs/schema.yaml")
    if not isinstance(schema, dict) or "columns" not in schema:
        return fail("configs/schema.yaml не найден или в нём нет раздела columns")
    rel, path = _sample(ctx)
    if not path.exists():
        return fail(f"нет {rel}")
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = next(csv.reader(fh), [])
    cols = set(schema["columns"])
    if set(header) != cols:
        return fail(f"schema.yaml и {rel} расходятся: нет в схеме {sorted(set(header) - cols)}, "
                    f"нет в данных {sorted(cols - set(header))}")
    return ok()


def c_tests(ctx: Ctx) -> Outcome:
    return tests_check(ctx, "data", 2)


def c_split(ctx: Ctx) -> Outcome:
    summary = ctx.json("reports/split_summary.json")
    if summary is None:
        return fail("нет reports/split_summary.json: запустите uv run python -m mlproject.data "
                    "(на полном датасете) и закоммитьте отчёт")
    if summary.get("__invalid__"):
        return fail("reports/split_summary.json не разбирается")
    hints, score = [], 0.0
    strategy = ctx.cfg_get("split", "strategy")
    if summary.get("smoke"):
        hints.append("split_summary.json от smoke-прогона — пересоберите без --smoke")
    elif summary.get("strategy") != strategy:
        hints.append(f"в отчёте стратегия {summary.get('strategy')}, в config.yaml {strategy}")
    elif not summary.get("leakage", {}).get("ok"):
        hints.append(f"в отчёте утечка: {summary.get('leakage')}")
    else:
        score += 0.5
    good, msg, out_dir, _ = ctx.smoke("data", timeout=180)
    if not good:
        hints.append(f"python -m mlproject.data --smoke падает: {msg}")
    else:
        live = json.loads((out_dir / "reports" / "split_summary.json").read_text("utf-8"))
        if live.get("leakage", {}).get("ok"):
            score += 0.5
        else:
            hints.append(f"повторное разбиение sample даёт утечку: {live.get('leakage')}")
    return part(score, "; ".join(hints))


def c_eda(ctx: Ctx) -> Outcome:
    notebooks = [p for p in ctx.path("notebooks").glob("*") if p.suffix in (".py", ".ipynb")
                 and "eda" in p.name.lower()] if ctx.exists("notebooks") else []
    figs = list(ctx.path("reports/figures").glob("eda_*.png")) if ctx.exists(
        "reports/figures") else []
    score, hints = 0.0, []
    if notebooks:
        score += 1 / 3
    else:
        hints.append("нет ноутбука notebooks/*eda*.py (marimo) или .ipynb")
    if len(figs) >= 3:
        score += 2 / 3
    else:
        hints.append(f"графиков reports/figures/eda_*.png {len(figs)} из 3 — запустите ноутбук "
                     "EDA и закоммитьте картинки")
    return part(score, "; ".join(hints))


def c_readme(ctx: Ctx) -> Outcome:
    return readme_filled(ctx, "Этап 2", 40)


def notes(ctx: Ctx) -> list[str]:
    out = []
    rel, path = _sample(ctx)
    demo = str(ctx.cfg_get("project", "variant", default="")).lower() == "demo"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == DEMO_SAMPLE_SHA256:
        demo = True
    if demo:
        out.append("Вы работаете на демо-датасете шаблона (project.variant: demo или та же "
                   "выборка). Сдать этапы на нём нельзя — подключите свой вариант из lab.md.")
    return out


CHECKS = [
    Check("2.1", "Разделы docs/data_card.md заполнены", 12, c_card),
    Check("2.2", "Источник со ссылкой и лицензия", 4, c_source_license),
    Check("2.3", "Полный датасет скачивается заново, raw не в git", 4, c_download),
    Check("2.4", "data/sample.csv: до 5 МБ, от 200 строк, id и target", 6, c_sample),
    Check("2.5", "configs/schema.yaml описывает все колонки", 4, c_schema),
    Check("2.6", "Data-тесты: свои (≥ 2) и все проходят", 12, c_tests),
    Check("2.7", "Разбиение без утечек (отчёт + повторный прогон)", 8, c_split),
    Check("2.8", "EDA: ноутбук и ≥ 3 графика", 6, c_eda),
    Check("2.9", "README: раздел «Этап 2»", 4, c_readme),
]
