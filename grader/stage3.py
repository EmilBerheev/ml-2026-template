"""Этап 3. Baseline и модели (вопросы 11–19, 21–30)."""

from __future__ import annotations

import csv
import json

from grader import md
from grader.common import higher_is_better, readme_section
from grader.core import Check, Ctx, Outcome, Skip, fail, ok, part

KEY = "stage3"
TITLE = "Этап 3. Baseline и модели"
QUESTIONS = "11–19, 21–30"
METRICS = "reports/metrics.json"
NN_FRAMEWORKS = {"torch", "pytorch", "transformers", "peft", "lightning", "timm"}
BOOSTING = {"lightgbm", "xgboost", "catboost", "sklearn"}


def load(ctx: Ctx) -> dict:
    m = ctx.json(METRICS)
    if m is None:
        raise ValueError(f"нет {METRICS}: запустите uv run python -m mlproject.train")
    if m.get("__invalid__"):
        raise ValueError(f"{METRICS} не разбирается как JSON")
    return m


def _metric(rec: dict, part_name: str, metric: str) -> float | None:
    v = (rec.get(part_name) or {}).get(metric)
    return float(v) if isinstance(v, (int, float)) else None


def _csv_rows(ctx: Ctx, rel: str) -> tuple[list[str], list[dict]]:
    p = ctx.path(rel)
    if not p.exists():
        return [], []
    with open(p, encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def c_structure(ctx: Ctx) -> Outcome:
    try:
        m = load(ctx)
    except ValueError as exc:
        return fail(str(exc))
    need = ["task", "primary_metric", "models", "best", "baseline"]
    missing = [k for k in need if k not in m]
    if missing:
        return fail(f"{METRICS}: нет полей {missing} (контракт — в lab.md)")
    metric = m["primary_metric"]
    if metric != ctx.cfg_get("metric", "primary"):
        return fail(f"в metrics.json основная метрика {metric}, в config.yaml "
                    f"{ctx.cfg_get('metric', 'primary')}")
    if m.get("smoke"):
        return fail("metrics.json получен smoke-прогоном — запустите обучение без --smoke")
    broken = [n for n, r in m["models"].items() if _metric(r, "test", metric) is None]
    if broken:
        return fail(f"у моделей {broken} нет test.{metric}")
    return ok()


def c_baseline(ctx: Ctx) -> Outcome:
    m = load(ctx)
    name = m.get("baseline")
    if name in m.get("models", {}):
        return ok()
    return fail("в metrics.json укажите baseline — имя модели из models")


def c_boosting(ctx: Ctx) -> Outcome:
    m = load(ctx)
    models = m.get("models", {})
    if ctx.track == "llm":
        if any(r.get("family") == "rag" for r in models.values()):
            return ok()
        return fail("LLM-трек: в metrics.json нужна модель с family: rag")
    good = [n for n, r in models.items() if r.get("family") == "boosting"
            and str(r.get("framework", "")).lower() in BOOSTING]
    return ok() if good else fail("нет модели с family: boosting (LightGBM, XGBoost или CatBoost)")


def c_nn(ctx: Ctx) -> Outcome:
    if ctx.track == "llm":
        ev = ctx.json("reports/llm_eval.json")
        if not ev or ev.get("__invalid__"):
            return fail("LLM-трек: нет reports/llm_eval.json (результаты promptfoo или DeepEval)")
        n = len(ev.get("cases", []))
        return ok() if n >= 30 else part(n / 30, f"в llm_eval.json {n} кейсов, нужно от 30")
    m = load(ctx)
    nns = {n: str(r.get("framework", "")).lower() for n, r in m.get("models", {}).items()
           if r.get("family") in ("nn", "pretrained")}
    if not nns:
        return fail("нет нейросети: модель с family nn (PyTorch) или pretrained (fine-tuning)")
    if any(f in NN_FRAMEWORKS for f in nns.values()):
        return ok()
    return part(0.375, f"нейросеть обучена через {set(nns.values())}; для полного балла нужен "
                       "PyTorch (uv sync --group torch) или дообученная pretrained-модель")


def c_beats(ctx: Ctx) -> Outcome:
    m = load(ctx)
    metric, best, base = m["primary_metric"], m.get("best"), m.get("baseline")
    models = m.get("models", {})
    if best not in models or base not in models:
        return fail("в metrics.json не найдены best или baseline")
    if best == base:
        return fail("лучшей оказалась baseline — значит, остальные модели пока не работают")
    b, a = _metric(models[best], "test", metric), _metric(models[base], "test", metric)
    hib = m.get("higher_is_better", higher_is_better(metric))
    if b is None or a is None:
        return fail("нет test-метрик у best или baseline")
    if (b > a) if hib else (b < a):
        return ok()
    return fail(f"на test {best} {metric}={b:.4f} не лучше baseline {a:.4f}")


def c_threshold(ctx: Ctx) -> Outcome:
    m = load(ctx)
    metric = m["primary_metric"]
    thr = m.get("success_threshold", ctx.cfg_get("metric", "threshold"))
    best = m.get("models", {}).get(m.get("best"), {})
    v = _metric(best, "test", metric)
    if thr is None or v is None:
        return fail("нет порога успеха или метрики лучшей модели")
    hib = m.get("higher_is_better", higher_is_better(metric))
    if (v >= thr) if hib else (v <= thr):
        return ok()
    return fail(f"{metric}={v:.4f}, порог из этапа 1 — {thr}. Не дотянули: разберите причины в "
                "README (это оценят на защите)")


def c_optuna(ctx: Ctx) -> Outcome:
    if ctx.track == "llm":
        cols, rows = _csv_rows(ctx, "reports/mlflow_runs.csv")
        params = [c for c in cols if c.startswith("params.")]
        configs = {tuple(r.get(c, "") for c in params) for r in rows} if params else set()
        n = len(configs)
        return ok() if n >= 4 else part(n / 4, f"LLM-трек: конфигураций RAG в MLflow {n} из 4")
    cols, rows = _csv_rows(ctx, "reports/optuna_trials.csv")
    if not rows:
        return fail("нет reports/optuna_trials.csv — подбор гиперпараметров Optuna")
    return ok() if len(rows) >= 10 else part(len(rows) / 10, f"проб Optuna {len(rows)} из 10")


def c_mlflow(ctx: Ctx) -> Outcome:
    cols, rows = _csv_rows(ctx, "reports/mlflow_runs.csv")
    if not rows:
        return fail("нет reports/mlflow_runs.csv (выгрузка mlflow.search_runs после обучения)")
    hints = []
    if not any(c.startswith("params.") for c in cols):
        hints.append("в выгрузке нет параметров (params.*)")
    if not any(c.startswith("metrics.") for c in cols):
        hints.append("в выгрузке нет метрик (metrics.*)")
    if len(rows) < 5:
        hints.append(f"запусков {len(rows)} из 5")
    return ok() if not hints else part(0.5 if len(rows) else 0, "; ".join(hints))


def c_smoke(ctx: Ctx) -> Outcome:
    if ctx.track == "llm":
        res = ctx.tests()
        if isinstance(res, str):
            return fail(res)
        cases = res.get("llm", [])
        if not cases:
            return fail("LLM-трек: нужны тесты tests/test_llm*.py с подменой модели")
        bad = [c.name for c in cases if c.outcome == "failed"]
        return fail(f"падают {bad}") if bad else ok()
    good, msg, out_dir, seconds = ctx.smoke("train", timeout=300)
    if not good:
        return fail(f"uv run python -m mlproject.train --smoke падает: {msg}")
    produced = json.loads((out_dir / "reports" / "metrics.json").read_text("utf-8"))
    if not (out_dir / "mlflow.db").exists() or not produced.get("models"):
        return fail("smoke-прогон не записал metrics.json или запуск MLflow")
    if seconds > 180:
        return part(0.5, f"smoke-обучение идёт {seconds:.0f} с — урежьте блок smoke в config.yaml")
    return ok(f"{seconds:.0f} с")


def c_readme(ctx: Ctx) -> Outcome:
    body = readme_section(ctx, "Этап 3")
    good, why = md.filled(body, 40)
    if not good:
        return fail(f"README.md, раздел «Этап 3»: {why}")
    if not md.table_rows(body):
        return part(0.5, "в разделе «Этап 3» нужна таблица сравнения моделей")
    return ok()


def require_metrics(fn):
    def wrapper(ctx: Ctx) -> Outcome:
        try:
            return fn(ctx)
        except ValueError as exc:
            return fail(str(exc))
        except Skip:
            raise
    return wrapper


CHECKS = [
    Check("3.1", "reports/metrics.json по контракту", 6, c_structure),
    Check("3.2", "Baseline (sklearn Pipeline)", 4, require_metrics(c_baseline)),
    Check("3.3", "Градиентный бустинг (LLM-трек: RAG)", 6, require_metrics(c_boosting)),
    Check("3.4", "Нейросеть PyTorch / fine-tuning (LLM: evals)", 8, require_metrics(c_nn)),
    Check("3.5", "Лучшая модель обгоняет baseline на test", 10, require_metrics(c_beats)),
    Check("3.6", "Достигнут порог успеха из этапа 1", 4, require_metrics(c_threshold)),
    Check("3.7", "Optuna: ≥ 10 проб (LLM: ≥ 4 конфигурации)", 6, c_optuna),
    Check("3.8", "MLflow: ≥ 5 запусков с параметрами и метриками", 6, c_mlflow),
    Check("3.9", "Smoke-обучение воспроизводится (CI)", 6, c_smoke),
    Check("3.10", "README: раздел «Этап 3» с таблицей", 4, c_readme),
]
