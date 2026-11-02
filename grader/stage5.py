"""Этап 5. Деплой (вопрос 34)."""

from __future__ import annotations

import json
import re

from grader import md
from grader.common import readme_section, tests_check
from grader.core import Check, Ctx, Outcome, fail, ok, part

KEY = "stage5"
TITLE = "Этап 5. Деплой"
QUESTIONS = "34"
MODEL = "models/model.joblib"
META = "models/model_meta.json"


def _meta(ctx: Ctx) -> dict:
    m = ctx.json(META)
    return m if isinstance(m, dict) and not m.get("__invalid__") else {}


def _has_model(ctx: Ctx) -> bool:
    return ctx.exists(MODEL) or bool(_meta(ctx).get("artifact_url"))


def probe(ctx: Ctx) -> dict:
    """Один прогон grader/probes/api_probe.py на все живые проверки API."""
    if "api_probe" in ctx._cache:
        return ctx._cache["api_probe"]
    ctx.require_live()
    if not ctx.exists(MODEL):
        res = {"error": f"нет {MODEL} — сервису нечего загружать (обучите и закоммитьте модель "
                        "или скачивайте её по artifact_url в model_meta.json)"}
    else:
        sample = ctx.path(ctx.cfg_get("data", "sample_path", default="data/sample.csv"))
        code, out, _ = ctx.python(str(ctx.root / "grader" / "probes" / "api_probe.py"),
                                  str(sample), timeout=180, MODEL_PATH=str(ctx.path(MODEL)),
                                  LLM_BACKEND="fake")
        lines = [ln for ln in out.splitlines() if ln.startswith("{")]
        try:
            res = json.loads(lines[-1]) if lines else {"error": out[-500:]}
        except json.JSONDecodeError:
            res = {"error": out[-500:]}
        if code != 0 and "error" not in res:
            res["error"] = out[-500:]
    ctx._cache["api_probe"] = res
    return res


def c_tests(ctx: Ctx) -> Outcome:
    return tests_check(ctx, "api", 2)


def c_model(ctx: Ctx) -> Outcome:
    if not _has_model(ctx):
        return fail(f"нет {MODEL} (или artifact_url в {META}) — сервису нечего загружать")
    meta = _meta(ctx)
    hints = []
    if not meta.get("model_version"):
        hints.append(f"в {META} нет model_version")
    if meta.get("smoke"):
        hints.append("закоммичена smoke-модель — обучите полную")
    if ctx.exists(MODEL) and ctx.path(MODEL).stat().st_size > 95 * 2**20:
        hints.append("модель больше 95 МБ: GitHub её не примет — используйте artifact_url")
    return ok() if not hints else part(0.5, "; ".join(hints))


def c_health(ctx: Ctx) -> Outcome:
    res = probe(ctx)
    h = res.get("health")
    if not h:
        return fail(res.get("error", "сервис не поднялся"))
    if h["status"] == 200 and isinstance(h["body"], dict) and h["body"].get("status") == "ok":
        return ok()
    return fail(f"GET /health → {h['status']} {h['body']}; нужен 200 и status: ok")


def c_predict(ctx: Ctx) -> Outcome:
    res = probe(ctx)
    p = res.get("predict")
    if not p:
        return fail(res.get("error", "не удалось вызвать /predict"))
    if p["status"] != 200:
        return fail(f"POST /predict → {p['status']}: {str(p['body'])[:200]}")
    body = p["body"] if isinstance(p["body"], dict) else {}
    hints = []
    if "prediction" not in body and "answer" not in body:
        hints.append("в ответе нет поля prediction")
    if not body.get("model_version"):
        hints.append("в ответе нет model_version")
    if ctx.task in ("binary",) and ctx.track != "llm" and body.get("probability") is None:
        hints.append("для классификации верните probability")
    if res.get("constraints", 0) == 0:
        hints.append("в схеме запроса нет ограничений (Field(ge=…, le=…), Literal) — "
                     "валидируйте вход")
    return ok() if not hints else part(1 - 0.25 * len(hints), "; ".join(hints))


def c_invalid(ctx: Ctx) -> Outcome:
    res = probe(ctx)
    inv = res.get("invalid")
    if not inv:
        return fail(res.get("error", "проверка не выполнялась"))
    if inv["status"] == 422:
        return ok()
    return fail(f"на неверный вход /predict вернул {inv['status']}, ожидали 422")


def c_docker(ctx: Ctx) -> Outcome:
    text = ctx.text("Dockerfile")
    if text is None:
        return fail("нет Dockerfile")
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    body = "\n".join(lines)
    score, hints = 0.0, []
    base_ok = re.search(r"uv sync[^\n]*--(locked|frozen)|uv export|requirements[^\n]*\.txt", body)
    model_ok = re.search(r"^COPY[^\n]*models", body, re.M) or "artifact_url" in body.lower()
    cmd_ok = re.search(r"^(CMD|ENTRYPOINT)[^\n]*(uvicorn|fastapi|gunicorn|granian)", body, re.M)
    if base_ok and model_ok and cmd_ok:
        score += 4
    else:
        hints.append("зависимости по lock-файлу, COPY models и CMD с uvicorn")
    users = re.findall(r"^USER\s+(\S+)", body, re.M)
    if users and users[-1] not in ("root", "0", "0:0"):
        score += 3
    else:
        hints.append("запустите сервис не от root: USER app")
    if re.search(r"^HEALTHCHECK\s+", body, re.M):
        score += 3
    else:
        hints.append("добавьте HEALTHCHECK на /health")
    ignore = ctx.text(".dockerignore") or ""
    if ".venv" in ignore and re.search(r"data/raw|mlflow|mlruns", ignore):
        score += 2
    else:
        hints.append(".dockerignore: исключите .venv, data/raw, mlruns и mlflow.db")
    return part(score / 12, "; ".join(hints))


def c_readme(ctx: Ctx) -> Outcome:
    body = readme_section(ctx, "Этап 5")
    good, why = md.filled(body, 40)
    if not good:
        return fail(f"README.md, раздел «Этап 5»: {why}")
    text = md.strip_comments(body or "")
    score, hints = 0.25, []
    if "docker build" in text and "docker run" in text:
        score += 0.25
    else:
        hints.append("команды docker build и docker run")
    if re.search(r"curl |httpie|Invoke-RestMethod|http POST|requests\.post", text):
        score += 0.25
    else:
        hints.append("пример запроса (curl)")
    if re.search(r"\d+(?:[.,]\d+)?\s*(мс|ms)\b|p95|p99", text, re.I):
        score += 0.25
    else:
        hints.append("замер задержки (p50/p95, мс)")
    return part(score, "README «Этап 5»: добавьте " + ", ".join(hints) if hints else "")


CHECKS = [
    Check("5.1", "API-тесты: свои (≥ 2) и все проходят", 16, c_tests),
    Check("5.2", "Модель для сервиса и model_meta.json", 6, c_model),
    Check("5.3", "GET /health → 200, status ok", 6, c_health),
    Check("5.4", "POST /predict по строке из sample.csv", 8, c_predict),
    Check("5.5", "Неверный вход → 422", 4, c_invalid),
    Check("5.6", "Dockerfile: lock, модель, non-root, HEALTHCHECK", 12, c_docker),
    Check("5.7", "README: запуск, docker, curl, задержка", 8, c_readme),
]
