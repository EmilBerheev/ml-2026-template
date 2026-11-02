"""Каркас автопроверки: проверки, контекст, запуск подпроцессов, отчёт."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # грейдер не должен падать, даже если окружение сломано
    yaml = None


class Skip(Exception):
    """Проверку нельзя выполнить в этом окружении.

    award=True — баллы не снимаются (например, нужно то, чего нет локально);
    award=False — запуск с --no-live: проверка не выполнялась, баллы не начислены.
    """

    def __init__(self, message: str, award: bool = True):
        super().__init__(message)
        self.award = award


@dataclass
class Outcome:
    fraction: float
    hint: str = ""


def ok(hint: str = "") -> Outcome:
    return Outcome(1.0, hint)


def fail(hint: str) -> Outcome:
    return Outcome(0.0, hint)


def part(fraction: float, hint: str) -> Outcome:
    return Outcome(max(0.0, min(1.0, fraction)), hint)


@dataclass
class Check:
    id: str
    title: str
    points: int
    fn: Callable[[Ctx], Outcome]


@dataclass
class CheckResult:
    id: str
    title: str
    points: int
    earned: float
    status: str
    hint: str


@dataclass
class StageResult:
    key: str
    title: str
    questions: str
    checks: list[CheckResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def earned(self) -> float:
        return round(sum(c.earned for c in self.checks), 1)

    @property
    def max(self) -> int:
        return sum(c.points for c in self.checks)


@dataclass
class TestCase:
    group: str
    name: str
    base: str
    outcome: str
    message: str = ""


class Ctx:
    """Всё, что проверкам нужно знать о репозитории. Тяжёлые прогоны кэшируются."""

    def __init__(self, root: Path, live: bool = True):
        self.root = root
        self.live = live
        self.ci = os.environ.get("GITHUB_ACTIONS") == "true"
        self.tmp = Path(tempfile.mkdtemp(prefix="grader-"))
        self._cache: dict[str, Any] = {}
        self.cfg, self.cfg_error = self._load_config()

    # ---------- файлы
    def path(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def text(self, rel: str) -> str | None:
        p = self.path(rel)
        if not p.is_file():
            return None
        return p.read_text("utf-8", errors="replace")

    def json(self, rel: str) -> Any:
        txt = self.text(rel)
        if txt is None:
            return None
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return {"__invalid__": True}

    def yaml(self, rel: str) -> Any:
        txt = self.text(rel)
        if txt is None or yaml is None:
            return None
        try:
            return yaml.safe_load(txt)
        except yaml.YAMLError:
            return {"__invalid__": True}

    def _load_config(self) -> tuple[dict | None, str | None]:
        if yaml is None:
            return None, "не установлен PyYAML — выполните uv sync"
        data = self.yaml("configs/config.yaml")
        if data is None:
            return None, "нет configs/config.yaml"
        if not isinstance(data, dict) or data.get("__invalid__"):
            return None, "configs/config.yaml не разбирается как YAML"
        return data, None

    def cfg_get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.cfg or {}
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def track(self) -> str:
        return str(self.cfg_get("project", "track", default="classic")).lower()

    @property
    def task(self) -> str:
        return str(self.cfg_get("task", "type", default="binary")).lower()

    # ---------- подпроцессы
    def env(self, **extra: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                    "MLFLOW_DISABLE_AGENT_HINT": "1", "GIT_PYTHON_REFRESH": "quiet",
                    "MLPROJECT_ROOT": str(self.root)})
        env.update(extra)
        return env

    def run(self, args: list[str], timeout: int = 300, **env: str) -> tuple[int, str, float]:
        start = time.perf_counter()
        try:
            proc = subprocess.run(args, cwd=self.root, env=self.env(**env), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
            out = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode, out, time.perf_counter() - start
        except subprocess.TimeoutExpired:
            return -1, f"превышено время ({timeout} с)", time.perf_counter() - start
        except OSError as exc:
            return -2, str(exc), time.perf_counter() - start

    def python(self, *args: str, timeout: int = 300, **env: str) -> tuple[int, str, float]:
        return self.run([sys.executable, *args], timeout=timeout, **env)

    def require_live(self) -> None:
        if not self.live:
            raise Skip("не запускалось (--no-live)", award=False)

    # ---------- pytest: один прогон на все этапы
    def tests(self) -> dict[str, list[TestCase]] | str:
        if "tests" in self._cache:
            return self._cache["tests"]
        if not self.live:
            raise Skip("не запускалось (--no-live)", award=False)
        junit = self.tmp / "junit.xml"
        code, out, _ = self.python("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                                   f"--junitxml={junit}", "-o", "junit_family=xunit2",
                                   timeout=600)
        if not junit.exists():
            result: dict[str, list[TestCase]] | str = (
                "pytest не запустился: " + _tail(out))
        else:
            result = _parse_junit(junit)
        self._cache["tests"] = result
        return result

    # ---------- smoke-прогоны пайплайна в отдельной папке (ваши reports/ не трогаем)
    def smoke(self, module: str, timeout: int = 300) -> tuple[bool, str, Path, float]:
        key = f"smoke:{module}"
        if key in self._cache:
            return self._cache[key]
        self.require_live()
        out_dir = self.tmp / "smoke"
        code, out, seconds = self.python("-m", f"mlproject.{module}", "--smoke", "--output-dir",
                                         str(out_dir), timeout=timeout)
        res = (code == 0, _tail(out), out_dir, seconds)
        self._cache[key] = res
        return res

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def _tail(text: str, lines: int = 6) -> str:
    rows = [r for r in text.strip().splitlines() if r.strip()]
    return " | ".join(rows[-lines:])[-600:]


def _group(classname: str) -> str:
    for piece in classname.split("."):
        if piece.startswith("test_"):
            name = piece.removeprefix("test_")
            for g in ("data", "model", "api", "llm", "monitor"):
                if name.startswith(g):
                    return g
            return name
    return "other"


def _parse_junit(path: Path) -> dict[str, list[TestCase]]:
    groups: dict[str, list[TestCase]] = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        name = tc.get("name", "")
        outcome, message = "passed", ""
        for child in tc:
            if child.tag in ("failure", "error"):
                outcome, message = "failed", (child.get("message") or "")[:200]
            elif child.tag == "skipped":
                outcome, message = "skipped", (child.get("message") or "")[:200]
        g = _group(tc.get("classname", ""))
        groups.setdefault(g, []).append(
            TestCase(g, name, name.split("[")[0], outcome, message))
    return groups


def run_stage(ctx: Ctx, key: str, title: str, questions: str, checks: list[Check],
              notes: Callable[[Ctx], list[str]] | None = None) -> StageResult:
    res = StageResult(key, title, questions)
    for chk in checks:
        try:
            out = chk.fn(ctx)
            earned = round(chk.points * out.fraction, 1)
            status = "ok" if out.fraction >= 1 else ("частично" if out.fraction > 0 else "нет")
            hint = out.hint if out.fraction < 1 else ""
        except Skip as s:
            earned = float(chk.points) if s.award else 0.0
            status, hint = "пропущено", str(s)
        except Exception as exc:  # ошибка внутри одной проверки не роняет весь грейдер
            earned, status = 0.0, "ошибка проверки"
            hint = f"{type(exc).__name__}: {exc}"[:300]
        res.checks.append(CheckResult(chk.id, chk.title, chk.points, earned, status, hint))
    if notes:
        try:
            res.notes = notes(ctx)
        except Exception as exc:
            res.notes = [f"не удалось собрать примечания: {exc}"]
    return res


STATUS_MARK = {"ok": "✅", "частично": "🟡", "нет": "❌", "пропущено": "⏭️",
               "ошибка проверки": "⚠️"}


def to_markdown(stages: list[StageResult], manual_points: int = 40) -> str:
    lines = ["# Автопроверка", ""]
    lines += ["| Этап | Авто | Макс |", "|---|---:|---:|"]
    for s in stages:
        lines.append(f"| {s.title} | {s.earned:g} | {s.max} |")
    lines += ["", f"Ручную часть ({manual_points} баллов на этап) выставляет преподаватель "
                  "при проверке репозитория.", ""]
    for s in stages:
        lines += [f"## {s.title} — {s.earned:g} из {s.max}", f"Вопросы: {s.questions}", "",
                  "| Проверка | Баллы | Статус | Что исправить |", "|---|---:|---|---|"]
        for c in s.checks:
            hint = c.hint.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {c.id} {c.title} | {c.earned:g}/{c.points} | "
                         f"{STATUS_MARK.get(c.status, '')} {c.status} | {hint} |")
        for n in s.notes:
            lines.append(f"\n> {n}")
        lines.append("")
    return "\n".join(lines)


def to_console(stages: list[StageResult]) -> str:
    out = []
    for s in stages:
        out.append(f"\n=== {s.title}: {s.earned:g} / {s.max}  (вопросы {s.questions})")
        width = max((len(f"{c.id} {c.title}") for c in s.checks), default=10)
        for c in s.checks:
            label = f"{c.id} {c.title}".ljust(width)
            score = f"{c.earned:g}/{c.points}".rjust(6)
            line = f"  {label}  {score}  {c.status}"
            if c.hint:
                line += f"\n      → {c.hint}"
            out.append(line)
        for n in s.notes:
            out.append(f"  ! {n}")
    total = sum(s.earned for s in stages if s.key.startswith("stage"))
    maximum = sum(s.max for s in stages if s.key.startswith("stage"))
    if maximum:
        out.append(f"\nИтого авто по этапам: {total:g} / {maximum}")
    return "\n".join(out)


def to_json(stages: list[StageResult], ctx: Ctx) -> dict[str, Any]:
    from grader import __version__

    return {
        "grader_version": __version__,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": os.environ.get("GITHUB_SHA"),
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "track": ctx.track,
        "stages": {
            s.key: {
                "title": s.title, "questions": s.questions, "auto": s.earned, "max": s.max,
                "checks": [c.__dict__ for c in s.checks], "notes": s.notes,
            } for s in stages
        },
        "total_auto": round(sum(s.earned for s in stages if s.key.startswith("stage")), 1),
        "max_auto": sum(s.max for s in stages if s.key.startswith("stage")),
    }
