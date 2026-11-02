"""Автопроверка: uv run python -m grader stage1 | ... | stage6 | bonus | all

Ключи:
    --json PATH   куда записать результат (по умолчанию grade.json)
    --no-live     не запускать тесты и smoke-прогоны (быстро, но живые проверки «пропущены»)
Код возврата всегда 0, если не упал сам грейдер: оценка — это не «красный CI».
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from grader import bonus, stage1, stage2, stage3, stage4, stage5, stage6
from grader.core import Ctx, run_stage, to_console, to_json, to_markdown

STAGES = {m.KEY: m for m in (stage1, stage2, stage3, stage4, stage5, stage6)}
ALIASES = {**{f"lab{i}": f"stage{i}" for i in range(1, 7)}, **{str(i): f"stage{i}"
                                                                for i in range(1, 7)}}


def tag_note(ctx: Ctx, key: str) -> str | None:
    """Сдача этапа — git-тег stage-N. Баллы за тег не ставятся, это напоминание."""
    if not (ctx.root / ".git").exists():
        return None
    code, out, _ = ctx.run(["git", "tag", "--list", key.replace("stage", "stage-")], timeout=30)
    if code != 0:
        return None
    tag = key.replace("stage", "stage-")
    if out.strip():
        return f"Тег {tag} найден — этап отмечен как сданный."
    return (f"Тега {tag} нет. Когда этап готов: git tag {tag} && git push origin {tag}"
            + (" (в CI теги видны при fetch-depth: 0)" if ctx.ci else ""))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="python -m grader", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default="all",
                    help="stage1…stage6, bonus или all")
    ap.add_argument("--json", default="grade.json")
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    target = ALIASES.get(args.target, args.target)
    if target == "all":
        keys = [*STAGES, "bonus"]
    elif target in STAGES or target == "bonus":
        keys = [target]
    else:
        ap.error(f"неизвестная цель {args.target}")

    root = Path(args.root).resolve()
    ctx = Ctx(root, live=not args.no_live)
    started = time.perf_counter()
    results = []
    try:
        for key in keys:
            mod = bonus if key == "bonus" else STAGES[key]
            res = run_stage(ctx, mod.KEY, mod.TITLE, mod.QUESTIONS, mod.CHECKS,
                            getattr(mod, "notes", None))
            if key == "bonus":
                msg = bonus.gate(ctx)
                if msg:
                    for c in res.checks:
                        c.earned = 0.0
                    res.notes.append(msg)
            elif note := tag_note(ctx, key):
                res.notes.append(note)
            if ctx.cfg_error:
                res.notes.insert(0, f"Конфигурация: {ctx.cfg_error}")
            results.append(res)
    finally:
        ctx.cleanup()

    print(to_console(results))
    print(f"\nВремя проверки: {time.perf_counter() - started:.0f} с. "
          "Ручная часть (40 баллов на этап) — на защите.")
    report = to_json(results, ctx)
    report["seconds"] = round(time.perf_counter() - started, 1)
    Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(to_markdown(results) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
