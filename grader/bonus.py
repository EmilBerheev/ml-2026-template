"""Бонус для «автомата» (bonus.md): голос, LoRA для диффузионной модели или QLoRA для LLM."""

from __future__ import annotations

import re

from grader import md
from grader.common import doc_sections
from grader.core import Check, Ctx, Outcome, fail, ok, part

KEY = "bonus"
TITLE = "Бонус (по желанию)"
QUESTIONS = "26, 29, 38"
README = "bonus/README.md"
CONSENT = "bonus/CONSENT.md"
RESULTS = "bonus/eval/results.json"
SECTIONS = ["Цель", "Данные", "Обучение", "Оценка", "Маркировка ИИ-контента",
            "Ограничения и риски", "Как воспроизвести"]
VARIANTS = {"voice", "lora", "qlora"}
WEIGHTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
MEDIA = (".wav", ".mp3", ".flac", ".ogg", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".md",
         ".json", ".jsonl")


def variant(ctx: Ctx) -> str | None:
    head = md.kv(ctx.text(README) or "")
    v = (head.get("вариант") or "").lower().split()
    return v[0] if v and v[0] in VARIANTS else None


def consent_ok(ctx: Ctx) -> tuple[bool, str]:
    text = ctx.text(CONSENT)
    if text is None:
        return False, f"нет {CONSENT}"
    clean = md.strip_comments(text)
    boxes = re.findall(r"^\s*[-*]\s*\[( |x|X)\]", clean, re.M)
    if not boxes:
        return False, f"в {CONSENT} нет пунктов-флажков"
    if any(b == " " for b in boxes):
        return False, f"{CONSENT}: подтвердите все пункты ([x]) или откажитесь от варианта"
    kv = md.kv(clean)
    missing = [k for k in ("фио", "дата") if not md.value_ok(kv.get(k))]
    if missing:
        return False, f"{CONSENT}: заполните {missing}"
    return True, ""


def c_consent(ctx: Ctx) -> Outcome:
    good, why = consent_ok(ctx)
    return ok() if good else fail(why)


def c_readme(ctx: Ctx) -> Outcome:
    if variant(ctx) is None:
        return fail(f"{README}: в шапке «- Вариант: voice | lora | qlora»")
    return doc_sections(ctx, README, SECTIONS, min_words=20)


def c_config(ctx: Ctx) -> Outcome:
    folder = ctx.path("bonus/configs")
    files = [p for p in folder.glob("*") if p.suffix in (".yaml", ".yml", ".json", ".toml")] \
        if folder.exists() else []
    return ok() if files else fail("положите конфиг обучения в bonus/configs/ (yaml/json/toml)")


def c_eval(ctx: Ctx) -> Outcome:
    res = ctx.json(RESULTS)
    if not isinstance(res, dict) or res.get("__invalid__"):
        return fail(f"нет {RESULTS} (формат — в bonus.md)")
    v = variant(ctx)
    if v == "voice":
        need, n_key, n_min = ("cer", "speaker_similarity"), "n_utterances", 20
    elif v == "lora":
        need, n_key, n_min = ("clip_score", "identity_similarity"), "n_prompts", 30
    elif v == "qlora":
        need, n_key, n_min = ("base", "tuned", "metric"), "n_eval", 50
    else:
        return fail("сначала укажите вариант в bonus/README.md")
    hints = []
    missing = [k for k in need if k not in res]
    if v == "voice" and "cer" in missing and "wer" in res:
        missing.remove("cer")  # CER или WER — что удобнее для вашего ASR
    if v == "lora" and len(missing) == 1:
        missing = []  # достаточно одной из метрик: CLIP score или сходство идентичности
    if missing:
        hints.append(f"нет полей {missing}")
    n = res.get(n_key, 0)
    if not isinstance(n, int) or n < n_min:
        hints.append(f"{n_key}={n}, нужно от {n_min}")
    if v == "qlora" and not missing:
        metric = res["metric"]
        base, tuned = res["base"].get(metric), res["tuned"].get(metric)
        if not isinstance(base, (int, float)) or not isinstance(tuned, (int, float)):
            hints.append(f"в base и tuned нужна метрика {metric}")
        elif tuned <= base and not res.get("lower_is_better"):
            hints.append(f"дообученная модель не лучше базовой по {metric}")
    if "baseline" not in res and v != "qlora":
        hints.append("нужно сравнение с baseline (zero-shot или базовая модель): поле baseline")
    return ok() if not hints else part(max(0.0, 1 - 0.34 * len(hints)), "; ".join(hints))


def c_labels(ctx: Ctx) -> Outcome:
    folder = ctx.path("bonus/samples")
    manifest = ctx.text("bonus/samples/MANIFEST.md")
    if manifest is None:
        return fail("нет bonus/samples/MANIFEST.md с перечнем примеров и пометкой «ИИ»")
    manifest = md.strip_comments(manifest)
    files = [p for p in folder.iterdir() if p.is_file() and p.name != "MANIFEST.md"
             and p.suffix.lower() in MEDIA]
    total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    if not files:
        return fail("в bonus/samples/ нет примеров генерации (и перечислите их в MANIFEST.md)")
    hints = []
    unlisted = [p.name for p in files if p.name not in manifest]
    if unlisted:
        hints.append(f"не перечислены в MANIFEST.md: {unlisted[:3]}")
    if not re.search(r"сгенерировано ИИ|создано ИИ|ИИ-контент|AI-generated", manifest, re.I):
        hints.append("пометка «Сгенерировано ИИ» в MANIFEST.md")
    if total > 50 * 2**20:
        hints.append("примеры больше 50 МБ — оставьте лучшие")
    return ok() if not hints else part(max(0.0, 1 - 0.34 * len(hints)), "; ".join(hints))


def c_weights(ctx: Ctx) -> Outcome:
    big = []
    for p in ctx.path("bonus").rglob("*"):
        if p.is_file() and (p.suffix.lower() in WEIGHTS and p.stat().st_size > 10 * 2**20
                            or p.stat().st_size > 50 * 2**20):
            big.append(p.name)
    if big:
        return fail(f"веса в репозитории: {big[:3]} — выложите на Hugging Face Hub (private) и "
                    "дайте ссылку")
    text = md.strip_comments(ctx.text(README) or "")
    if re.search(r"huggingface\.co|hf\.co|drive\.google|releases|веса не публику", text, re.I):
        return ok()
    return fail(f"{README}: где лежат веса (ссылка на HF Hub/Drive) или «веса не публикуются»")


def gate(ctx: Ctx) -> str | None:
    """Этика — обязательное условие: без согласия бонус не засчитывается целиком."""
    if variant(ctx) is None and not ctx.exists(RESULTS):
        return None
    good, why = consent_ok(ctx)
    return None if good else f"Этическое условие не выполнено ({why}) — бонус не засчитан."


CHECKS = [
    Check("B.1", "Согласие и права (обязательно)", 10, c_consent),
    Check("B.2", "bonus/README.md: разделы заполнены", 14, c_readme),
    Check("B.3", "Конфиг обучения", 6, c_config),
    Check("B.4", "Оценка в bonus/eval/results.json", 14, c_eval),
    Check("B.5", "Примеры с пометкой «Сгенерировано ИИ»", 8, c_labels),
    Check("B.6", "Веса не в git, есть ссылка", 8, c_weights),
]
