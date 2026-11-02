"""Разбор Markdown-отчётов: разделы, заполненность, пары «ключ: значение»."""

from __future__ import annotations

import re

COMMENT = re.compile(r"<!--.*?-->", re.S)
WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{2,}")
# Заглушки шаблона: TODO, дата-плейсхолдер и подсказки в угловых скобках «<впишите …>»
PLACEHOLDER = re.compile(r"\bTODO\b|ДД\.ММ\.ГГГГ|<[А-Яа-яЁё][^<>\n]{0,80}>")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def strip_comments(text: str) -> str:
    return COMMENT.sub("", text)


def norm(heading: str) -> str:
    h = heading.strip().strip("*").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", h)


def sections(text: str | None, level: int = 2) -> dict[str, str]:
    """{'заголовок в нижнем регистре': 'текст раздела вместе с подразделами'}."""
    if not text:
        return {}
    out: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m and len(m.group(1)) <= level:
            if current is not None:
                out[current] = "\n".join(buf)
            current = norm(m.group(2)) if len(m.group(1)) == level else None
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf)
    return out


def find(secs: dict[str, str], name: str) -> str | None:
    """Раздел, заголовок которого начинается с name (или хотя бы содержит его)."""
    key = norm(name)
    for h, body in secs.items():
        if h.startswith(key):
            return body
    for h, body in secs.items():
        if key in h:
            return body
    return None


def words(text: str) -> int:
    return len(WORD.findall(strip_comments(text)))


def filled(body: str | None, min_words: int = 20) -> tuple[bool, str]:
    """Раздел есть, в нём нет TODO и заглушек, и своих слов достаточно."""
    if body is None:
        return False, "нет раздела"
    clean = strip_comments(body)
    if PLACEHOLDER.search(clean):
        return False, "остались TODO или заглушки"
    n = words(clean)
    if n < min_words:
        return False, f"слишком коротко ({n} слов, нужно от {min_words})"
    return True, ""


def kv(body: str | None) -> dict[str, str]:
    """Строки вида «- Ключ: значение» (в том числе с **жирным** ключом)."""
    out: dict[str, str] = {}
    for line in strip_comments(body or "").splitlines():
        m = re.match(r"^\s*[-*]\s+\**([^:*|]+?)\**\s*:\s*\**\s*(.*)$", line)
        if m:
            key = norm(m.group(1))
            value = m.group(2).strip().strip("`*").strip()
            out[key] = value
    return out


def value_ok(v: str | None) -> bool:
    return bool(v and v.strip()) and not PLACEHOLDER.search(v or "")


def table_rows(body: str | None) -> list[list[str]]:
    """Строки Markdown-таблицы без заголовка и разделителя."""
    rows = []
    for line in strip_comments(body or "").splitlines():
        s = line.strip()
        if s.startswith("|") and not re.match(r"^\|[\s:|-]+\|?$", s):
            rows.append([c.strip() for c in s.strip("|").split("|")])
    return rows[1:] if rows else []
