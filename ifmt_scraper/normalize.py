from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime

from dateutil import parser as date_parser


PT_MONTHS = {
    "jan": 1,
    "janeiro": 1,
    "fev": 2,
    "fevereiro": 2,
    "mar": 3,
    "marco": 3,
    "março": 3,
    "abr": 4,
    "abril": 4,
    "mai": 5,
    "maio": 5,
    "jun": 6,
    "junho": 6,
    "jul": 7,
    "julho": 7,
    "ago": 8,
    "agosto": 8,
    "set": 9,
    "setembro": 9,
    "out": 10,
    "outubro": 10,
    "nov": 11,
    "novembro": 11,
    "dez": 12,
    "dezembro": 12,
}


def normalize_text(value: str) -> str:
    value = value or ""
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_title(value: str) -> str:
    value = strip_accents(normalize_text(value)).lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def clean_body_text(paragraphs: list[str] | str) -> str:
    if isinstance(paragraphs, str):
        raw_parts = re.split(r"\n{2,}", paragraphs)
    else:
        raw_parts = paragraphs

    cleaned: list[str] = []
    previous = ""
    for part in raw_parts:
        text = normalize_text(part)
        if not text or text == previous:
            continue
        if _is_repetitive_portal_text(text):
            continue
        cleaned.append(text)
        previous = text
    return "\n\n".join(cleaned)


def extract_lead(body_text: str) -> str:
    for paragraph in re.split(r"\n{2,}", body_text or ""):
        text = normalize_text(paragraph)
        if len(text.split()) >= 5:
            return text
    return ""


def calculate_word_count(body_text: str) -> int:
    return len(re.findall(r"\b[\wÀ-ÿ]+\b", body_text or "", flags=re.UNICODE))


def sha256_text(value: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def parse_publication_datetime(raw_value: str) -> tuple[str, str, str]:
    raw = normalize_text(raw_value)
    if not raw:
        return "", "", ""

    iso = _parse_iso_datetime(raw)
    if iso:
        date_value = iso.date().isoformat()
        time_value = f"{iso.hour:02d}:{iso.minute:02d}"
        return date_value, time_value, raw

    time_value = extract_time(raw)

    numeric = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", raw)
    if numeric:
        day, month, year = (int(part) for part in numeric.groups())
        if year < 100:
            year += 2000
        return _safe_date(year, month, day), time_value, raw

    pt_match = re.search(
        r"\b(\d{1,2})\s*(?:de\s*)?([A-Za-zÀ-ÿçÇ]{3,})"
        r"(?:\s*(?:de\s*)?(\d{4}))?\b",
        raw,
        flags=re.IGNORECASE,
    )
    if pt_match:
        day = int(pt_match.group(1))
        month_label = pt_match.group(2).lower()
        year_label = pt_match.group(3)
        month = PT_MONTHS.get(month_label)
        if month and year_label:
            return _safe_date(int(year_label), month, day), time_value, raw
        return "", time_value, raw

    english_like = re.search(
        r"\b([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\b",
        raw,
        flags=re.IGNORECASE,
    )
    if english_like:
        try:
            parsed = date_parser.parse(english_like.group(0), fuzzy=True)
            return parsed.date().isoformat(), time_value, raw
        except (ValueError, OverflowError):
            pass

    return "", time_value, raw


def extract_time(raw_value: str) -> str:
    raw = normalize_text(raw_value).lower()
    match = re.search(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?\b", raw)
    if not match:
        return ""
    hour = int(match.group(1))
    minute = int(match.group(2))
    suffix = match.group(3)
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _parse_iso_datetime(raw: str) -> datetime | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?)?", raw)
    if not match:
        return None
    value = match.group(0).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return date_parser.parse(value)
        except (ValueError, OverflowError):
            return None


def _safe_date(year: int, month: int, day: int) -> str:
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return ""


def _is_repetitive_portal_text(text: str) -> bool:
    normalized = strip_accents(text).lower()
    blocked_exact = {
        "a+",
        "a-",
        "a",
        "menu",
        "compartilhe",
        "ultimas noticias",
        "últimas notícias",
    }
    if normalized in blocked_exact:
        return True
    blocked_fragments = (
        "orgaos do governo",
        "acesso a informacao",
        "redes sociais",
        "todos os direitos reservados",
    )
    return any(fragment in normalized for fragment in blocked_fragments)
