from __future__ import annotations

import re
import unicodedata
from typing import Any

from .io_utils import normalize_space


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def text_key(value: Any) -> str:
    text = strip_accents(normalize_space(value)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(strip_accents(normalize_space(value)).lower())]


def remove_query_details(value: str) -> str:
    text = normalize_space(value)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "", text)
    text = re.sub(r"\b\d{4}\b", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:600]
