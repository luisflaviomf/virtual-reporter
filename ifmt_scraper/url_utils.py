from __future__ import annotations

import posixpath
import re
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse

from .config import LEGACY_IP_HOST


BLOCKED_SCHEMES = {"", "http", "https"}
STATIC_EXTENSIONS = {
    ".7z",
    ".bmp",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".svg",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


def is_legacy_ip_url(url: str) -> bool:
    return urlparse(url).hostname == LEGACY_IP_HOST


def resolve_link(href: str, base_url: str) -> str | None:
    href = (href or "").strip()
    if not href:
        return None
    if href.startswith("#"):
        return None
    lowered = href.lower()
    if lowered.startswith(("mailto:", "tel:", "javascript:")):
        return None

    resolved = urljoin(base_url, href)
    resolved, _fragment = urldefrag(resolved)
    parsed = urlparse(resolved)
    if parsed.scheme not in BLOCKED_SCHEMES or not parsed.netloc:
        return None
    return resolved


def has_static_extension(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in STATIC_EXTENSIONS)


def is_relative_href(href: str) -> bool:
    href = (href or "").strip()
    return href.startswith("/") or (
        bool(href) and not urlparse(href).scheme and not href.startswith("#")
    )


def normalize_url(url: str) -> str:
    parsed = urlparse(urldefrag(url.strip())[0])
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    path = posixpath.normpath(path)
    if not path.startswith("/"):
        path = f"/{path}"
    if parsed.path.endswith("/") and not path.endswith("/"):
        path = f"{path}/"
    if path == "/.":
        path = "/"

    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_query_param(key)
    ]
    query = urlencode(query_items, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return ""
    return path.split("/")[-1]


def _is_tracking_query_param(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered.startswith("utm_")
        or lowered
        in {
            "fbclid",
            "gclid",
            "mc_cid",
            "mc_eid",
            "ref",
            "source",
        }
    )
