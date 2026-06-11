from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, replace
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from .config import LEGACY_IP_HOST
from .url_utils import has_static_extension, is_relative_href, resolve_link


ARTICLE_CONTEXT_HINTS = {
    "article",
    "blog",
    "card",
    "content",
    "entry",
    "item",
    "lista",
    "noticia",
    "news",
    "post",
}
NAV_CONTEXT_HINTS = {
    "breadcrumb",
    "footer",
    "menu",
    "nav",
    "pagin",
    "pagination",
    "sidebar",
    "social",
    "topbar",
}
KNOWN_LEGACY_PUBLISHERS = (
    "Campus Cuiaba - Bela Vista",
    "Campus Cuiabá - Bela Vista",
    "Campus Barra do Garcas",
    "Campus Barra do Garças",
    "Campus Primavera do Leste",
    "Campus Lucas do Rio Verde",
    "Campus Campo Novo do Parecis",
    "Campus Pontes e Lacerda",
    "Campus Varzea Grande",
    "Campus Várzea Grande",
    "Campus Alta Floresta",
    "Campus Rondonopolis",
    "Campus Rondonópolis",
    "Campus Tangara da Serra",
    "Campus Tangará da Serra",
    "Campus Sao Vicente",
    "Campus São Vicente",
    "Campus Caceres",
    "Campus Cáceres",
    "Campus Campo Verde",
    "Campus Diamantino",
    "Campus Guaranta do Norte",
    "Campus Guarantã do Norte",
    "Campus Avancado de Lucas do Rio Verde",
    "Campus Avançado de Lucas do Rio Verde",
    "Campus Confresa",
    "Campus Cuiaba",
    "Campus Cuiabá",
    "Campus Juina",
    "Campus Juína",
    "Campus Sorriso",
    "Campus Sinop",
    "Reitoria",
)


@dataclass(frozen=True)
class CandidateLink:
    url: str
    raw_href: str
    text: str
    pattern: str
    relative_href: bool
    title: str = ""
    summary: str = ""
    date_label_raw: str = ""
    publisher_unit: str = ""
    image_url: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def page_fingerprint(html: str) -> str:
    normalized = re.sub(r"\s+", " ", html or "").strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def extract_page_title(html: str) -> str | None:
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.get_text(strip=True):
        return clean_text(soup.title.get_text(" ", strip=True))
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" ", strip=True))
    return None


def extract_news_links(
    html: str,
    base_url: str,
    max_links: int | None = None,
) -> list[CandidateLink]:
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    links: list[CandidateLink] = []
    seen: dict[str, int] = {}

    for anchor in soup.find_all("a", href=True):
        raw_href = anchor.get("href", "")
        resolved = resolve_link(raw_href, base_url)
        if not resolved:
            continue

        text = _anchor_text(anchor)
        pattern = classify_news_url(resolved, text, anchor)
        if not pattern:
            continue
        metadata = _listing_metadata(anchor, resolved, base_url, pattern, text)
        candidate = CandidateLink(
            url=resolved,
            raw_href=raw_href,
            text=text,
            pattern=pattern,
            relative_href=is_relative_href(raw_href),
            title=metadata["title"],
            summary=metadata["summary"],
            date_label_raw=metadata["date_label_raw"],
            publisher_unit=metadata["publisher_unit"],
            image_url=metadata["image_url"],
        )

        if resolved in seen:
            existing_index = seen[resolved]
            if _candidate_quality(candidate) > _candidate_quality(links[existing_index]):
                links[existing_index] = replace(candidate)
            continue

        seen[resolved] = len(links)
        links.append(candidate)
        if max_links is not None and len(links) >= max_links:
            break

    return links


def detect_last_listing_page(html: str, source_type: str) -> int | None:
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    pages: set[int] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        parsed = urlparse(href)
        if source_type in {"legacy_ip", "old_subdomain_portal"}:
            for value in parse_qs(parsed.query).get("page", []):
                if value.isdigit():
                    pages.add(int(value))
            match = re.search(r"[?&]page=(\d+)", href)
            if match:
                pages.add(int(match.group(1)))
        else:
            match = re.search(r"/page/(\d+)/?", parsed.path)
            if match:
                pages.add(int(match.group(1)))
        text = clean_text(anchor.get_text(" ", strip=True))
        if text.isdigit() and source_type != "old_subdomain_portal":
            pages.add(int(text))
    return max(pages) if pages else None


def count_news_links(html: str, base_url: str) -> int:
    return len(extract_news_links(html=html, base_url=base_url, max_links=None))


def classify_news_url(url: str, text: str, anchor) -> str | None:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path.rstrip("/") + "/"
    path_lower = path.lower()

    if has_static_extension(url):
        return None
    if not _is_ifmt_host(host):
        return None
    if _is_archive_or_taxonomy_path(path_lower):
        return None
    if _is_navigation_context(anchor):
        return None

    if re.match(r"^/conteudo/noticia/[^/]+/$", path_lower):
        return "legacy_article"
    if re.match(r"^/blog/[^/]+/$", path_lower):
        return "current_wordpress_article"
    if re.search(r"/conteudo/noticia/[^/]+/", path_lower):
        return "legacy_article"
    if re.search(r"/noticia(s)?/[^/]+/", path_lower):
        return "article_like_path"
    if _is_article_context(anchor) and len(text) >= 18 and _has_article_like_slug(path_lower):
        return "article_context"
    return None


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _anchor_text(anchor) -> str:
    candidates = [
        anchor.get_text(" ", strip=True),
        anchor.get("title"),
        anchor.get("aria-label"),
    ]
    image = anchor.find("img")
    if image is not None:
        candidates.extend([image.get("alt"), image.get("title")])
    for candidate in candidates:
        text = clean_text(str(candidate or ""))
        if text:
            return text
    return ""


def _listing_metadata(anchor, url: str, base_url: str, pattern: str, text: str) -> dict[str, str]:
    if pattern == "legacy_article":
        return _legacy_listing_metadata(text)

    container = _article_container(anchor)
    title = text
    summary = ""
    date_label_raw = ""
    image_url = ""

    if container is not None:
        title = _best_title_from_container(container, anchor, text)
        summary = _summary_from_container(container, title)
        date_label_raw = _date_from_container(container)
        image_url = _image_from_container(container, base_url)

    return {
        "title": title,
        "summary": summary,
        "date_label_raw": date_label_raw,
        "publisher_unit": "",
        "image_url": image_url,
    }


def _legacy_listing_metadata(text: str) -> dict[str, str]:
    cleaned = clean_text(text)
    match = re.match(
        r"^(?P<date>\d{1,2}\s+[A-Za-zÀ-ÿ]{3,})\s*-\s*Publicado por\s+(?P<rest>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return {
            "title": cleaned,
            "summary": "",
            "date_label_raw": "",
            "publisher_unit": "",
            "image_url": "",
        }

    rest = clean_text(match.group("rest"))
    publisher_unit = ""
    title = rest
    for publisher in sorted(KNOWN_LEGACY_PUBLISHERS, key=len, reverse=True):
        if _strip_accents(rest).lower().startswith(_strip_accents(publisher).lower()):
            publisher_unit = publisher
            title = clean_text(rest[len(publisher) :])
            break

    if not publisher_unit and rest.lower().startswith("campus "):
        words = rest.split()
        publisher_unit = " ".join(words[:3]) if len(words) >= 3 else rest
        title = clean_text(rest[len(publisher_unit) :])

    return {
        "title": title or rest,
        "summary": "",
        "date_label_raw": clean_text(match.group("date")),
        "publisher_unit": publisher_unit,
        "image_url": "",
    }


def _article_container(anchor):
    current = anchor
    for _ in range(7):
        current = current.parent
        if current is None:
            return anchor.parent
        tokens = _node_tokens(current)
        if current.name in {"article", "li"}:
            return current
        if any(any(hint in token for hint in ARTICLE_CONTEXT_HINTS) for token in tokens):
            return current
    return anchor.parent


def _best_title_from_container(container, anchor, fallback: str) -> str:
    for tag_name in ("h1", "h2", "h3", "h4"):
        heading = container.find(tag_name)
        if heading:
            text = clean_text(heading.get_text(" ", strip=True))
            if text:
                return text
    return fallback or _anchor_text(anchor)


def _summary_from_container(container, title: str) -> str:
    preferred = container.find(
        class_=re.compile(r"(excerpt|resumo|summary|descricao|description)", re.I)
    )
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(container.find_all("p"))
    title_norm = _strip_accents(title).lower()
    for candidate in candidates:
        text = clean_text(candidate.get_text(" ", strip=True))
        if not text:
            continue
        text_norm = _strip_accents(text).lower()
        if text_norm == title_norm:
            continue
        if _looks_like_date(text):
            continue
        return text
    return ""


def _date_from_container(container) -> str:
    time_tag = container.find("time")
    if time_tag is not None:
        text = clean_text(time_tag.get_text(" ", strip=True))
        if text:
            return text
        datetime_value = time_tag.get("datetime")
        if datetime_value:
            return clean_text(str(datetime_value))

    preferred = container.find(class_=re.compile(r"(date|data|posted|published)", re.I))
    if preferred is not None:
        text = clean_text(preferred.get_text(" ", strip=True))
        if text:
            return text

    text = clean_text(container.get_text(" ", strip=True))
    patterns = (
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+[A-Za-zÀ-ÿçÇ]+(?:\s+de\s+\d{4})?\b",
        r"\b\d{1,2}\s+[A-Za-zÀ-ÿçÇ]{3,}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_text(match.group(0))
    return ""


def _image_from_container(container, base_url: str) -> str:
    image = container.find("img")
    if image is None:
        return ""
    for attr in ("src", "data-src", "data-lazy-src", "data-original"):
        value = image.get(attr)
        if value:
            resolved = resolve_link(str(value), base_url)
            if resolved:
                return resolved
    srcset = image.get("srcset")
    if srcset:
        first = str(srcset).split(",")[0].strip().split(" ")[0]
        resolved = resolve_link(first, base_url)
        if resolved:
            return resolved
    return ""


def _candidate_quality(candidate: CandidateLink) -> int:
    return sum(
        1
        for value in (
            candidate.text,
            candidate.title,
            candidate.summary,
            candidate.date_label_raw,
            candidate.publisher_unit,
            candidate.image_url,
        )
        if value
    )


def _looks_like_date(text: str) -> bool:
    return bool(
        re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", text)
        or re.fullmatch(r"\d{1,2}\s+[A-Za-zÀ-ÿçÇ]{3,}", text, flags=re.IGNORECASE)
    )


def _strip_accents(value: str) -> str:
    replacements = {
        "á": "a",
        "à": "a",
        "â": "a",
        "ã": "a",
        "ä": "a",
        "é": "e",
        "ê": "e",
        "è": "e",
        "í": "i",
        "ì": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ö": "o",
        "ú": "u",
        "ü": "u",
        "ç": "c",
    }
    translated = []
    for char in value:
        lowered = char.lower()
        replacement = replacements.get(lowered)
        if replacement is None:
            translated.append(char)
        elif char.isupper():
            translated.append(replacement.upper())
        else:
            translated.append(replacement)
    return "".join(translated)


def _is_ifmt_host(host: str) -> bool:
    return host == LEGACY_IP_HOST or host == "ifmt.edu.br" or host.endswith(".ifmt.edu.br")


def _is_archive_or_taxonomy_path(path_lower: str) -> bool:
    blocked_fragments = (
        "/categoria/",
        "/category/",
        "/tag/",
        "/author/",
        "/feed/",
        "/wp-content/",
        "/wp-json/",
    )
    if any(fragment in path_lower for fragment in blocked_fragments):
        return True
    return path_lower in {"/", "/blog/", "/conteudo/noticias/", "/noticias/"}


def _context_tokens(anchor) -> set[str]:
    tokens: set[str] = set()
    current = anchor
    for _ in range(5):
        current = current.parent
        if current is None:
            break
        for attr in ("class", "id", "role"):
            value = current.get(attr)
            if isinstance(value, list):
                tokens.update(str(item).lower() for item in value)
            elif value:
                tokens.add(str(value).lower())
        if getattr(current, "name", None):
            tokens.add(str(current.name).lower())
    return tokens


def _node_tokens(node) -> set[str]:
    tokens: set[str] = set()
    for attr in ("class", "id", "role"):
        value = node.get(attr)
        if isinstance(value, list):
            tokens.update(str(item).lower() for item in value)
        elif value:
            tokens.add(str(value).lower())
    if getattr(node, "name", None):
        tokens.add(str(node.name).lower())
    return tokens


def _is_navigation_context(anchor) -> bool:
    tokens = _context_tokens(anchor)
    return any(any(hint in token for hint in NAV_CONTEXT_HINTS) for token in tokens)


def _is_article_context(anchor) -> bool:
    tokens = _context_tokens(anchor)
    return any(any(hint in token for hint in ARTICLE_CONTEXT_HINTS) for token in tokens)


def _has_article_like_slug(path_lower: str) -> bool:
    parts = [part for part in path_lower.split("/") if part]
    if not parts:
        return False
    slug = parts[-1]
    if len(slug) < 8:
        return False
    if slug.isdigit():
        return False
    return "-" in slug or len(parts) >= 2
