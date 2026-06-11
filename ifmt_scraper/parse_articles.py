from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .config import DISCOVERY_SOURCES, LEGACY_IP_HOST
from .normalize import (
    calculate_word_count,
    clean_body_text,
    extract_lead,
    normalize_text,
    normalize_title,
    parse_publication_datetime,
    sha256_text,
)
from .url_utils import normalize_url, resolve_link, slug_from_url


DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".rar",
}

OLDPORTAL_CONTENT_TOKEN = re.compile(
    r"(content|conteudo|noticia|post|texto|principal|miolo|barra_central|corpo)",
    re.IGNORECASE,
)

DOCUMENT_LINK_KEYWORDS = {
    "edital",
    "anexo",
    "formulario",
    "ficha",
    "resultado",
    "cronograma",
    "retificacao",
    "arquivo",
    "documento",
    "clique aqui",
    "acesse aqui",
    "chamada",
    "lista",
    "inscricao",
    "homologacao",
    "convocacao",
}

FORM_OR_SELECTION_HOSTS = {
    "forms.gle",
    "sistemas.ifmt.edu.br",
    "seletivo.ifmt.edu.br",
    "processoseletivo.ifmt.edu.br",
    "selecao.ifmt.edu.br",
}

NOISE_SELECTOR = re.compile(
    r"(menu|nav|navbar|footer|header|sidebar|breadcrumb|share|social|"
    r"related|comments?|comment-form|pagination|paginacao|últimas|ultimas|"
    r"elementor-location-header|elementor-location-footer)",
    re.IGNORECASE,
)


@dataclass
class ParsedArticle:
    article: dict[str, Any]
    media_assets: list[dict[str, Any]] = field(default_factory=list)
    article_links: list[dict[str, Any]] = field(default_factory=list)


def parse_article_html(
    html: str,
    discovered_record: dict[str, Any],
    request_url: str,
    final_url: str | None,
    http_status: int | None,
    redirected: bool,
    used_ip_access: bool,
    ssl_verify_disabled: bool,
) -> ParsedArticle:
    base_url = _link_base_url(request_url, final_url)
    soup = BeautifulSoup(html or "", "lxml")
    oldportal_fields = (
        _extract_oldportal_article_fields(soup, discovered_record)
        if discovered_record.get("source_environment") == "old_subdomain_portal"
        else {}
    )

    title = oldportal_fields.get("title") or _extract_title(soup, discovered_record)
    if (
        discovered_record.get("source_environment") == "old_subdomain_portal"
        and _is_oldportal_bad_title(title)
        and discovered_record.get("title_from_listing")
    ):
        title = normalize_text(discovered_record.get("title_from_listing", ""))
    canonical_url = _extract_canonical_url(soup, base_url)
    author_name, author_url = _extract_author(soup, base_url)
    author_name = oldportal_fields.get("author_name") or author_name
    date_raw = (
        oldportal_fields.get("date_raw")
        or _extract_date_raw(soup)
        or discovered_record.get("date_label_raw", "")
    )
    publication_date, publication_time, parsed_date_raw = parse_publication_datetime(date_raw)
    date_label_raw = parsed_date_raw or discovered_record.get("date_label_raw", "")
    publisher_unit = (
        oldportal_fields.get("publisher_unit")
        or _extract_publisher_unit(soup)
        or discovered_record.get("publisher_unit")
        or ""
    )

    _remove_noise(soup)
    content_node = oldportal_fields.get("content_node") or _find_content_node(
        soup, discovered_record.get("source_environment", "")
    )
    body_html = oldportal_fields.get("body_html") or (str(content_node) if content_node is not None else "")
    body_text = oldportal_fields.get("body_text") or _extract_body_text(content_node)
    subtitle = _extract_subtitle(soup, title, body_text)
    lead = extract_lead(body_text)
    word_count = calculate_word_count(body_text)

    source_id = discovered_record.get("source_id", "")
    campus_name = _campus_name(discovered_record, publisher_unit)
    campus_code = _campus_code(discovered_record)
    institutional_unit = publisher_unit or campus_name or discovered_record.get("source_name", "")
    categories = _extract_terms(soup, "category")
    tags = _extract_terms(soup, "tag")

    article = {
        "article_id": "",
        "discovered_id": discovered_record.get("discovered_id", ""),
        "source_id": source_id,
        "source_name": discovered_record.get("source_name", ""),
        "source_environment": discovered_record.get("source_environment", ""),
        "source_base_url": discovered_record.get("source_base_url", ""),
        "source_host": discovered_record.get("source_host", ""),
        "source_access_mode": discovered_record.get("source_access_mode", ""),
        "discovered_from_url": discovered_record.get("listing_url", ""),
        "discovered_page_number": discovered_record.get("listing_page_number", ""),
        "original_url": discovered_record.get("discovered_url", ""),
        "request_url": request_url,
        "final_url_after_redirect": final_url or "",
        "canonical_url": canonical_url,
        "http_status": http_status,
        "redirected": redirected,
        "used_ip_access": used_ip_access,
        "ssl_verify_disabled": ssl_verify_disabled,
        "slug": slug_from_url(final_url or request_url),
        "title": title,
        "subtitle": subtitle,
        "lead": lead,
        "body_text": body_text,
        "body_html": body_html,
        "publication_date": publication_date,
        "publication_time": publication_time,
        "date_label_raw": date_label_raw,
        "author_name": author_name,
        "author_url": author_url,
        "publisher_unit": publisher_unit,
        "campus_name": campus_name,
        "campus_code": campus_code,
        "institutional_unit": institutional_unit,
        "categories": categories,
        "tags": tags,
        "estimated_reading_time": _estimated_reading_time(word_count),
        "comments_count": "",
        "access_count": "",
        "word_count": word_count,
        "has_images": False,
        "has_links": False,
        "has_attachments": False,
        "content_hash": sha256_text(body_text),
        "title_hash": sha256_text(normalize_title(title)),
        "title_date_hash": sha256_text(
            f"{normalize_title(title)}|{publication_date or date_label_raw}"
        )
        if title and (publication_date or date_label_raw)
        else "",
        "duplicate_group_id": "",
        "duplicate_of": "",
        "is_canonical": True,
        "duplicate_reason": "",
        "duplicate_confidence": 0,
        "collected_at": "",
        "updated_at": "",
        "status": "collected",
        "error_message": "",
    }

    media_assets = _extract_media_assets(
        soup=soup,
        content_node=content_node,
        base_url=base_url,
        featured_image_url=discovered_record.get("image_url_from_listing", ""),
    )
    article_links = _extract_article_links(content_node, base_url, final_url or request_url)

    article["has_images"] = bool(media_assets)
    article["has_links"] = bool(article_links)
    article["has_attachments"] = any(_is_attachment_like_link(link) for link in article_links)

    missing = []
    if not title:
        missing.append("title")
    if not body_text or word_count < 20:
        missing.append("body_text")
    if missing:
        article["status"] = "needs_review"
        article["error_message"] = "Missing or weak extraction: " + ", ".join(missing)

    return ParsedArticle(article=article, media_assets=media_assets, article_links=article_links)


def build_error_article(
    discovered_record: dict[str, Any],
    request_url: str,
    final_url: str | None,
    http_status: int | None,
    redirected: bool,
    used_ip_access: bool,
    ssl_verify_disabled: bool,
    error_message: str,
) -> dict[str, Any]:
    title = discovered_record.get("title_from_listing", "")
    publisher_unit = discovered_record.get("publisher_unit", "")
    campus_name = _campus_name(discovered_record, publisher_unit)
    return {
        "article_id": "",
        "discovered_id": discovered_record.get("discovered_id", ""),
        "source_id": discovered_record.get("source_id", ""),
        "source_name": discovered_record.get("source_name", ""),
        "source_environment": discovered_record.get("source_environment", ""),
        "source_base_url": discovered_record.get("source_base_url", ""),
        "source_host": discovered_record.get("source_host", ""),
        "source_access_mode": discovered_record.get("source_access_mode", ""),
        "discovered_from_url": discovered_record.get("listing_url", ""),
        "discovered_page_number": discovered_record.get("listing_page_number", ""),
        "original_url": discovered_record.get("discovered_url", ""),
        "request_url": request_url,
        "final_url_after_redirect": final_url or "",
        "canonical_url": "",
        "http_status": http_status,
        "redirected": redirected,
        "used_ip_access": used_ip_access,
        "ssl_verify_disabled": ssl_verify_disabled,
        "slug": slug_from_url(request_url),
        "title": title,
        "subtitle": discovered_record.get("summary_from_listing", ""),
        "lead": "",
        "body_text": "",
        "body_html": "",
        "publication_date": "",
        "publication_time": "",
        "date_label_raw": discovered_record.get("date_label_raw", ""),
        "author_name": "",
        "author_url": "",
        "publisher_unit": publisher_unit,
        "campus_name": campus_name,
        "campus_code": _campus_code(discovered_record),
        "institutional_unit": publisher_unit or campus_name or discovered_record.get("source_name", ""),
        "categories": [],
        "tags": [],
        "estimated_reading_time": 0,
        "comments_count": "",
        "access_count": "",
        "word_count": 0,
        "has_images": False,
        "has_links": False,
        "has_attachments": False,
        "content_hash": "",
        "title_hash": sha256_text(normalize_title(title)),
        "title_date_hash": "",
        "duplicate_group_id": "",
        "duplicate_of": "",
        "is_canonical": True,
        "duplicate_reason": "",
        "duplicate_confidence": 0,
        "collected_at": "",
        "updated_at": "",
        "status": "error",
        "error_message": error_message,
    }


def _extract_oldportal_article_fields(
    soup: BeautifulSoup,
    discovered_record: dict[str, Any],
) -> dict[str, Any]:
    content_node = _select_oldportal_content_node(soup, discovered_record)
    if content_node is None:
        return {}

    listing_title = normalize_text(discovered_record.get("title_from_listing", ""))
    title_node = content_node.select_one(".itens_noticia li.atual a, .itens_noticia li.atual, li.atual a, li.atual, h1, h2, h3")
    title = normalize_text(title_node.get_text(" ", strip=True)) if title_node else ""
    if _is_oldportal_bad_title(title) and listing_title:
        title = listing_title
    elif not title:
        title = listing_title

    body_blocks: list[str] = []
    author_name = ""
    in_documents_section = False
    for node in content_node.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"], recursive=True):
        if node.find_parent("li", class_="atual") is not None:
            continue
        text = normalize_text(node.get_text(" ", strip=True))
        if not text or text == title:
            continue
        if _oldportal_is_breadcrumb_text(text):
            continue
        if _is_oldportal_document_heading(text):
            in_documents_section = True
            continue
        if in_documents_section:
            continue
        if _looks_like_oldportal_author(text):
            author_name = text
            continue
        body_blocks.append(text)

    if not body_blocks:
        body_blocks = _oldportal_text_blocks_from_node(content_node, title)

    body_text = clean_body_text(body_blocks)
    if calculate_word_count(body_text) < 80:
        fallback_blocks = _oldportal_body_blocks_from_best_text(content_node, title)
        fallback_text = clean_body_text(fallback_blocks)
        if calculate_word_count(fallback_text) > calculate_word_count(body_text):
            body_text = fallback_text

    date_raw = _first_oldportal_date(
        " ".join(
            [
                title,
                body_text,
                author_name,
                _oldportal_near_title_text(content_node, title),
                discovered_record.get("date_label_raw", ""),
            ]
        )
    )
    publisher_unit = _oldportal_publisher_unit(author_name)

    return {
        "content_node": content_node,
        "body_html": _oldportal_sanitized_html(content_node),
        "title": title,
        "body_text": body_text,
        "date_raw": date_raw,
        "author_name": author_name,
        "publisher_unit": publisher_unit,
    }


def _select_oldportal_content_node(
    soup: BeautifulSoup,
    discovered_record: dict[str, Any],
):
    listing_title = normalize_text(discovered_record.get("title_from_listing", ""))
    title_key = _oldportal_text_key(listing_title)

    for selector in ("#barra_central", ".itens_noticia"):
        for node in soup.select(selector):
            text = normalize_text(node.get_text(" ", strip=True))
            if selector == "#barra_central" and calculate_word_count(text) >= 20 and node.select_one(".itens_noticia, li.atual"):
                return node
            if selector == ".itens_noticia" and calculate_word_count(text) >= 20:
                return node
            if (
                calculate_word_count(text) >= 20
                and (not title_key or title_key in _oldportal_text_key(text))
            ):
                return node

    candidates = _oldportal_content_candidates(soup)
    if not candidates:
        return None

    def score(node) -> int:
        text = normalize_text(node.get_text(" ", strip=True))
        useful_text = _oldportal_useful_text(text)
        paragraph_count = len(node.find_all("p"))
        link_bonus = 35 * len(
            [
                link
                for link in node.find_all("a", href=True)
                if _oldportal_link_looks_documentary(
                    str(link.get("href") or ""),
                    normalize_text(link.get_text(" ", strip=True)),
                )
            ]
        )
        title_bonus = 350 if title_key and title_key in _oldportal_text_key(text) else 0
        date_bonus = 120 if _first_oldportal_date(text) else 0
        preferred_bonus = _oldportal_preferred_node_bonus(node)
        noise_penalty = _oldportal_noise_penalty(node)
        return (
            len(useful_text)
            + paragraph_count * 90
            + title_bonus
            + date_bonus
            + link_bonus
            + preferred_bonus
            - noise_penalty
        )

    best = max(candidates, key=score)
    if score(best) <= 0:
        return None
    return best


def _oldportal_content_candidates(soup: BeautifulSoup) -> list[Any]:
    selectors = (
        "#barra_central",
        ".itens_noticia",
        "main",
        "article",
        "#content",
        "#conteudo",
        "#corpo",
        "#corpo_geral",
        ".content",
        ".conteudo",
        ".noticia",
        ".post",
        ".texto",
        ".principal",
        ".miolo",
    )
    candidates: list[Any] = []
    seen: set[int] = set()

    def add(node) -> None:
        if node is None:
            return
        identity = id(node)
        if identity in seen:
            return
        seen.add(identity)
        candidates.append(node)

    for selector in selectors:
        for node in soup.select(selector):
            add(node)

    for node in soup.find_all(["section", "div"]):
        token_text = " ".join(
            str(value)
            for attr in ("class", "id", "role", "aria-label")
            for value in _attr_values(node.get(attr))
        )
        if token_text and OLDPORTAL_CONTENT_TOKEN.search(token_text):
            add(node)

    title_node = soup.select_one("li.atual a, li.atual, h1, h2")
    if title_node is not None:
        current = title_node
        for _ in range(4):
            current = current.parent
            if current is None:
                break
            if getattr(current, "name", None) in {"div", "section", "article", "main"}:
                add(current)

    return candidates


def _oldportal_preferred_node_bonus(node) -> int:
    token_text = " ".join(
        str(value)
        for attr in ("class", "id")
        for value in _attr_values(node.get(attr))
    ).lower()
    if "barra_central" in token_text:
        return 650
    if "itens_noticia" in token_text:
        return 450
    if "corpo" in token_text:
        return 120
    return 0


def _oldportal_noise_penalty(node) -> int:
    token_text = " ".join(
        str(value)
        for attr in ("class", "id", "role", "aria-label")
        for value in _attr_values(node.get(attr))
    )
    penalty = 0
    if token_text and NOISE_SELECTOR.search(token_text):
        penalty += 1000
    text = _oldportal_text_key(node.get_text(" ", strip=True))
    menu_hits = sum(
        1
        for phrase in (
            "acesso a informacao",
            "institucional",
            "processo seletivo",
            "servicos",
            "sistemas",
            "boletim de servicos",
            "gerenciamento do site",
        )
        if phrase in text
    )
    penalty += menu_hits * 90
    return penalty


def _oldportal_useful_text(text: str) -> str:
    lines = []
    for line in normalize_text(text).splitlines():
        cleaned = normalize_text(line)
        if not cleaned:
            continue
        key = _oldportal_text_key(cleaned)
        if key in {"inicio", "noticias", "documentos"}:
            continue
        if key.startswith("instituto federal de educacao ciencia e tecnologia"):
            continue
        lines.append(cleaned)
    return "\n".join(lines)


def _oldportal_body_blocks_from_best_text(content_node, title: str) -> list[str]:
    blocks: list[str] = []
    title_key = _oldportal_text_key(title)
    seen_title = not bool(title_key)
    in_documents_section = False

    for line in content_node.get_text("\n", strip=True).splitlines():
        text = normalize_text(line)
        if not text:
            continue
        key = _oldportal_text_key(text)
        if title_key and key == title_key:
            seen_title = True
            continue
        if not seen_title:
            continue
        if _oldportal_is_breadcrumb_text(text):
            continue
        if _is_oldportal_document_heading(text):
            in_documents_section = True
            continue
        if in_documents_section:
            continue
        if key in {"cursos", "a rede", "noticias", "inicio"}:
            break
        if _looks_like_oldportal_author(text):
            continue
        blocks.append(text)
    return blocks


def _oldportal_near_title_text(content_node, title: str) -> str:
    title_key = _oldportal_text_key(title)
    if not title_key:
        return normalize_text(content_node.get_text(" ", strip=True))[:1000]
    lines = [
        normalize_text(line)
        for line in content_node.get_text("\n", strip=True).splitlines()
        if normalize_text(line)
    ]
    for index, line in enumerate(lines):
        key = _oldportal_text_key(line)
        if key == title_key or title_key in key or key in title_key:
            return " ".join(lines[index : index + 8])
    return " ".join(lines[:12])


def _oldportal_text_blocks_from_node(content_node, title: str) -> list[str]:
    blocks = []
    seen_title = False
    title_key = _oldportal_text_key(title)
    in_documents_section = False
    for line in content_node.get_text("\n", strip=True).splitlines():
        text = normalize_text(line)
        if not text:
            continue
        key = _oldportal_text_key(text)
        if title_key and key == title_key:
            seen_title = True
            continue
        if not seen_title and title_key:
            continue
        if _oldportal_is_breadcrumb_text(text):
            continue
        if _is_oldportal_document_heading(text):
            in_documents_section = True
            continue
        if in_documents_section:
            continue
        if key in {"cursos", "a rede"}:
            break
        if _looks_like_oldportal_author(text):
            continue
        blocks.append(text)
    return blocks


def _oldportal_sanitized_html(content_node) -> str:
    soup = BeautifulSoup(str(content_node), "lxml")
    for tag in soup.find_all(True):
        for attr in ("src", "href", "srcset"):
            value = tag.get(attr)
            if value and str(value).lower().startswith("data:"):
                tag[attr] = "[embedded-data-uri]"
    body = soup.body
    if body:
        children = [child for child in body.children if getattr(child, "name", None)]
        if len(children) == 1:
            return str(children[0])
    return str(body or soup)


def _is_oldportal_bad_title(title: str) -> bool:
    key = _oldportal_text_key(title)
    if key.startswith("instituto federal de educacao ciencia e tecnologia"):
        return True
    return key in {
        "",
        "acesso a informacao",
        "institucional",
        "estudante",
        "eventos",
        "servicos",
        "biblioteca",
        "processo seletivo",
        "sistemas",
        "email institucional",
        "ativa incubadora",
        "cursos",
        "a rede",
        "documentos",
    }


def _looks_like_oldportal_author(text: str) -> bool:
    key = _oldportal_text_key(text)
    if key.startswith("site desativado"):
        return False
    return bool(
        re.search(r"\b(ascom|assessoria|comunica)", key)
        or (
            "ifmt" in key
            and "/" in text
            and len(text) <= 140
            and len(text.split()) <= 10
        )
    )


def _oldportal_publisher_unit(author_name: str) -> str:
    key = _oldportal_text_key(author_name)
    if "reitoria" in key:
        return "Reitoria"
    match = re.search(r"campus\s+([A-Za-zÀ-ÿ0-9 \-]+)", author_name, flags=re.IGNORECASE)
    if match:
        return normalize_text(f"Campus {match.group(1)}")
    return ""


def _first_oldportal_date(text: str) -> str:
    labeled_patterns = (
        r"(?:Publicado em|Publicado:|Data de publica(?:\w+)?:?)\s*(\d{1,2}/\d{1,2}/\d{2,4}(?:\s+\d{1,2}:\d{2})?)",
        r"(?:Publicado em|Publicado:|Data de publica(?:\w+)?:?)\s*(\d{1,2}\s+de\s+[\wÀ-ÿ]+(?:\s+de\s+\d{4})?(?:\s+(?:as|às|Ã s)\s+\d{1,2}:\d{2})?)",
    )
    for pattern in labeled_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(1))

    broad_patterns = (
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+[\wÀ-ÿ]+(?:\s+de\s+\d{4})?(?:\s+(?:as|às|Ã s)\s+\d{1,2}:\d{2})?\b",
        r"\b\d{1,2}\s+[\wÀ-ÿ]{3,}\b",
    )
    for pattern in broad_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(0))
    return ""

    patterns = (
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+[A-Za-zÀ-ÿçÇ]+(?:\s+de\s+\d{4})?\b",
        r"\b\d{1,2}\s+[A-Za-zÀ-ÿçÇ]{3,}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(0))
    return ""


def _is_oldportal_document_heading(text: str) -> bool:
    key = _oldportal_text_key(text)
    lowered = text.lower()
    if key in {"documentos", "documentos:"}:
        return True
    if lowered.rstrip(".").endswith(
        (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar")
    ):
        return True
    return False


def _oldportal_is_breadcrumb_text(text: str) -> bool:
    key = _oldportal_text_key(text)
    return key in {"inicio", "noticias", "inicio noticias"} or text.strip().startswith(">>")


def _oldportal_link_looks_documentary(raw_href: str, anchor_text: str) -> bool:
    value = f"{raw_href} {anchor_text}"
    key = normalize_title(value)
    if any(keyword in key for keyword in DOCUMENT_LINK_KEYWORDS):
        return True
    lowered = value.lower()
    return any(ext in lowered for ext in DOCUMENT_EXTENSIONS) or "/get_file/" in lowered


def _oldportal_text_key(value: str) -> str:
    return normalize_title(value)
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9À-ÿ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_title(soup: BeautifulSoup, discovered_record: dict[str, Any]) -> str:
    for h1 in soup.find_all("h1"):
        text = normalize_text(h1.get_text(" ", strip=True))
        if text:
            return text

    for selector in (
        'meta[property="og:title"]',
        'meta[name="twitter:title"]',
        'meta[name="title"]',
    ):
        meta = soup.select_one(selector)
        if meta and meta.get("content"):
            return _clean_title_suffix(str(meta["content"]))

    if soup.title and soup.title.get_text(strip=True):
        return _clean_title_suffix(soup.title.get_text(" ", strip=True))

    return normalize_text(discovered_record.get("title_from_listing", ""))


def _extract_canonical_url(soup: BeautifulSoup, base_url: str) -> str:
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical and canonical.get("href"):
        return resolve_link(str(canonical["href"]), base_url) or ""
    return ""


def _extract_author(soup: BeautifulSoup, base_url: str) -> tuple[str, str]:
    meta = soup.find("meta", attrs={"name": "author"})
    if meta and meta.get("content"):
        return normalize_text(str(meta["content"])), ""

    rel_author = soup.find("a", rel=lambda value: value and "author" in value)
    if rel_author:
        return (
            normalize_text(rel_author.get_text(" ", strip=True)),
            resolve_link(str(rel_author.get("href", "")), base_url) or "",
        )

    node = soup.find(class_=re.compile(r"author|autor", re.IGNORECASE))
    if node:
        return normalize_text(node.get_text(" ", strip=True)), ""
    return "", ""


def _extract_date_raw(soup: BeautifulSoup) -> str:
    for selector in (
        'meta[property="article:published_time"]',
        'meta[name="date"]',
        'meta[name="pubdate"]',
        'meta[itemprop="datePublished"]',
    ):
        meta = soup.select_one(selector)
        if meta and meta.get("content"):
            return normalize_text(str(meta["content"]))

    time_parts: list[str] = []
    for time_tag in soup.find_all("time"):
        if time_tag.get("datetime"):
            return normalize_text(str(time_tag["datetime"]))
        text = normalize_text(time_tag.get_text(" ", strip=True))
        if text:
            time_parts.append(text)
    if time_parts:
        return " ".join(time_parts[:2])

    text = normalize_text(soup.get_text(" ", strip=True))
    legacy = re.search(
        r"Publicado por:\s*[^/]+/\s*([0-9]{1,2}\s+de\s+[A-Za-zÀ-ÿçÇ]+\s+de\s+\d{4}\s+às\s+\d{1,2}:\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    if legacy:
        return legacy.group(1)

    patterns = (
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(?:am|pm)?\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+[A-Za-zÀ-ÿçÇ]+\s+de\s+\d{4}(?:\s+às\s+\d{1,2}:\d{2})?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(0))
    return ""


def _extract_publisher_unit(soup: BeautifulSoup) -> str:
    text = normalize_text(soup.get_text(" ", strip=True))
    match = re.search(r"Publicado por:\s*([^/]+)", text, flags=re.IGNORECASE)
    if match:
        return normalize_text(match.group(1))
    return ""


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style", "noscript", "svg", "form", "button"]):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        if tag.name in {"html", "body"}:
            continue
        if not getattr(tag, "attrs", None):
            continue
        token_text = " ".join(
            str(value)
            for attr in ("class", "id", "role", "aria-label")
            for value in _attr_values(tag.get(attr))
        )
        token_text = re.sub(r"\bno-sidebar\b", "", token_text, flags=re.IGNORECASE)
        if token_text and NOISE_SELECTOR.search(token_text):
            tag.decompose()


def _find_content_node(soup: BeautifulSoup, source_environment: str):
    preferred_selectors = (
        ".elementor-widget-theme-post-content",
        ".entry-content",
        ".post-content",
        ".single-post-content",
        "article",
        "main article",
        "main",
        "#content",
        ".content",
    )
    for selector in preferred_selectors:
        for node in soup.select(selector):
            if _node_text_score(node) >= 80:
                return node

    candidates = soup.find_all(["article", "main", "section", "div"])
    if not candidates:
        return soup.body or soup
    return max(candidates, key=_node_text_score, default=soup.body or soup)


def _extract_body_text(content_node) -> str:
    if content_node is None:
        return ""

    blocks: list[str] = []
    for node in content_node.find_all(["h2", "h3", "h4", "p", "li", "blockquote"]):
        text = normalize_text(node.get_text(" ", strip=True))
        if text:
            blocks.append(text)
    if not blocks:
        blocks = [normalize_text(content_node.get_text("\n\n", strip=True))]
    return clean_body_text(blocks)


def _extract_subtitle(soup: BeautifulSoup, title: str, body_text: str) -> str:
    for selector in (
        'meta[property="og:description"]',
        'meta[name="description"]',
        ".subtitle",
        ".subtitulo",
        ".lead",
        ".excerpt",
    ):
        node = soup.select_one(selector)
        if not node:
            continue
        text = normalize_text(str(node.get("content", "")) if node.name == "meta" else node.get_text(" ", strip=True))
        if text and text != title and text not in body_text[:300]:
            return text
    return ""


def _extract_terms(soup: BeautifulSoup, term_type: str) -> list[str]:
    terms: list[str] = []
    rel_name = "tag" if term_type == "tag" else "category"
    for anchor in soup.find_all("a", rel=lambda value: value and rel_name in value):
        text = normalize_text(anchor.get_text(" ", strip=True))
        if text and text not in terms:
            terms.append(text)

    class_pattern = re.compile(r"tag|categoria|category", re.IGNORECASE)
    for node in soup.find_all(class_=class_pattern):
        text = normalize_text(node.get_text(" ", strip=True))
        if text and len(text) < 80 and text not in terms:
            terms.append(text)
    return terms[:20]


def _extract_media_assets(
    soup: BeautifulSoup,
    content_node,
    base_url: str,
    featured_image_url: str,
) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    seen: set[str] = set()

    og_image = soup.select_one('meta[property="og:image"]')
    featured_candidates = [featured_image_url]
    if og_image and og_image.get("content"):
        featured_candidates.append(str(og_image["content"]))

    for candidate in featured_candidates:
        url = resolve_link(candidate, base_url) if candidate else None
        if url:
            _add_media(media, seen, url, "featured_image", "", "", 0)

    if content_node is not None:
        position = 1
        for image in content_node.find_all("img"):
            url = _image_url(image, base_url)
            if not url:
                continue
            alt_text = normalize_text(str(image.get("alt", "")))
            caption = _image_caption(image)
            media_type = "gallery_image" if _has_gallery_parent(image) else "inline_image"
            _add_media(media, seen, url, media_type, alt_text, caption, position)
            position += 1
    return media


def _extract_article_links(content_node, base_url: str, article_url: str) -> list[dict[str, Any]]:
    if content_node is None:
        return []

    links: list[dict[str, Any]] = []
    seen: set[str] = set()
    article_host = urlparse(article_url).hostname or ""
    for anchor in content_node.find_all("a", href=True):
        url = resolve_link(str(anchor.get("href", "")), base_url)
        if not url:
            continue
        normalized = normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        parsed = urlparse(url)
        extension = PurePosixPath(parsed.path).suffix.lower()
        anchor_text = normalize_text(anchor.get_text(" ", strip=True))
        link_type = _classify_link(url, anchor_text, article_host, extension)
        should_download = link_type == "document" and extension in DOCUMENT_EXTENSIONS
        links.append(
            {
                "link_id": "",
                "article_id": "",
                "url": url,
                "normalized_url": normalized,
                "anchor_text": anchor_text,
                "link_type": link_type,
                "domain": parsed.hostname or "",
                "file_extension": extension,
                "should_download": should_download,
                "downloaded": False,
                "notes": "",
            }
        )
    return links


def _classify_link(url: str, anchor_text: str, article_host: str, extension: str) -> str:
    lowered = url.lower()
    text_key = normalize_title(anchor_text)
    combined_key = normalize_title(f"{anchor_text} {url}")
    host = urlparse(url).hostname or ""
    path = urlparse(url).path.lower()
    if "forms.gle" in lowered or "docs.google.com/forms" in lowered:
        return "form"
    if "drive.google.com" in lowered or "docs.google.com" in lowered:
        return "drive"
    if host in FORM_OR_SELECTION_HOSTS or host.endswith(".sistemas.ifmt.edu.br"):
        return "form"
    if extension in DOCUMENT_EXTENSIONS:
        return "document"
    if _is_institutional_document_path(path):
        return "document"
    if _link_text_has_document_keyword(combined_key) and _is_institutional_document_link(host, path):
        return "document"
    if any(domain in lowered for domain in ("youtube.com", "youtu.be", "vimeo.com")):
        return "video"
    if any(domain in lowered for domain in ("meet.google.com", "zoom.us", "teams.microsoft.com")):
        return "meeting"
    if "edital" in lowered or "edital" in text_key:
        return "edital"
    if host == article_host or host == LEGACY_IP_HOST or host.endswith(".ifmt.edu.br") or host == "ifmt.edu.br":
        return "internal"
    return "external"


def _link_text_has_document_keyword(value_key: str) -> bool:
    return any(keyword in value_key for keyword in DOCUMENT_LINK_KEYWORDS)


def _is_institutional_document_link(host: str, path: str) -> bool:
    host = (host or "").lower()
    path = (path or "").lower()
    if _is_institutional_document_path(path):
        return True
    return host == LEGACY_IP_HOST or host == "ifmt.edu.br" or host.endswith(".ifmt.edu.br")


def _is_institutional_document_path(path: str) -> bool:
    return any(marker in (path or "").lower() for marker in ("/get_file/", "/download", "/arquivo", "/document"))


def _is_attachment_like_link(link: dict[str, Any]) -> bool:
    return str(link.get("link_type") or "") in {"document", "edital", "drive"}


def _add_media(
    media: list[dict[str, Any]],
    seen: set[str],
    url: str,
    media_type: str,
    alt_text: str,
    caption: str,
    position: int,
) -> None:
    normalized = normalize_url(url)
    if normalized in seen:
        return
    seen.add(normalized)
    media.append(
        {
            "media_id": "",
            "article_id": "",
            "media_url": url,
            "normalized_media_url": normalized,
            "local_path": "",
            "media_type": media_type,
            "alt_text": alt_text,
            "caption": caption,
            "position_in_article": position,
            "should_download": False,
            "downloaded": False,
            "error_message": "",
        }
    )


def _image_url(image, base_url: str) -> str:
    for attr in ("src", "data-src", "data-lazy-src", "data-original"):
        value = image.get(attr)
        if value:
            return resolve_link(str(value), base_url) or ""
    srcset = image.get("srcset")
    if srcset:
        first = str(srcset).split(",")[0].strip().split(" ")[0]
        return resolve_link(first, base_url) or ""
    return ""


def _image_caption(image) -> str:
    figure = image.find_parent("figure")
    if figure:
        caption = figure.find("figcaption")
        if caption:
            return normalize_text(caption.get_text(" ", strip=True))
    parent = image.parent
    if parent:
        caption = parent.find(class_=re.compile(r"caption|legenda", re.IGNORECASE))
        if caption:
            return normalize_text(caption.get_text(" ", strip=True))
    return ""


def _has_gallery_parent(image) -> bool:
    current = image
    for _ in range(4):
        current = current.parent
        if current is None:
            return False
        token_text = " ".join(str(value) for attr in ("class", "id") for value in _attr_values(current.get(attr)))
        if "gallery" in token_text.lower() or "galeria" in token_text.lower():
            return True
    return False


def _link_base_url(request_url: str, final_url: str | None) -> str:
    if urlparse(request_url).hostname == LEGACY_IP_HOST:
        if final_url and urlparse(final_url).hostname == LEGACY_IP_HOST:
            return final_url
        return request_url
    return final_url or request_url


def _node_text_score(node) -> int:
    paragraphs = node.find_all("p") if hasattr(node, "find_all") else []
    text = normalize_text(node.get_text(" ", strip=True)) if hasattr(node, "get_text") else ""
    return len(text) + len(paragraphs) * 100


def _attr_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _clean_title_suffix(title: str) -> str:
    title = normalize_text(title)
    return re.sub(r"\s+-\s+IFMT(?:\s+\|.*)?$", "", title).strip()


def _estimated_reading_time(word_count: int) -> int:
    if word_count <= 0:
        return 0
    return max(1, round(word_count / 200))


def _campus_name(discovered_record: dict[str, Any], publisher_unit: str) -> str:
    source = DISCOVERY_SOURCES.get(discovered_record.get("source_id", ""))
    if source and source.campus_name:
        return source.campus_name
    if publisher_unit.lower().startswith("campus "):
        return publisher_unit
    return ""


def _campus_code(discovered_record: dict[str, Any]) -> str:
    source = DISCOVERY_SOURCES.get(discovered_record.get("source_id", ""))
    return source.campus_code if source and source.campus_code else ""
