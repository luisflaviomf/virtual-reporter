from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .crawl import NEWS_ARTICLE_FIELDS
from .normalize import normalize_text, normalize_title
from .url_utils import normalize_url, slug_from_url

try:  # rapidfuzz is optional but listed in requirements.
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only without optional dep.
    fuzz = None


DEDUPED_ARTICLE_FIELDS = [
    field
    for field in NEWS_ARTICLE_FIELDS
    if field
    not in {
        "duplicate_group_id",
        "duplicate_of",
        "is_canonical",
        "duplicate_reason",
        "duplicate_confidence",
    }
] + [
    "duplicate_group_id",
    "duplicate_of",
    "is_canonical",
    "duplicate_reason",
    "duplicate_confidence",
]

DUPLICATE_GROUP_FIELDS = [
    "duplicate_group_id",
    "canonical_article_id",
    "duplicate_count",
    "duplicate_confidence",
    "duplicate_reasons",
    "article_ids",
    "titles",
    "publication_dates",
    "original_urls",
]


@dataclass
class DedupeOptions:
    input_path: Path = Path("data/output/news_articles_latest.jsonl")
    output_dir: Path = Path("data/output")
    output_prefix: str = "news_articles"
    title_similarity_threshold: int = 95
    body_similarity_threshold: int = 90


class DisjointSet:
    def __init__(self, ids: Iterable[str]) -> None:
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def run_dedupe(options: DedupeOptions) -> dict[str, Any]:
    articles = _read_articles(options.input_path)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(options.output_dir, options.output_prefix)

    for index, article in enumerate(articles, start=1):
        article.setdefault("article_id", f"art_{index:08d}")
        _clear_duplicate_fields(article)

    article_ids = [article["article_id"] for article in articles]
    dsu = DisjointSet(article_ids)
    pair_reasons: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    by_id = {article["article_id"]: article for article in articles}

    _union_exact_keys(articles, dsu, pair_reasons)
    _union_approximate(articles, dsu, pair_reasons, options)

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        components[dsu.find(article["article_id"])].append(article)

    duplicate_groups: list[dict[str, Any]] = []
    duplicate_index = 1
    for component in components.values():
        if len(component) == 1:
            component[0]["is_canonical"] = True
            continue

        canonical = choose_canonical(component)
        group_id = f"dup_{duplicate_index:08d}"
        duplicate_index += 1
        reasons, confidence = _component_reasons(component, pair_reasons)

        for article in component:
            article["duplicate_group_id"] = group_id
            article["duplicate_of"] = "" if article["article_id"] == canonical["article_id"] else canonical["article_id"]
            article["is_canonical"] = article["article_id"] == canonical["article_id"]
            article["duplicate_reason"] = "; ".join(reasons)
            article["duplicate_confidence"] = round(confidence, 4)

        duplicate_groups.append(
            {
                "duplicate_group_id": group_id,
                "canonical_article_id": canonical["article_id"],
                "duplicate_count": len(component),
                "duplicate_confidence": round(confidence, 4),
                "duplicate_reasons": reasons,
                "article_ids": [article["article_id"] for article in component],
                "titles": [article.get("title", "") for article in component],
                "publication_dates": [article.get("publication_date", "") for article in component],
                "original_urls": [article.get("original_url", "") for article in component],
            }
        )

    _write_jsonl(output_paths["deduped_jsonl"], articles)
    _write_csv(output_paths["deduped_csv"], DEDUPED_ARTICLE_FIELDS, articles)
    _write_csv(output_paths["groups_csv"], DUPLICATE_GROUP_FIELDS, duplicate_groups)
    output_paths["groups_json"].write_text(
        json.dumps(duplicate_groups, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "total_articles": len(articles),
        "total_duplicate_groups": len(duplicate_groups),
        "total_duplicate_articles": sum(max(0, len(group["article_ids"]) - 1) for group in duplicate_groups),
        "output_paths": {key: str(path) for key, path in output_paths.items()},
        "examples_duplicate_groups": duplicate_groups[:10],
    }
    return report


def print_dedupe_report(report: dict[str, Any]) -> None:
    print("")
    print("IFMT dedupe report")
    print(f"Total articles: {report['total_articles']}")
    print(f"Duplicate groups: {report['total_duplicate_groups']}")
    print(f"Duplicate articles: {report['total_duplicate_articles']}")
    print(f"Deduped JSONL: {report['output_paths']['deduped_jsonl']}")
    print(f"Deduped CSV: {report['output_paths']['deduped_csv']}")
    print(f"Duplicate groups CSV: {report['output_paths']['groups_csv']}")
    print(f"Duplicate groups JSON: {report['output_paths']['groups_json']}")
    print("")


def choose_canonical(component: list[dict[str, Any]]) -> dict[str, Any]:
    def score(article: dict[str, Any]) -> tuple[Any, ...]:
        return (
            article.get("status") == "collected",
            int(article.get("word_count") or 0),
            bool(article.get("publication_date")),
            bool(article.get("author_name") or article.get("publisher_unit")),
            bool(article.get("has_images")),
            bool(article.get("has_links")),
            _source_richness(article),
            -component.index(article),
        )

    return max(component, key=score)


def _union_exact_keys(
    articles: list[dict[str, Any]],
    dsu: DisjointSet,
    pair_reasons: dict[tuple[str, str], list[tuple[str, float]]],
) -> None:
    keys: dict[str, dict[str, list[str]]] = {
        "normalized_url": defaultdict(list),
        "slug": defaultdict(list),
        "title_date": defaultdict(list),
        "content_hash": defaultdict(list),
    }

    for article in articles:
        article_id = article["article_id"]
        url_key = _article_url_key(article)
        if url_key:
            keys["normalized_url"][url_key].append(article_id)
        slug_key = normalize_title(slug_from_url(article.get("original_url", "") or article.get("request_url", "")))
        if slug_key:
            keys["slug"][slug_key].append(article_id)
        if article.get("title_hash") and article.get("publication_date"):
            keys["title_date"][f"{article['title_hash']}|{article['publication_date']}"].append(article_id)
        if article.get("content_hash") and _has_dedupable_body(article):
            keys["content_hash"][article["content_hash"]].append(article_id)

    confidence_by_key = {
        "normalized_url": 1.0,
        "slug": 0.86,
        "title_date": 0.95,
        "content_hash": 1.0,
    }
    for reason, groups in keys.items():
        for ids in groups.values():
            if len(ids) < 2:
                continue
            first = ids[0]
            for other in ids[1:]:
                dsu.union(first, other)
                _add_pair_reason(pair_reasons, first, other, reason, confidence_by_key[reason])


def _union_approximate(
    articles: list[dict[str, Any]],
    dsu: DisjointSet,
    pair_reasons: dict[tuple[str, str], list[tuple[str, float]]],
    options: DedupeOptions,
) -> None:
    n = len(articles)
    if n > 3000:
        buckets = _approx_buckets(articles)
        candidate_pairs = set()
        for ids in buckets.values():
            for i, left in enumerate(ids):
                for right in ids[i + 1 :]:
                    candidate_pairs.add(tuple(sorted((left, right))))
        id_to_article = {article["article_id"]: article for article in articles}
        pairs = [(id_to_article[left], id_to_article[right]) for left, right in candidate_pairs]
    else:
        pairs = [
            (articles[i], articles[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]

    for left, right in pairs:
        title_score = _similarity(
            normalize_title(left.get("title", "")),
            normalize_title(right.get("title", "")),
        )
        body_score = 0
        if _has_dedupable_body(left) and _has_dedupable_body(right):
            body_score = _similarity(
                _body_similarity_text(left.get("body_text", "")),
                _body_similarity_text(right.get("body_text", "")),
            )
        if title_score >= options.title_similarity_threshold:
            dsu.union(left["article_id"], right["article_id"])
            _add_pair_reason(
                pair_reasons,
                left["article_id"],
                right["article_id"],
                f"title_similarity_{title_score}",
                title_score / 100,
            )
        elif body_score >= options.body_similarity_threshold:
            dsu.union(left["article_id"], right["article_id"])
            _add_pair_reason(
                pair_reasons,
                left["article_id"],
                right["article_id"],
                f"body_similarity_{body_score}",
                body_score / 100,
            )


def _component_reasons(
    component: list[dict[str, Any]],
    pair_reasons: dict[tuple[str, str], list[tuple[str, float]]],
) -> tuple[list[str], float]:
    article_ids = [article["article_id"] for article in component]
    reasons: list[str] = []
    confidences: list[float] = []
    for i, left in enumerate(article_ids):
        for right in article_ids[i + 1 :]:
            for reason, confidence in pair_reasons.get(tuple(sorted((left, right))), []):
                if reason not in reasons:
                    reasons.append(reason)
                confidences.append(confidence)
    return reasons, max(confidences) if confidences else 0


def _add_pair_reason(
    pair_reasons: dict[tuple[str, str], list[tuple[str, float]]],
    left: str,
    right: str,
    reason: str,
    confidence: float,
) -> None:
    pair_reasons[tuple(sorted((left, right)))].append((reason, confidence))


def _article_url_key(article: dict[str, Any]) -> str:
    for key in ("canonical_url", "final_url_after_redirect", "request_url", "original_url"):
        value = article.get(key)
        if value:
            return normalize_url(str(value))
    return ""


def _source_richness(article: dict[str, Any]) -> int:
    environment = article.get("source_environment", "")
    if environment == "current_wordpress_portal":
        return 3
    if environment == "campus_current_portal":
        return 2
    if environment == "legacy_ip_portal":
        return 1
    return 0


def _similarity(left: str, right: str) -> int:
    if not left or not right:
        return 0
    if fuzz is not None:
        return int(round(fuzz.ratio(left, right)))
    return int(round(SequenceMatcher(None, left, right).ratio() * 100))


def _body_similarity_text(value: str) -> str:
    return normalize_text(value)[:4000]


def _has_dedupable_body(article: dict[str, Any], min_words: int = 50) -> bool:
    if article.get("status") != "collected":
        return False
    if int(article.get("word_count") or 0) < min_words:
        return False
    text = normalize_text(article.get("body_text", ""))
    if len(text) < 200:
        return False
    return True


def _approx_buckets(articles: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for article in articles:
        title = normalize_title(article.get("title", ""))
        year = str(article.get("publication_date", ""))[:4]
        key = f"{year}|{title[:12]}"
        if title:
            buckets[key].append(article["article_id"])
    return buckets


def _clear_duplicate_fields(article: dict[str, Any]) -> None:
    article["duplicate_group_id"] = ""
    article["duplicate_of"] = ""
    article["is_canonical"] = True
    article["duplicate_reason"] = ""
    article["duplicate_confidence"] = 0


def _read_articles(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Article input not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, fields: list[str], records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            csv_record = {}
            for field in fields:
                value = record.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                csv_record[field] = value
            writer.writerow(csv_record)


def _output_paths(output_dir: Path, output_prefix: str = "news_articles") -> dict[str, Path]:
    if output_prefix == "news_articles_antigoportal":
        return {
            "groups_csv": output_dir / "duplicate_groups_antigoportal_latest.csv",
            "groups_json": output_dir / "duplicate_groups_antigoportal_latest.json",
            "deduped_csv": output_dir / "news_articles_antigoportal_deduped_latest.csv",
            "deduped_jsonl": output_dir / "news_articles_antigoportal_deduped_latest.jsonl",
        }
    return {
        "groups_csv": output_dir / "duplicate_groups_latest.csv",
        "groups_json": output_dir / "duplicate_groups_latest.json",
        "deduped_csv": output_dir / "news_articles_deduped_latest.csv",
        "deduped_jsonl": output_dir / "news_articles_deduped_latest.jsonl",
    }
