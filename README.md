# Virtual Reporter

Pipeline experimental para coletar noticias institucionais do IFMT e avaliar
componentes do sistema "Virtual Reporter": classificacao de tipo de noticia,
recuperacao de noticias semelhantes e geracao de briefings estruturados a partir
de pitches incompletos.

O projeto tem dois pacotes principais:

- `ifmt_scraper`: coleta, normalizacao, deduplicacao e consolidacao do dataset.
- `ifmt_models`: processamento, rotulagem fraca, treinamento, recuperacao,
  simulacao do Virtual Reporter e geracao de tabelas/graficos para o artigo.

Os datasets, modelos treinados e resultados gerados ficam em `data/` e nao sao
versionados no Git por tamanho e rastreabilidade.

## Virtual Reporter Experiments

```bash
python -m ifmt_models.cli run-all
python -m ifmt_models.cli evaluate-retrieval-strict
python -m ifmt_models.cli evaluate-briefings-strict
python -m ifmt_models.cli make-paper-assets
python -m ifmt_models.cli report-results
```

Principais saidas locais:

- `data/processed/`
- `data/models/`
- `data/results/`

## IFMT Scraper

Diagnostic-first scraper for building a structured IFMT institutional news
dataset. The current implementation covers source access diagnostics and
listing-page URL discovery, especially for the mandatory historical IP source
`200.129.244.212`.

## Setup

Use Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Diagnostics

```bash
python -m ifmt_scraper.cli diagnose
```

The default diagnostic command tests:

- `https://200.129.244.212/conteudo/noticias/`
- `http://200.129.244.212/conteudo/noticias/`
- `https://ifmt.edu.br/conteudo/noticias/`
- `https://ifmt.edu.br/blog/categoria/noticias/`
- `https://cba.ifmt.edu.br/categoria/noticias/`
- `https://bag.ifmt.edu.br/categoria/noticias/`

The HTTPS IP request disables certificate verification only for host
`200.129.244.212`. Domain requests keep normal SSL verification enabled. Relative
links discovered from the IP source are resolved with the IP as the base URL, not
with `ifmt.edu.br`.

Reports are written to:

- `data/output/diagnostic_report_latest.json`
- `data/output/diagnostic_report_<timestamp>.json`

Raw HTML snapshots are written to:

- `data/raw_html/diagnose/<timestamp>/`

The diagnostic source list is also documented in `sources.json`.

## Discover Listing URLs

Discovery walks only listing/archive pages and writes unique article URLs. It
does not fetch individual article pages.

```bash
python -m ifmt_scraper.cli discover --sources legacy_ip --max-pages 5
python -m ifmt_scraper.cli discover --sources current --max-pages 3
python -m ifmt_scraper.cli discover --sources campuses --max-pages 3
python -m ifmt_scraper.cli discover --sources legacy_ip,current,campuses --max-pages 10
```

Useful discovery options:

```bash
python -m ifmt_scraper.cli discover --sources legacy_ip --max-pages 3 --dry-run
python -m ifmt_scraper.cli discover --sources campuses --max-pages 5 --limit 100
python -m ifmt_scraper.cli discover --sources legacy_ip --max-pages 50 --resume
python -m ifmt_scraper.cli discover --sources current --output-prefix current_discovered
```

Default outputs:

- `data/output/discovered_urls_latest.jsonl`
- `data/output/discovered_urls_latest.csv`
- `data/output/discovery_report_latest.json`

Discovery currently supports source groups `legacy_ip`, `current`, `campuses`,
and `all`. The `legacy_ip` source uses `verify=False` only for
`200.129.244.212`, keeps relative article links on the IP host, and paginates as
`/conteudo/noticias/?page=N`.

## Crawl Article Samples

Crawl reads discovered URLs and fetches individual article pages. Use `--limit`
for validation samples before any larger run.

```bash
python -m ifmt_scraper.cli crawl --limit 20
python -m ifmt_scraper.cli crawl --sources legacy_ip --limit 20 --save-html
python -m ifmt_scraper.cli crawl --sources current --limit 10 --save-html
python -m ifmt_scraper.cli crawl --sources campuses --limit 10 --save-html
python -m ifmt_scraper.cli crawl --sources legacy_ip,current,campuses --limit 50
```

Useful crawl options:

```bash
python -m ifmt_scraper.cli crawl --input data/output/discovered_urls_latest.jsonl
python -m ifmt_scraper.cli crawl --sources legacy_ip --limit 20 --dry-run
python -m ifmt_scraper.cli crawl --sources current --limit 20 --resume
python -m ifmt_scraper.cli crawl --sources campuses --limit 20 --force
python -m ifmt_scraper.cli crawl --sources legacy_ip --output-prefix legacy_news_articles
```

Default crawl outputs:

- `data/output/news_articles_latest.jsonl`
- `data/output/news_articles_latest.csv`
- `data/output/media_assets_latest.csv`
- `data/output/article_links_latest.csv`
- `data/output/crawl_report_latest.json`
- `data/output/manual_review_sample.csv`
- `data/raw_html/articles/` when `--save-html` is used

## Dedupe, Export, And Report

Run duplicate grouping without deleting records:

```bash
python -m ifmt_scraper.cli dedupe
```

This writes:

- `data/output/duplicate_groups_latest.csv`
- `data/output/duplicate_groups_latest.json`
- `data/output/news_articles_deduped_latest.csv`
- `data/output/news_articles_deduped_latest.jsonl`

Generate the final consolidated dataset:

```bash
python -m ifmt_scraper.cli export
```

Generate or refresh the dataset report and Markdown quality report:

```bash
python -m ifmt_scraper.cli report
```

Final outputs are written under `data/output/final/`, including full and
canonical article datasets, side tables, `dataset_report.json`, and
`quality_report.md`. Parquet files are generated when the local Python
environment has Parquet support available.

## Useful Options

```bash
python -m ifmt_scraper.cli diagnose --preview-links 10
python -m ifmt_scraper.cli diagnose --delay 1.0 --max-retries 3
python -m ifmt_scraper.cli diagnose --url https://svc.ifmt.edu.br/categoria/noticias/
python -m ifmt_scraper.cli diagnose --no-save-html
```

## Tests

```bash
python -m unittest discover -s tests
```

## Current Scope

Implemented:

- `diagnose`
- IP-only `verify=False` behavior for `https://200.129.244.212`
- HTTP fallback when the HTTPS IP request fails
- redirect/status/final URL/HTML length/title/fingerprint reporting
- candidate news-link extraction from listing pages
- raw HTML snapshots for audit
- `discover` for listing-page URL discovery
- JSONL/CSV export of discovered URLs
- discovery report with page counts, unique URL counts, stop reasons, and IP
  preservation checks
- `crawl` for bounded article-page extraction
- news article, media asset, article link, crawl report, and manual review
  outputs
- raw article HTML snapshots when `--save-html` is used
- `dedupe` with URL, slug, title/date, content hash, and approximate similarity
  grouping
- final `export` and `report` commands

The remaining large-scale step is operational: run source-specific `discover`
and `crawl --resume` batches, then rerun `dedupe`, `export`, and `report`.
