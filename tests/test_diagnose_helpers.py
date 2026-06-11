from __future__ import annotations

import unittest

from ifmt_scraper.parse_listings import detect_last_listing_page, extract_news_links
from ifmt_scraper.url_utils import normalize_url, resolve_link


class DiagnoseHelperTests(unittest.TestCase):
    def test_relative_legacy_ip_link_stays_on_ip_host(self) -> None:
        url = resolve_link(
            "/conteudo/noticia/exemplo/",
            "https://200.129.244.212/conteudo/noticias/",
        )
        self.assertEqual(
            url,
            "https://200.129.244.212/conteudo/noticia/exemplo/",
        )

    def test_extracts_legacy_article_links(self) -> None:
        html = """
        <html><body>
          <main>
            <a href="/conteudo/noticia/pesquisa-diagnostica/">
              10 Out - Publicado por Reitoria IFMT realiza pesquisa diagnostica
            </a>
            <a href="/conteudo/noticias/?page=2">2</a>
          </main>
        </body></html>
        """
        links = extract_news_links(
            html,
            "https://200.129.244.212/conteudo/noticias/",
        )
        self.assertEqual(len(links), 1)
        self.assertEqual(
            links[0].url,
            "https://200.129.244.212/conteudo/noticia/pesquisa-diagnostica/",
        )
        self.assertEqual(links[0].pattern, "legacy_article")
        self.assertEqual(links[0].date_label_raw, "10 Out")
        self.assertEqual(links[0].publisher_unit, "Reitoria")
        self.assertEqual(
            links[0].title,
            "IFMT realiza pesquisa diagnostica",
        )

    def test_extracts_current_blog_article_links(self) -> None:
        html = """
        <html><body>
          <article class="post">
            <a href="https://ifmt.edu.br/blog/titulo-da-noticia/">
              Titulo da noticia do IFMT
            </a>
          </article>
          <nav><a href="https://ifmt.edu.br/blog/categoria/noticias/page/2/">2</a></nav>
        </body></html>
        """
        links = extract_news_links(html, "https://ifmt.edu.br/blog/categoria/noticias/")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].pattern, "current_wordpress_article")

    def test_detects_legacy_last_listing_page(self) -> None:
        html = """
        <a href="/conteudo/noticias/?page=2">2</a>
        <a href="/conteudo/noticias/?page=305">Última</a>
        """
        self.assertEqual(detect_last_listing_page(html, "legacy_ip"), 305)

    def test_normalize_url_keeps_ip_and_removes_tracking(self) -> None:
        normalized = normalize_url(
            "HTTPS://200.129.244.212/conteudo/noticia/exemplo/?utm_source=x&ok=1#top"
        )
        self.assertEqual(
            normalized,
            "https://200.129.244.212/conteudo/noticia/exemplo/?ok=1",
        )


if __name__ == "__main__":
    unittest.main()
