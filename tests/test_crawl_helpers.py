from __future__ import annotations

import unittest

from ifmt_scraper.normalize import parse_publication_datetime
from ifmt_scraper.parse_articles import parse_article_html


class CrawlHelperTests(unittest.TestCase):
    def test_parse_portuguese_publication_datetime(self) -> None:
        date_value, time_value, raw = parse_publication_datetime(
            "22 de Outubro de 2025 às 13:28"
        )
        self.assertEqual(date_value, "2025-10-22")
        self.assertEqual(time_value, "13:28")
        self.assertEqual(raw, "22 de Outubro de 2025 às 13:28")

    def test_parse_article_html_extracts_media_and_links(self) -> None:
        html = """
        <html>
          <head>
            <meta property="article:published_time" content="2026-05-31T13:12:32+00:00">
            <meta name="author" content="Thiago Almeida">
            <meta property="og:image" content="/wp-content/uploads/pesar.png">
            <link rel="canonical" href="https://ifmt.edu.br/blog/teste/">
          </head>
          <body>
            <main>
              <h1>Titulo da noticia</h1>
              <div class="elementor-widget-theme-post-content">
                <p>Primeiro paragrafo relevante da noticia institucional.</p>
                <p>Segundo paragrafo com <a href="/arquivo.pdf">anexo</a>.</p>
                <figure><img src="/imagem.jpg" alt="Foto"><figcaption>Legenda</figcaption></figure>
              </div>
            </main>
          </body>
        </html>
        """
        discovered = {
            "discovered_id": "disc_1",
            "source_id": "current_portal",
            "source_name": "IFMT",
            "source_environment": "current_wordpress_portal",
            "source_base_url": "https://ifmt.edu.br",
            "source_host": "ifmt.edu.br",
            "source_access_mode": "domain",
            "listing_url": "https://ifmt.edu.br/blog/categoria/noticias/",
            "listing_page_number": 1,
            "discovered_url": "https://ifmt.edu.br/blog/teste/",
            "title_from_listing": "Titulo da noticia",
        }
        parsed = parse_article_html(
            html=html,
            discovered_record=discovered,
            request_url="https://ifmt.edu.br/blog/teste/",
            final_url="https://ifmt.edu.br/blog/teste/",
            http_status=200,
            redirected=False,
            used_ip_access=False,
            ssl_verify_disabled=False,
        )
        self.assertEqual(parsed.article["title"], "Titulo da noticia")
        self.assertEqual(parsed.article["publication_date"], "2026-05-31")
        self.assertEqual(parsed.article["publication_time"], "13:12")
        self.assertEqual(parsed.article["author_name"], "Thiago Almeida")
        self.assertTrue(parsed.article["has_images"])
        self.assertTrue(parsed.article["has_attachments"])
        self.assertEqual(len(parsed.media_assets), 2)
        self.assertEqual(parsed.article_links[0]["link_type"], "document")


if __name__ == "__main__":
    unittest.main()
