from __future__ import annotations

import unittest

from ifmt_scraper.normalize_units import match_unit_value, normalize_article


class UnitNormalizationTests(unittest.TestCase):
    def test_common_ascii_variations_match_canonical_units(self) -> None:
        caceres = match_unit_value("Campus Caceres")
        self.assertTrue(caceres.applied)
        self.assertEqual(caceres.unit.code, "cas")
        self.assertEqual(caceres.status, "exact_match")

        bela_vista = match_unit_value("Bela Vista")
        self.assertTrue(bela_vista.applied)
        self.assertEqual(bela_vista.unit.code, "blv")

    def test_noisy_caceres_label_uses_rule_match(self) -> None:
        match = match_unit_value("Campus C\u00e1ceres Inscri\u00e7\u00f5es")
        self.assertTrue(match.applied)
        self.assertEqual(match.unit.code, "cas")
        self.assertEqual(match.status, "rule_match")

    def test_truncated_advanced_campus_uses_context_when_available(self) -> None:
        article = {
            "title": "Campus Avan\u00e7ado Lucas do Rio Verde inicia a semana tecnol\u00f3gica",
            "original_url": "https://200.129.244.212/conteudo/noticia/ifmt-lrv/",
        }
        match = match_unit_value("Campus Avan\u00e7ado de", article)
        self.assertTrue(match.applied)
        self.assertEqual(match.unit.code, "lrv")

    def test_source_campus_defaults_when_labels_are_blank(self) -> None:
        normalized = normalize_article(
            {
                "source_id": "campus_svc",
                "publisher_unit": "",
                "campus_name": "",
            }
        )
        self.assertEqual(normalized["campus_code"], "svc")
        self.assertEqual(normalized["unit_normalization_status"], "exact_match")


if __name__ == "__main__":
    unittest.main()
