from __future__ import annotations

import unittest

from ifmt_models.evaluation_cases import (
    MACRO_GROUPS,
    make_masked_case,
    sampling_summary,
    select_stratified_articles,
    serialize_pitch,
)


class EvaluationCaseSelectionTests(unittest.TestCase):
    def test_selects_equal_unique_groups_and_excludes_pilot(self) -> None:
        labels = [
            "edital_selection",
            "event",
            "teaching",
            "research",
            "extension",
            "award_result",
        ]
        rows = [
            {"article_id": f"{label}-{index:03d}", "label": label}
            for label in labels
            for index in range(8)
        ]
        pilot = {"event-000", "award_result-000"}

        selected = select_stratified_articles(
            rows,
            pilot_article_ids=pilot,
            per_group=5,
            seed=42,
            labeler=lambda row: str(row["label"]),
        )

        self.assertEqual(len(selected), 30)
        self.assertEqual(len({item.article_id for item in selected}), 30)
        self.assertTrue(pilot.isdisjoint({item.article_id for item in selected}))
        for group in MACRO_GROUPS:
            self.assertEqual(sum(item.macro_group == group for item in selected), 5)
        repeated = select_stratified_articles(
            rows,
            pilot_article_ids=pilot,
            per_group=5,
            seed=42,
            labeler=lambda row: str(row["label"]),
        )
        self.assertEqual(selected, repeated)

    def test_rejects_a_group_without_enough_articles(self) -> None:
        rows = [
            {"article_id": f"{label}-{index}", "label": label}
            for label in (
                "edital_selection",
                "teaching",
                "research",
                "extension",
                "award_result",
            )
            for index in range(2)
        ]
        rows.append({"article_id": "only-one-event", "label": "event"})

        with self.assertRaisesRegex(ValueError, "event requires 2 articles, found 1"):
            select_stratified_articles(
                rows,
                pilot_article_ids=set(),
                per_group=2,
                seed=42,
                labeler=lambda row: str(row["label"]),
            )


class EvaluationMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.article = {
            "article_id": "article-1",
            "title": "Unique What Value",
            "author_name": "Unique Who Value",
            "publication_date": "2026-01-02",
            "campus_name": "Unique Unit Value",
            "lead": "Unique Why Value",
            "body_text": (
                "Unique Why Value. Inscricoes devem ser feitas usando Unique How Value."
            ),
            "original_url": "https://example.invalid/unique-source",
        }

    def test_mask_is_deterministic_retains_what_and_hides_removed_values(self) -> None:
        first = make_masked_case(self.article, 1, "event", "event", 42, 4)
        second = make_masked_case(self.article, 1, "event", "event", 42, 4)

        self.assertEqual(first, second)
        self.assertIn("Unique What Value", first["incomplete_pitch"])
        self.assertNotIn("removed_fields", first["incomplete_pitch"])
        self.assertNotIn("Dados ainda nao confirmados:", first["incomplete_pitch"])
        for field in first["removed_fields"]:
            value = str(first["ground_truth"].get(field, ""))
            if value:
                self.assertNotIn(value, first["incomplete_pitch"])

    def test_mask_never_removes_what_and_reaches_requested_depth(self) -> None:
        case = make_masked_case(self.article, 2, "event", "event", 42, 3)

        self.assertNotIn("what", case["removed_fields"])
        self.assertGreaterEqual(len(case["removed_fields"]), 3)
        self.assertEqual(case["mask_depth"], len(case["removed_fields"]))
        self.assertEqual(case["requested_mask_depth"], 3)

    def test_masks_alias_fields_and_redacts_removed_values_inside_title(self) -> None:
        article = dict(self.article)
        article["title"] = "Evento em Unique Unit Value no dia 2026-01-02"
        article["author_name"] = "Unique Unit Value"

        cases = [
            make_masked_case(article, number, "event", "event", 42, 5)
            for number in range(1, 9)
        ]

        self.assertTrue(any("when" in case["removed_fields"] for case in cases))
        self.assertTrue(any("unit" in case["removed_fields"] for case in cases))
        for case in cases:
            removed_values = {
                str(case["ground_truth"][field])
                for field in case["removed_fields"]
                if case["ground_truth"].get(field)
            }
            for value in removed_values:
                self.assertNotIn(value, case["incomplete_pitch"])
            aliases = {"who", "where", "unit"}
            if aliases & set(case["removed_fields"]):
                self.assertTrue(aliases <= set(case["removed_fields"]))

    def test_serialized_pitch_contains_only_supplied_confirmed_fields(self) -> None:
        pitch = serialize_pitch({"what": "Projeto X", "when": "2026-01-02"})

        self.assertIn("Fato informado: Projeto X.", pitch)
        self.assertIn("Data confirmada: 2026-01-02.", pitch)
        self.assertNotIn("Responsavel confirmado:", pitch)
        self.assertTrue(pitch.endswith("Ha informacoes ainda pendentes de confirmacao."))

    def test_sampling_summary_counts_groups_depths_and_removed_fields(self) -> None:
        cases = [
            {
                "macro_group": "event",
                "mask_depth": 2,
                "removed_fields": ["who", "when"],
            },
            {
                "macro_group": "event",
                "mask_depth": 3,
                "removed_fields": ["where", "source", "unit"],
            },
        ]

        summary = sampling_summary(cases)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["macro_groups"], {"event": 2})
        self.assertEqual(summary["mask_depths"], {"2": 1, "3": 1})
        self.assertEqual(summary["removed_fields"]["who"], 1)
        self.assertEqual(summary["removed_fields"]["unit"], 1)


if __name__ == "__main__":
    unittest.main()
