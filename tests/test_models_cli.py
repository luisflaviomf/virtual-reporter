from __future__ import annotations

import unittest

from ifmt_models.cli import build_parser


class ModelsCliTests(unittest.TestCase):
    def test_expanded_evaluation_arguments(self) -> None:
        args = build_parser().parse_args(
            [
                "run-expanded-evaluation",
                "--case-count",
                "300",
                "--seed",
                "42",
                "--output-dir",
                "data/results/evaluation_300_seed42",
                "--pilot-dir",
                "data/results/evaluation",
                "--resume",
            ]
        )

        self.assertEqual(args.case_count, 300)
        self.assertEqual(args.seed, 42)
        self.assertTrue(args.resume)
        self.assertEqual(
            args.output_dir, "data/results/evaluation_300_seed42"
        )
        self.assertEqual(args.pilot_dir, "data/results/evaluation")


if __name__ == "__main__":
    unittest.main()
