from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

from src.pipeline import ROOT, CONFIG_PATH, assign_clusters, load_config, run_kmeans


class AlgorithmTests(unittest.TestCase):
    def test_kmeans_is_reproducible(self) -> None:
        rng = np.random.default_rng(7)
        x = np.vstack([rng.normal(-2, 0.1, (20, 3)), rng.normal(2, 0.1, (20, 3))])
        first = run_kmeans(x, 2, 42, 3, 50)
        second = run_kmeans(x, 2, 42, 3, 50)
        self.assertTrue(np.array_equal(first.labels, second.labels))
        self.assertTrue(np.allclose(first.centers, second.centers))
        labels, _ = assign_clusters(x, first.centers)
        self.assertTrue(np.array_equal(labels, first.labels))


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.source = ROOT / cls.config["paths"]["restaurants"]
        cls.output = ROOT / cls.config["paths"]["outputs"] / "restaurants_labeled.csv"

    def test_output_exists(self) -> None:
        self.assertTrue(self.output.exists(), "请先运行 python -m src.pipeline")

    def test_output_only_adds_label_and_preserves_rows(self) -> None:
        with self.source.open("r", encoding="utf-8-sig", newline="") as source_handle:
            source_reader = csv.reader(source_handle)
            source_header = next(source_reader)
            source_rows = list(source_reader)
        with self.output.open("r", encoding="utf-8-sig", newline="") as output_handle:
            output_reader = csv.reader(output_handle)
            output_header = next(output_reader)
            output_rows = list(output_reader)
        self.assertEqual(output_header, source_header + ["类别标签"])
        self.assertEqual(len(output_rows), len(source_rows))
        for source_row, output_row in zip(source_rows, output_rows):
            self.assertEqual(output_row[:-1], source_row)
            self.assertTrue(output_row[-1].strip())

    def test_all_output_labels_are_documented(self) -> None:
        with self.output.open("r", encoding="utf-8-sig", newline="") as handle:
            labels = {row["类别标签"] for row in csv.DictReader(handle)}
        explanation = (ROOT / "docs" / "口碑分类说明.md").read_text(encoding="utf-8")
        for label in labels:
            self.assertIn(f"### {label}", explanation)

    def test_excel_delivery_exists(self) -> None:
        workbook = ROOT / self.config["paths"]["outputs"] / "口碑聚类结果.xlsx"
        self.assertTrue(workbook.exists())
        self.assertGreater(workbook.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
