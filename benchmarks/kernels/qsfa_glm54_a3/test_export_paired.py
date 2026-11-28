# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4

import export_paired as paired


class PairedExportTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parent / f".export-test-{uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)
        self.graph_path = self.directory / "graph.csv"
        self.selected = "project:/home/local-operator@internal"
        self.rows = [
            {
                **dict.fromkeys(paired.GRAPH_NONNEGATIVE, 0.02),
                "batch": batch,
                "tokens": 6 * batch,
                "valid_sparse_pairs": 24 * batch,
                "useful_flops": 1000 * batch,
                "logical_io_bytes": 100 * batch,
                "native_us": 20.0,
                "candidate_us": 20.1,
                "paired_speedup": 1.0,
                "speedup_ci95_low": 0.99,
                "speedup_ci95_high": 1.01,
                "selected": self.selected,
                "metadata": "=ENV(opaque)",
            }
            for batch in paired.BATCHES
        ]
        self.rows[0].update(paired_speedup=1.2, speedup_ci95_low=1.1, speedup_ci95_high=1.3)
        self.rows[7].update(paired_speedup=0.8, speedup_ci95_low=0.7, speedup_ci95_high=0.9)
        self.rows[15].update(paired_speedup=1.05, speedup_ci95_low=1.0, speedup_ci95_high=1.1)

    def write_table(self, name, fields, rows):
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_graph(self, rows):
        self.write_table("graph.csv", (*paired.GRAPH_FIELDS, "selected", "metadata"), rows)

    def hbm_row(self, implementation):
        return {
            **dict.fromkeys(paired.HBM_FLOAT, 42.5),
            "batch": 1,
            "tokens": 6,
            "implementation": implementation,
            "min_counter_samples": 3,
        }

    def test_cli_only_exports_numbers_and_reports_partial_hbm(self):
        self.write_graph(reversed(self.rows))
        hbm_path = self.write_table(
            "hbm.csv",
            ("batch", "tokens", "implementation", "samples", "min_counter_samples", *paired.HBM_FLOAT),
            [
                {**self.hbm_row("native"), "samples": 2},
                {**self.hbm_row("candidate"), "samples": 2},
            ],
        )
        output = self.directory / "export"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(paired.__file__)),
                "--graph",
                str(self.graph_path),
                "--output",
                str(output),
                "--hbm",
                str(hbm_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        with (output / "paired.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["batch"]) for row in rows], list(paired.BATCHES))
        for row in rows:
            for value in filter(None, row.values()):
                self.assertIsInstance(float(value), float)
        report = json.loads(result.stdout)
        self.assertEqual(report, json.loads((output / "summary.json").read_text(encoding="utf-8")))
        self.assertEqual(report["winner_counts"], {"candidate": 2, "native": 1, "tie": 77})
        self.assertEqual(report["ci95_counts"], {"above_one": 1, "below_one": 1, "includes_one": 78})
        self.assertEqual(report["hbm_coverage"]["paired_batches"], 1)
        self.assertEqual([row["batch"] for row in report["key_batches"]], list(paired.KEY_BATCHES))
        self.assertAlmostEqual(report["key_batches"][0]["native_hbm_bytes_over_logical"], 0.425)
        for path in output.iterdir():
            exported = path.read_text(encoding="utf-8")
            self.assertNotIn(self.selected, exported)
            self.assertNotIn("ENV(opaque)", exported)

    def test_graph_rejects_duplicate_missing_or_unpaired_rows(self):
        invalid_rows = (
            (self.rows[:-1], "missing B80"),
            ([*self.rows, self.rows[0]], "B1 occurs more than once"),
            ([{**self.rows[0], "candidate_us": ""}, *self.rows[1:]], "invalid numeric.*candidate_us"),
            ([{**self.rows[0], "speedup_ci95_low": 1.3}, *self.rows[1:]], "invalid paired speedup"),
        )
        for rows, error in invalid_rows:
            with self.subTest(error=error):
                self.write_graph(rows)
                with self.assertRaisesRegex(ValueError, error):
                    paired.graph_data(self.graph_path)

    def test_joined_summary_rejects_stale_timing(self):
        self.write_graph(self.rows)
        graph, selected = paired.graph_data(self.graph_path)
        row = {
            **self.hbm_row("candidate"),
            "hbm_samples": 2,
            "operator": self.selected,
            **{field: self.rows[0][field] for field in (*paired.WORKLOAD, *paired.SPEEDUP, "candidate_native_max_abs")},
            **{field: self.rows[0][f"candidate_{suffix}"] for field, suffix in paired.SUMMARY_GRAPH_FIELDS.items()},
        }
        path = self.write_table("summary.csv", list(row), [row])
        report = paired.hbm_data(path, graph, selected, joined=True, hbm_candidate=None)
        self.assertEqual(report[(1, "candidate")]["physical_mbu_pct"], 42.5)
        row["latency_us"] *= 2
        self.write_table("summary.csv", list(row), [row])
        with self.assertRaisesRegex(ValueError, "disagrees with the graph on latency_us"):
            paired.hbm_data(path, graph, selected, joined=True, hbm_candidate=None)

    def test_raw_profile_settings_verify_workload_and_schedule(self):
        self.write_graph(self.rows)
        graph, selected = paired.graph_data(self.graph_path)
        hbm_path = self.write_table(
            "hbm.csv",
            ("batch", "tokens", "implementation", "samples", "min_counter_samples", *paired.HBM_FLOAT),
            [{**self.hbm_row("candidate"), "samples": 2}],
        )
        settings = dict.fromkeys(paired.PROFILE_SHAPE, 1)
        profile_path = self.directory / "profile_settings.json"
        (self.directory / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        profile = {**settings, "candidate": self.selected, "mode": "hbm"}
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        candidate = paired.verified_hbm_candidate(self.graph_path, hbm_path)
        self.assertEqual(candidate, self.selected)
        with self.assertRaisesRegex(ValueError, "different candidate schedule"):
            paired.hbm_data(hbm_path, graph, selected, joined=False, hbm_candidate="other")
        profile["heads"] = 2
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "settings differ on heads"):
            paired.verified_hbm_candidate(self.graph_path, hbm_path)

    def test_hbm_settings_match_explicit_and_legacy_baselines(self):
        graph_settings = self.directory / "settings.json"
        profile_settings = self.directory / "profile_settings.json"
        hbm_path = self.directory / "hbm.csv"
        shape = dict.fromkeys(paired.PROFILE_SHAPE, 1)
        stock = {"baseline_candidate": None}
        tilelang = {"baseline_candidate": "qsfa_native_reference:qsfa_glm54_mtp5"}
        for graph, hbm, matches in (
            ({}, {}, True),
            ({}, stock, True),
            (stock, {}, True),
            (stock, stock, True),
            (tilelang, tilelang, True),
            (stock, tilelang, False),
            (tilelang, {}, False),
        ):
            with self.subTest(graph=graph, hbm=hbm):
                graph_settings.write_text(json.dumps({**shape, **graph}), encoding="utf-8")
                profile = {**shape, **hbm, "candidate": self.selected, "mode": "hbm"}
                profile_settings.write_text(json.dumps(profile), encoding="utf-8")
                if matches:
                    self.assertEqual(paired.verified_hbm_candidate(self.graph_path, hbm_path), self.selected)
                else:
                    with self.assertRaisesRegex(ValueError, "settings differ on baseline_candidate"):
                        paired.verified_hbm_candidate(self.graph_path, hbm_path)

    def test_full_hbm_mode_requires_two_rows_for_every_batch(self):
        with self.assertRaisesRegex(ValueError, "0/80"):
            paired.export({}, {}, hbm_verified=False, require_full_hbm=True, destination=self.directory / "rejected")
        self.assertFalse((self.directory / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
