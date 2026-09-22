# SPDX-License-Identifier: Apache-2.0
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.ci import glm53flash_analyze as analyzer


class AnalyzeTest(unittest.TestCase):
    def setUp(self):
        self.data = {}
        for run in ("graph-0", "graph-1", "graph-2", "eager-0"):
            graph = run.startswith("graph")
            settings = {"enforce_eager": not graph}
            if graph:
                settings["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
            self.data[run] = {
                "result.json": {"status": "PASS", "lengths": list(range(1, 20)), "replays": [7 if graph else 0] * 4},
                "settings.json": settings,
                "runtime-settings.json": {"block_size": 1152},
                "weights.json": [{"rank": i, "sha256": "a" * 64} for i in range(4)],
                "performance.json": [
                    {
                        "iteration": i,
                        "output_tokens_s": 256,
                        "wall_ms": 1000,
                        "requests": [{"preemptions": 0, "ttft_ms": 2, "tpot_ms": 3}] * 4,
                    }
                    for i in range(15)
                ],
            }
        self.logits = np.zeros((8, 154880), dtype=np.float32)

    def evaluate(self):
        with (
            patch.object(analyzer, "read", side_effect=lambda p: copy.deepcopy(self.data[p.parent.name][p.name])),
            patch.object(analyzer.np, "load", return_value=self.logits),
            patch.object(Path, "iterdir", return_value=[]),
        ):
            return analyzer.analyze(Path("runs"))

    def test_complete_unapproved(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "COLLECTED_NOT_APPROVED")
        self.assertEqual(result["performance"]["samples"], 45)
        self.assertEqual(result["comparisons"]["eager-0"]["max_abs"], 0)

    def test_manifest_paths_are_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in self.data:
                (root / run).mkdir()
                (root / run / "sentinel").write_text("fixture")
            with (
                patch.object(analyzer, "read", side_effect=lambda p: copy.deepcopy(self.data[p.parent.name][p.name])),
                patch.object(analyzer.np, "load", return_value=self.logits),
            ):
                result = analyzer.analyze(root)
            self.assertIn("graph-0/sentinel", result["sha256"])
            self.assertFalse(any("\\" in name for name in result["sha256"]))

    def test_missing_samples(self):
        self.data["graph-1"]["performance.json"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete performance"):
            self.evaluate()

    def test_weight_drift(self):
        self.data["graph-2"]["weights.json"] = []
        with self.assertRaisesRegex(ValueError, "Weight/runtime drift"):
            self.evaluate()

    def test_nonfinite(self):
        self.logits[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.evaluate()

    def test_preemption(self):
        self.data["graph-0"]["performance.json"][0]["requests"][0]["preemptions"] = 1
        with self.assertRaisesRegex(ValueError, "Preempted"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
