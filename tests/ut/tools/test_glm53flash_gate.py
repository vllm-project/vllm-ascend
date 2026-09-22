# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.ci import run_glm53flash_gates as gate
from tools.ci.glm53flash_make_policy import candidate_policy


class GateTest(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "approved": False,
            "baseline_correctness_validated": True,
            "scope": "unit-test",
            "precision_atol": 0,
            "eager_graph_atol": 0.02,
            "minimum_tokens_s": 90,
            "maximum_ttft_p95_ms": 10,
            "maximum_tpot_p95_ms": 10,
            "baseline_sha256": {"x": "hash"},
        }
        self.summary = {
            "sha256": {"x": "hash"},
            "comparisons": {"eager-0": {"max_abs": 0.01, "top1_equal": True}},
            "performance": {"median_tokens_s": 100, "ttft_p95_ms": 2, "tpot_p95_ms": 3},
        }
        self.current = copy.deepcopy(self.summary)
        self.logit_result = {"status": "PASS"}

    def evaluate(self, enforce=False):
        with (
            patch.object(gate, "analyze", side_effect=[self.summary, self.current]),
            patch.object(gate, "read", return_value={"lengths": [128]}),
            patch.object(gate.np, "load", return_value=np.zeros((1,))),
            patch.object(gate, "check_logits", return_value=self.logit_result),
        ):
            return gate.evaluate(
                Path("baseline"),
                Path("candidate"),
                self.policy,
                enforce,
                {"scope": "unit-test", "validated": True, "exclusive": True, "checked_by": "unit-test"},
            )

    def test_missing_environment(self):
        with self.assertRaisesRegex(ValueError, "Trusted runner"):
            gate.evaluate("baseline", "candidate", self.policy)

    def test_pass(self):
        self.assertEqual(self.evaluate()["status"], "PASS")

    def test_unapproved_enforcement(self):
        with self.assertRaisesRegex(ValueError, "approval"):
            self.evaluate(True)

    def test_baseline_tamper(self):
        self.policy["baseline_sha256"] = {}
        with self.assertRaisesRegex(ValueError, "Baseline files changed"):
            self.evaluate()

    def test_performance_warn_and_enforce(self):
        self.current["performance"]["median_tokens_s"] = 80
        self.assertEqual(self.evaluate()["status"], "WARN")
        self.policy.update(approved=True, approved_by="unit-test-only")
        result = self.evaluate(True)
        self.assertEqual((result["status"], result["exit_code"]), ("FAIL", 1))

    def test_precision_failure(self):
        self.logit_result = {"status": "FAIL"}
        self.assertFalse(self.evaluate()["precision"]["passed"])

    def test_latency_regression(self):
        self.current["performance"]["ttft_p95_ms"] = 20
        self.assertFalse(self.evaluate()["performance"]["passed"])

    def test_eager_mismatch(self):
        self.current["comparisons"]["eager-0"]["top1_equal"] = False
        self.assertFalse(self.evaluate()["precision"]["passed"])

    def test_invalid_threshold(self):
        self.policy["precision_atol"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid threshold"):
            self.evaluate()

    def test_policy_is_unapproved_and_uses_measured_margins(self):
        summary = {
            "comparisons": {
                run: {"max_abs": 0, "top1_equal": True} for run in ("graph-0", "graph-1", "graph-2", "eager-0")
            },
            "performance": {"run_medians_tokens_s": [100, 101, 99], "ttft_p95_ms": 10, "tpot_p95_ms": 2},
            "sha256": {},
        }
        policy = candidate_policy(summary, "unit-test-scope")
        self.assertFalse(policy["approved"])
        self.assertFalse(policy["baseline_correctness_validated"])
        self.assertEqual(policy["scope"], "unit-test-scope")
        self.assertTrue(policy["blocking_findings"])
        self.assertAlmostEqual(policy["minimum_tokens_s"], 89.1)
        summary["comparisons"]["eager-0"]["top1_equal"] = False
        with self.assertRaisesRegex(ValueError, "Top-1 discrepancy"):
            candidate_policy(summary)

    def test_unvalidated_correctness_blocks_enforcement(self):
        self.policy.update(approved=True, approved_by="unit-test", baseline_correctness_validated=False)
        with self.assertRaisesRegex(ValueError, "correctness"):
            self.evaluate(True)


if __name__ == "__main__":
    unittest.main()
