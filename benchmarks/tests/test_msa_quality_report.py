# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.msa_quality_report import precision_result, summarize_traces


class TestPrecisionCriteria(unittest.TestCase):
    def test_complete_but_empty_trace_does_not_prove_operator_coverage(self):
        for calls, shapes, observed in [
            ([], [], False),
            ([dict(count=3)], [dict(event="shape", kernel="sort", signature_key="a")], True),
        ]:
            rows = [dict(event="installed"), *shapes, dict(event="summary", calls=calls, restoration_conflicts=[])]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in rows))
                report = summarize_traces([path])[0]
            self.assertTrue(report["trace_complete"])
            self.assertEqual(report["relevant_calls_observed"], observed)
            self.assertEqual(report["relevant_call_count"], 3 if observed else 0)

    def test_inference_keys_exclude_already_warmed_keys(self):
        rows = [{"event": "installed"}]
        for phase, key in [("warmup", "shared"), ("inference", "shared"), ("inference", "new")]:
            rows.append(dict(event="jit_cache_miss", kernel="sort", phase=phase, key=key))
        rows.append(dict(event="summary", restoration_conflicts=[]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            report = summarize_traces([path])[0]
        self.assertTrue(report["trace_complete"])
        self.assertEqual(report["kernels"]["sort"]["inference_keys"], 2)
        self.assertEqual(report["kernels"]["sort"]["new_inference_keys"], 1)

    def test_exact_discrete_outputs_cannot_use_float_tolerances(self):
        self.assertTrue(precision_result([0, 16, -1], [0, 16, -1], "int32")["passed"])
        self.assertFalse(precision_result([0, 17, -1], [0, 16, -1], "int32")["passed"])

    def test_99_percent_rule_and_absolute_ceiling_both_apply(self):
        self.assertTrue(precision_result([0.005] + [0.0] * 99, [0.0] * 100, "float32")["passed"])
        self.assertFalse(precision_result([0.011] + [0.0] * 99, [0.0] * 100, "float32")["passed"])
        self.assertFalse(precision_result([0.005] * 2 + [0.0] * 98, [0.0] * 100, "float32")["passed"])

    def test_empty_or_nonfinite_is_not_a_precision_pass(self):
        self.assertFalse(precision_result([], [], "bfloat16")["passed"])
        self.assertFalse(precision_result([float("nan")], [0.0], "bfloat16")["passed"])
        self.assertFalse(precision_result([float("inf")], [float("inf")], "float16")["passed"])

    def test_report_preserves_dtype_specific_thresholds(self):
        for dtype, rtol, atol, maximum in [
            ("float16", 2**-9, 2**-9, 0.1),
            ("bfloat16", 2**-6, 2**-6, 1.0),
            ("float32", 2**-10, 2**-16, 0.01),
        ]:
            result = precision_result([1.0], [1.0], dtype)
            self.assertEqual(
                (result["rtol"], result["atol"], result["maximum_absolute_error_limit"]), (rtol, atol, maximum)
            )


if __name__ == "__main__":
    unittest.main()
