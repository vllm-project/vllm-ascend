# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-NPU tests; this file can also run directly with Python's unittest runner."""

import copy
import importlib.util
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

_PATH = Path(__file__).resolve().parents[1] / "e2e/pull_request/four_card/glm53flash_quality.py"
_SPEC = importlib.util.spec_from_file_location("glm53flash_quality", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
quality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quality)


def _request():
    return SimpleNamespace(
        finished=True,
        prompt_token_ids=[10, 11],
        outputs=[
            SimpleNamespace(
                token_ids=[42, 43],
                logprobs=[
                    {42: SimpleNamespace(logprob=-1.0), 43: SimpleNamespace(logprob=-2.0)},
                    {42: SimpleNamespace(logprob=-2.5), 43: SimpleNamespace(logprob=-1.1)},
                ],
            )
        ],
    )


class TestSnapshots(unittest.TestCase):
    def test_json_round_trip(self):
        snapshot = quality.summarize_outputs([_request()])
        quality.compare_snapshots(snapshot, json.loads(json.dumps(snapshot)), atol=0)

    def test_tolerance(self):
        request = _request()
        reference = quality.summarize_outputs([request])
        candidate = quality.summarize_outputs([request])
        candidate[0]["logprobs"][0]["42"] += 0.01
        quality.compare_snapshots(candidate, reference, atol=0.02)
        with self.assertRaisesRegex(AssertionError, "logprob difference"):
            quality.compare_snapshots(candidate, reference, atol=0.001)

    def test_output_must_finish(self):
        request = _request()
        request.finished = False
        with self.assertRaisesRegex(ValueError, "finish"):
            quality.summarize_outputs([request])

    def test_one_completion_required(self):
        request = _request()
        request.outputs.append(request.outputs[0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            quality.summarize_outputs([request])

    def test_missing_logprobs(self):
        request = _request()
        request.outputs[0].logprobs = None
        with self.assertRaisesRegex(ValueError, "not returned"):
            quality.summarize_outputs([request])

    def test_missing_step_or_generated_token(self):
        for mode in ("step", "generated"):
            with self.subTest(mode=mode):
                request = _request()
                if mode == "step":
                    request.outputs[0].logprobs.pop()
                else:
                    request.outputs[0].logprobs[0][44] = request.outputs[0].logprobs[0].pop(42)
                with self.assertRaises(ValueError):
                    quality.summarize_outputs([request])

    def test_invalid_logprobs_fail(self):
        for value in (math.nan, math.inf, -math.inf, 0.1, True):
            with self.subTest(value=value):
                request = _request()
                request.outputs[0].logprobs[0][44] = SimpleNamespace(logprob=value)
                with self.assertRaises(ValueError):
                    quality.summarize_outputs([request])

    def test_degenerate_distribution(self):
        request = _request()
        request.outputs[0].logprobs[0][43].logprob = -1.0
        with self.assertRaisesRegex(ValueError, "degenerate"):
            quality.summarize_outputs([request])

    def test_invalid_probability_mass(self):
        request = _request()
        request.outputs[0].logprobs[0][42].logprob = -0.01
        request.outputs[0].logprobs[0][43].logprob = -0.02
        with self.assertRaisesRegex(ValueError, "sum"):
            quality.summarize_outputs([request])

    def test_empty_or_missing_baseline(self):
        candidate = quality.summarize_outputs([_request()])
        for reference in (None, [], {}, [{"token_ids": [42]}]):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                quality.compare_snapshots(candidate, reference, atol=0.01)

    def test_invalid_reference_fails_closed(self):
        reference = quality.summarize_outputs([_request()])
        reference[0]["logprobs"][0]["42"] = math.nan
        with self.assertRaises(ValueError):
            quality.compare_snapshots(quality.summarize_outputs([_request()]), reference, atol=0.01)

    def test_changed_prompt_or_token_path_fails(self):
        reference = quality.summarize_outputs([_request()])
        for field, index, value in (("prompt_token_ids", 0, 99), ("token_ids", 0, 43)):
            candidate = copy.deepcopy(reference)
            candidate[0][field][index] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                quality.compare_snapshots(candidate, reference, atol=1.0)

    def test_near_tie_does_not_excuse_context_divergence(self):
        reference = quality.summarize_outputs([_request()])
        reference[0]["logprobs"][0]["43"] = -1.000001
        candidate = copy.deepcopy(reference)
        candidate[0]["token_ids"][0] = 43
        with self.assertRaisesRegex(AssertionError, "near ties"):
            quality.compare_snapshots(candidate, reference, atol=0.01)

    def test_missing_frozen_candidate_fails(self):
        reference = quality.summarize_outputs([_request()])
        candidate = copy.deepcopy(reference)
        candidate[0]["logprobs"][0]["44"] = candidate[0]["logprobs"][0].pop("43")
        with self.assertRaisesRegex(AssertionError, "candidate token set"):
            quality.compare_snapshots(candidate, reference, atol=0.01)

    def test_truncated_reference_cannot_reduce_coverage(self):
        request = _request()
        for step in request.outputs[0].logprobs:
            step[44] = SimpleNamespace(logprob=-4.0)
        candidate = quality.summarize_outputs([request])
        reference = copy.deepcopy(candidate)
        del reference[0]["logprobs"][0]["44"]
        with self.assertRaisesRegex(AssertionError, "candidate token set"):
            quality.compare_snapshots(candidate, reference, atol=0.01)

    def test_request_count_mismatch_fails(self):
        reference = quality.summarize_outputs([_request()])
        with self.assertRaisesRegex(AssertionError, "Request count"):
            quality.compare_snapshots(reference * 2, reference, atol=0)

    def test_invalid_options(self):
        reference = quality.summarize_outputs([_request()])
        for atol in (-1, math.nan, math.inf, True):
            with self.subTest(atol=atol), self.assertRaises(ValueError):
                quality.compare_snapshots(reference, reference, atol=atol)


class TestPerformance(unittest.TestCase):
    def test_median_and_bound(self):
        summary = quality.summarize_performance([2, 4, 8], output_tokens_per_run=400)
        self.assertEqual(summary["median_elapsed_s"], 4)
        self.assertEqual(summary["median_output_tokens_s"], 100)
        quality.assert_performance(summary, minimum_output_tokens_s=100)
        with self.assertRaisesRegex(AssertionError, "throughput regression"):
            quality.assert_performance(summary, minimum_output_tokens_s=101)

    def test_invalid_timings(self):
        for value in (0, -1, math.nan, math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quality.summarize_performance([1, 2, value], output_tokens_per_run=100)
        with self.assertRaises(ValueError):
            quality.summarize_performance([1, 2], output_tokens_per_run=100)

    def test_invalid_token_count(self):
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quality.summarize_performance([1, 2, 3], output_tokens_per_run=value)

    def test_overflowed_throughput_fails(self):
        with self.assertRaisesRegex(ValueError, "throughput"):
            quality.summarize_performance([1e-308] * 3, output_tokens_per_run=100)

    def test_invalid_threshold(self):
        summary = quality.summarize_performance([1, 2, 3], output_tokens_per_run=100)
        for value in (0, -1, math.nan, math.inf, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quality.assert_performance(summary, minimum_output_tokens_s=value)

    def test_corrupt_summary_cannot_bypass_gate(self):
        for value in (1000, math.nan):
            summary = quality.summarize_performance([1, 2, 3], output_tokens_per_run=100)
            summary["median_output_tokens_s"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                quality.assert_performance(summary, minimum_output_tokens_s=100)


class TestCommittedBaseline(unittest.TestCase):
    def test_reference_shape_and_measured_thresholds(self):
        baseline = json.loads((_PATH.parent / "glm53flash_assets/quality_a5_baseline.json").read_text(encoding="utf-8"))
        quality.compare_snapshots(baseline["numerical"], baseline["numerical"], atol=baseline["logprob_atol"])
        protocol = baseline["protocol"]
        expected_keys = {str(token) for token in protocol["numerical_probe_token_ids"]}
        self.assertEqual(len(expected_keys), protocol["numerical_logprobs"])
        self.assertEqual(protocol["platform"], "A5")
        self.assertEqual(protocol["numerical_requests_per_scenario"], 4)
        self.assertEqual(len(baseline["numerical"]), len(protocol["numerical_prompts"]) * 4)
        self.assertEqual(protocol["numerical_result_order"], "scenario_then_dp_rank")
        self.assertEqual(protocol["performance_requests_per_dp_rank"], 1)
        for request in baseline["numerical"]:
            self.assertEqual(len(request["token_ids"]), protocol["numerical_output_tokens"])
            for step in request["logprobs"]:
                self.assertEqual(set(step), expected_keys)
        provenance = baseline["provenance"]
        fraction = provenance["minimum_reference_fraction"]
        self.assertGreater(fraction, 0)
        self.assertLessEqual(fraction, 1)
        for mode in ("eager", "graph"):
            samples = [item for item in provenance["performance_samples"] if item["mode"] == mode]
            self.assertGreaterEqual(len(samples), 2)
            threshold = baseline["minimum_output_tokens_s"][mode]
            for sample in samples:
                quality.assert_performance(sample, minimum_output_tokens_s=threshold)
                self.assertEqual(len(sample["elapsed_seconds"]), protocol["measured_runs"])
            expected = math.floor(min(sample["median_output_tokens_s"] for sample in samples) * fraction * 100) / 100
            self.assertEqual(threshold, expected)


if __name__ == "__main__":
    unittest.main()
