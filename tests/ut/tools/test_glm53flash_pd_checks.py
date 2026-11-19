# SPDX-License-Identifier: Apache-2.0
"""Fault injection against the exact assertions used by the live PD test."""

import unittest

from tests.e2e.common.glm53flash_pd_checks import check_completion, check_state_pulls


class TestFlashPDChecks(unittest.TestCase):
    def completion(self):
        return {
            "usage": {"completion_tokens": 2},
            "choices": [
                {
                    "finish_reason": "length",
                    "logprobs": {"tokens": ["token_id:10", "token_id:11"], "token_logprobs": [-1.0, -2.0]},
                }
            ],
        }

    def copies(self):
        return [
            {
                "copies": [
                    {"state": 42, "block": 3, "offset": offset, "pulls": 2, "bytes": 128, "expected_bytes": 128}
                    for offset in (0, 1)
                ]
            }
        ]

    def test_complete_transfer(self):
        check_state_pulls(self.copies())
        check_completion(self.completion(), 2)

    def test_missing_and_duplicate_shard(self):
        for copies in (self.copies()[0]["copies"][:1], [self.copies()[0]["copies"][0]] * 2):
            with self.assertRaisesRegex(AssertionError, "Missing PCP"):
                check_state_pulls([{"copies": copies}])

    def test_omitted_and_truncated_state(self):
        for size in (0, 127):
            copies = self.copies()
            copies[0]["copies"][0]["bytes"] = size
            with self.assertRaisesRegex(AssertionError, "Missing or truncated"):
                check_state_pulls(copies)

    def test_recomputation_without_transfer(self):
        with self.assertRaisesRegex(AssertionError, "No remote"):
            check_state_pulls([{"copies": []}])

    def test_old_replica_plan(self):
        copies = self.copies()
        copies[0]["copies"][0]["pulls"] = 1
        with self.assertRaisesRegex(AssertionError, "every PCP shard"):
            check_state_pulls(copies)

    def test_nonfinite_and_short_output(self):
        for value in (float("nan"), float("inf")):
            bad = self.completion()
            bad["choices"][0]["logprobs"]["token_logprobs"][0] = value
            with self.assertRaisesRegex(AssertionError, "Non-finite"):
                check_completion(bad, 2)
        with self.assertRaises(AssertionError):
            check_completion(self.completion(), 3)
        for key in ("tokens", "token_logprobs"):
            bad = self.completion()
            bad["choices"][0]["logprobs"][key].pop()
            with self.assertRaisesRegex(AssertionError, "count"):
                check_completion(bad, 2)


if __name__ == "__main__":
    unittest.main()
