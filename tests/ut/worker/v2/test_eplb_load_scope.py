# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest

from vllm_ascend.worker.v2.eplb import is_eplb_load_scope_matched


class TestEplbLoadScope(unittest.TestCase):
    def test_batch_scope_semantics(self):
        cases = [
            ("all", [True, False], True),
            ("all", [False, False], True),
            ("prefill", [True, False], True),
            ("decode", [True, False], False),
            ("prefill", [False, False], False),
            ("decode", [False, False], True),
        ]
        for load_scope, is_prefilling, expected in cases:
            with self.subTest(load_scope=load_scope, is_prefilling=is_prefilling):
                self.assertIs(
                    is_eplb_load_scope_matched(load_scope, any(is_prefilling)),
                    expected,
                )
