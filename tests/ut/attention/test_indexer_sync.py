# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest

from vllm_ascend.attention.indexer_sync import synchronize_long_indexer_if_needed


class TestIndexerSync(unittest.TestCase):
    def test_synchronizes_for_a_long_prefill_that_publishes_the_shared_cache(self):
        calls = []

        synchronized = synchronize_long_indexer_if_needed(
            seq_len=1_000_000,
            num_query_tokens=65,
            pp_world_size=4,
            use_index_cache=True,
            synchronize=lambda: calls.append("sync"),
        )

        self.assertTrue(synchronized)
        self.assertEqual(calls, ["sync"])

    def test_every_other_case_keeps_its_current_behaviour(self):
        calls = []

        for kwargs in (
            {"seq_len": 999_999, "num_query_tokens": 65, "pp_world_size": 4, "use_index_cache": True},
            {"seq_len": 1_000_000, "num_query_tokens": 64, "pp_world_size": 4, "use_index_cache": True},
            {"seq_len": 1_000_000, "num_query_tokens": 65, "pp_world_size": 1, "use_index_cache": True},
            {"seq_len": 1_000_000, "num_query_tokens": 65, "pp_world_size": 4, "use_index_cache": False},
        ):
            self.assertFalse(
                synchronize_long_indexer_if_needed(**kwargs, synchronize=lambda: calls.append("unexpected"))
            )

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
