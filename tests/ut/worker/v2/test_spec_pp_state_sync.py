# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace

import numpy as np

from vllm_ascend.patch.worker.patch_v2.patch_spec_pp import (
    sync_spec_pp_num_computed_tokens_cpu,
)


class FakeEvent:
    def __init__(self):
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1


def make_runner(num_computed_tokens_np, prefill_len_np, device_copy_np):
    req_states = SimpleNamespace(
        req_id_to_index={"prefill": 0, "decode": 1},
        num_computed_tokens_np=np.array(num_computed_tokens_np, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array(prefill_len_np, dtype=np.int32)),
        num_computed_tokens_cpu=np.zeros(2, dtype=np.int32),
    )
    return SimpleNamespace(
        req_states=req_states,
        num_computed_tokens_cpu=np.array(device_copy_np, dtype=np.int32),
        num_computed_tokens_event=FakeEvent(),
    )


def make_scheduler_output(*req_ids):
    return SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=list(req_ids)))


class TestSpecPPNumComputedTokensSync(unittest.TestCase):
    def test_sampled_requests_read_the_device_copy_and_prefills_the_scheduler_value(self):
        # "decode" was sampled and had a draft token rejected, so only the D2H
        # copy holds its reverted length; "prefill" is inside a prefill chunk.
        runner = make_runner([4096, 123], [4425, 100], [7, 120])

        sync_spec_pp_num_computed_tokens_cpu(runner, make_scheduler_output("prefill", "decode"))

        np.testing.assert_array_equal(
            runner.req_states.num_computed_tokens_cpu,
            np.array([4096, 120], dtype=np.int32),
        )
        self.assertEqual(runner.num_computed_tokens_event.sync_count, 1)

    def test_fully_prefilling_batch_does_not_wait_on_the_copy_event(self):
        runner = make_runner([4096, 2048], [4425, 8192], [7, 9])

        sync_spec_pp_num_computed_tokens_cpu(runner, make_scheduler_output("prefill", "decode"))

        np.testing.assert_array_equal(
            runner.req_states.num_computed_tokens_cpu,
            np.array([4096, 2048], dtype=np.int32),
        )
        self.assertEqual(runner.num_computed_tokens_event.sync_count, 0)

    def test_empty_batch_is_a_noop(self):
        runner = make_runner([1, 2], [8, 8], [3, 4])

        sync_spec_pp_num_computed_tokens_cpu(runner, make_scheduler_output())

        np.testing.assert_array_equal(
            runner.req_states.num_computed_tokens_cpu,
            np.zeros(2, dtype=np.int32),
        )
        self.assertEqual(runner.num_computed_tokens_event.sync_count, 0)


if __name__ == "__main__":
    unittest.main()
