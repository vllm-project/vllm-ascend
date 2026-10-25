# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm_ascend.worker.v1.model_runner import NPUModelRunner
from vllm_ascend.worker.v1.spec_decode.eagle_proposer import AscendEagleProposer


@pytest.mark.parametrize("with_drafter", [False, True])
@pytest.mark.parametrize("use_ubatching", [False, True])
@pytest.mark.parametrize("kernel_block_sizes", [[128], [128, 64]])
def test_target_metadata_builders_do_not_initialize_pp_drafter(with_drafter, use_ubatching, kernel_block_sizes):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(use_eagle=lambda: True)
    runner.drafter = MagicMock(spec=AscendEagleProposer) if with_drafter else None
    runner.vllm_config = MagicMock()
    runner.device = "cpu"
    runner.parallel_config = SimpleNamespace(use_ubatching=use_ubatching, num_ubatches=3)
    groups = [MagicMock(), MagicMock(), MagicMock()]
    runner.attn_groups = [[groups[0]], groups[1:]]
    runner.calculate_reorder_batch_threshold = MagicMock()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    events = MagicMock()
    for index, group in enumerate(groups):
        events.attach_mock(group.create_metadata_builders, f"group_{index}")
    events.attach_mock(runner.calculate_reorder_batch_threshold, "reorder")

    runner.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

    num_builders = 3 if use_ubatching else 1
    second_block_size = kernel_block_sizes[1] if len(kernel_block_sizes) > 1 else None
    assert events.mock_calls == [
        call.group_0(runner.vllm_config, "cpu", 128, num_metadata_builders=num_builders),
        call.group_1(runner.vllm_config, "cpu", second_block_size, num_metadata_builders=num_builders),
        call.group_2(runner.vllm_config, "cpu", second_block_size, num_metadata_builders=num_builders),
        call.reorder(),
    ]
    if runner.drafter is not None:
        # Ascend initializes the drafter only after binding KV tensors, in
        # initialize_kv_cache. Non-final PP stages have no drafter at all.
        runner.drafter.initialize_attn_backend.assert_not_called()


def test_target_metadata_builders_without_speculation():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculative_config = None
    runner.drafter = None
    runner.attn_groups = []
    runner.calculate_reorder_batch_threshold = MagicMock()

    runner.initialize_metadata_builders(SimpleNamespace(kv_cache_groups=[]), [])

    runner.calculate_reorder_batch_threshold.assert_called_once_with()
