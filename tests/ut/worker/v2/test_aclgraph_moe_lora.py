# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


@pytest.mark.parametrize(
    ("local_loras", "global_loras", "synchronized", "expected"),
    [(0, 2, True, 2), (1, 2, True, 2), (0, 0, True, 0), (0, 2, False, 0), (3, 0, True, 0), (3, 1, True, 1)],
)
def test_remote_expert_adapters_select_active_lora_graph(local_loras, global_loras, synchronized, expected):
    manager = object.__new__(ModelAclGraphManager)
    manager.model_runner = SimpleNamespace(
        _moe_lora_coordinator=SimpleNamespace(step_active=synchronized, num_active_loras=global_loras)
    )
    with patch.object(ModelCudaGraphManager, "dispatch") as dispatch:
        result = manager.dispatch(1, 16, 1, local_loras, max_query_len=1, num_ubatches=2)
    assert result is dispatch.return_value
    dispatch.assert_called_once_with(1, 16, 1, expected, max_query_len=1, num_ubatches=2)


def test_dense_and_single_dp_graph_selection_is_unchanged():
    manager = object.__new__(ModelAclGraphManager)
    manager.model_runner = SimpleNamespace()
    with patch.object(ModelCudaGraphManager, "dispatch") as dispatch:
        manager.dispatch(1, 16, 1, 0)
    dispatch.assert_called_once_with(1, 16, 1, 0, max_query_len=None, num_ubatches=1)
