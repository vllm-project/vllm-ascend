# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.ops.state_ops import gather_initial_states


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gather_fresh_and_existing_states_with_poison_npu(poison, dtype):
    state = torch.full((4, 1, 4, 4), 7.0, device="npu", dtype=dtype)
    state[0] = poison
    state[1] = poison
    saved = state.clone()
    indices = torch.tensor([-1, 3, 100, 1, 0], device="npu", dtype=torch.int32)
    mask = torch.tensor([False, True, False, True, False], device="npu")
    actual = gather_initial_states(state, indices, mask)
    expected = torch.zeros((5, 1, 4, 4), dtype=dtype)
    expected[1] = 7.0
    expected[3] = poison
    assert actual.dtype == dtype and actual.device == state.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(state.cpu(), saved.cpu(), rtol=0, atol=0, equal_nan=True)


def test_gather_graph_replay_observes_changed_history_mask_and_poison():
    state = torch.full((4, 1, 4, 4), float("nan"), device="npu")
    state[2] = 5.0
    indices = torch.tensor([-1, 2], device="npu", dtype=torch.int32)
    mask = torch.tensor([False, True], device="npu")
    gather_initial_states(state, indices, mask)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = gather_initial_states(state, indices, mask)
    graph.replay()
    expected = torch.zeros((2, 1, 4, 4))
    expected[1] = 5.0
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    # Reuse graph/storage while the active history and poisoned pool row change.
    state[0] = float("inf")
    state[3] = 9.0
    indices.copy_(torch.tensor([3, 100], device="npu", dtype=torch.int32))
    mask.copy_(torch.tensor([True, False], device="npu"))
    graph.replay()
    expected[0] = 9.0
    expected[1] = 0.0
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
