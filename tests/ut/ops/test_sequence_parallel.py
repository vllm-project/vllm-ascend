#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import torch

from vllm_ascend.ops.sequence_parallel import (
    gather_sequence_activation,
    reduce_partial_activation,
    shard_sequence_activation,
)
from vllm_ascend.sequence_parallel import SequenceParallelActivationState, SequenceParallelRuntimeState


def _partial_state(*, active: bool) -> SequenceParallelRuntimeState:
    return SequenceParallelRuntimeState.create(
        active=active,
        world_size=2,
        num_tokens=3,
    ).transition_to(SequenceParallelActivationState.TP_PARTIAL)


@patch(
    "vllm_ascend.ops.sequence_parallel.tensor_model_parallel_all_reduce",
    side_effect=lambda tensor: tensor + 1,
)
def test_reduce_partial_activation_uses_all_reduce(mock_all_reduce):
    tensor = torch.arange(6).view(3, 2)

    output, state = reduce_partial_activation(tensor, _partial_state(active=False))

    torch.testing.assert_close(output, tensor + 1)
    assert state.activation is SequenceParallelActivationState.FULL
    mock_all_reduce.assert_called_once()


@patch(
    "vllm_ascend.ops.sequence_parallel.tensor_model_parallel_reduce_scatter",
    side_effect=lambda tensor, dim: torch.chunk(tensor, 2, dim=dim)[0],
)
def test_reduce_partial_activation_pads_then_reduce_scatters(mock_reduce_scatter):
    tensor = torch.arange(6).view(3, 2)

    output, state = reduce_partial_activation(tensor, _partial_state(active=True))

    assert output.shape == (2, 2)
    assert state.activation is SequenceParallelActivationState.SEQUENCE_SHARDED
    padded = mock_reduce_scatter.call_args.args[0]
    assert padded.shape == (4, 2)
    torch.testing.assert_close(padded[-1], torch.zeros(2, dtype=tensor.dtype))


@patch(
    "vllm_ascend.ops.sequence_parallel.tensor_model_parallel_all_gather",
    side_effect=lambda tensor, dim: torch.cat((tensor, tensor), dim=dim),
)
def test_gather_sequence_activation_unpads(mock_all_gather):
    state = _partial_state(active=True).transition_to(SequenceParallelActivationState.SEQUENCE_SHARDED)
    tensor = torch.arange(4).view(2, 2)

    output, state = gather_sequence_activation(tensor, state)

    assert output.shape == (3, 2)
    assert state.activation is SequenceParallelActivationState.FULL
    mock_all_gather.assert_called_once()


@patch("vllm_ascend.ops.sequence_parallel.get_tensor_model_parallel_rank", return_value=1)
def test_shard_sequence_activation_selects_local_rank(mock_rank):
    state = SequenceParallelRuntimeState.create(
        active=True,
        world_size=2,
        num_tokens=3,
    )
    tensor = torch.arange(6).view(3, 2)

    output, state = shard_sequence_activation(tensor, state)

    expected = torch.tensor([[4, 5], [0, 0]])
    torch.testing.assert_close(output, expected)
    assert state.activation is SequenceParallelActivationState.SEQUENCE_SHARDED
    mock_rank.assert_called_once()
