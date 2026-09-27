# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.common.ops import sequence_parallel as sp


@pytest.mark.parametrize("custom_collective", [False, True])
@pytest.mark.parametrize("num_tokens,tp_size", [(8, 4), (10, 4), (1, 4), (3, 1), (0, 4)])
def test_reduce_scatter_padding(monkeypatch, custom_collective, num_tokens, tp_size):
    x = torch.arange(num_tokens * 8, dtype=torch.float32).reshape(num_tokens, 8)
    padding = (-num_tokens) % tp_size
    expected = torch.nn.functional.pad(x, (0, 0, 0, padding))
    inputs = []

    def reduce_scatter(value):
        inputs.append(value)
        torch.testing.assert_close(value, expected)
        if padding == 0:
            # The aligned path must preserve the input buffer, avoiding Cat.
            assert value is x
        return value[: value.shape[0] // tp_size]

    custom = Mock(side_effect=reduce_scatter) if custom_collective else Mock(return_value=None)
    fallback = Mock(side_effect=lambda value, dim: reduce_scatter(value))
    communicator = SimpleNamespace(custom_reduce_scatter=custom)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(sp, "get_tp_group", lambda: SimpleNamespace(device_communicator=communicator))
    monkeypatch.setattr(sp, "tensor_model_parallel_reduce_scatter", fallback)

    result = sp.sp_reduce_scatter(x)

    assert len(inputs) == 1
    torch.testing.assert_close(result, expected[: expected.shape[0] // tp_size])
    custom.assert_called_once()
    if custom_collective:
        fallback.assert_not_called()
    else:
        fallback.assert_called_once_with(inputs[0], 0)
