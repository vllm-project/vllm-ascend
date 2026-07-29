# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.ops import register_custom_ops


def test_pad_and_reduce_uses_tp_reduce_scatter(monkeypatch):
    reduce_scatter = MagicMock(side_effect=lambda tensor, dim: tensor[::2])

    monkeypatch.setattr(register_custom_ops, "tensor_model_parallel_reduce_scatter", reduce_scatter)
    monkeypatch.setattr(register_custom_ops, "_EXTRA_CTX", SimpleNamespace(pad_size=0))

    input_tensor = torch.arange(8).view(4, 2)
    output = register_custom_ops._pad_and_reduce_impl(input_tensor)

    reduce_scatter.assert_called_once_with(input_tensor, 0)
    assert torch.equal(output, input_tensor[::2])
