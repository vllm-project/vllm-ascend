# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
from unittest.mock import MagicMock

import pytest
import torch

adapter = importlib.import_module("vllm_ascend.ops.fused_moe.shared_dequant_swiglu_quant")


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 7, 8, 9, 64])
@pytest.mark.parametrize("width", [256, 384, 4096, 6144])
@pytest.mark.parametrize("limit", [1.0, 2.0, 10.0, 100.0])
def test_shared_adapter_layout_padding_and_parameters(monkeypatch, num_tokens, width, limit):
    x = torch.arange(num_tokens * width, dtype=torch.int32).reshape(num_tokens, width)
    weight_scale = torch.arange(width, dtype=torch.float32) + 1
    activation_scale = torch.arange(num_tokens, dtype=torch.float32) + 1
    original_x, original_weight = x.clone(), weight_scale.clone()
    padded_tokens = max(num_tokens, 8)
    output = torch.ones((padded_tokens, width // 2), dtype=torch.int8)
    scale = torch.ones(padded_tokens, dtype=torch.float32)
    operator = MagicMock(return_value=(output, scale))
    monkeypatch.setattr(adapter.torch_npu, "npu_dequant_swiglu_quant", operator, raising=False)

    actual, actual_scale = adapter.shared_dequant_swiglu_quant(x, weight_scale, activation_scale, limit)

    assert actual.shape == (num_tokens, width // 2)
    assert actual_scale.shape == (num_tokens,)
    assert actual.dtype == torch.int8
    assert actual_scale.dtype == torch.float32
    assert torch.equal(x, original_x)
    assert torch.equal(weight_scale, original_weight)
    if num_tokens == 0:
        operator.assert_not_called()
        return

    kwargs = operator.call_args.kwargs
    assert kwargs["x"].is_contiguous()
    assert torch.equal(kwargs["x"][:num_tokens, 0::2], x[:, : width // 2])
    assert torch.equal(kwargs["x"][:num_tokens, 1::2], x[:, width // 2 :])
    assert torch.count_nonzero(kwargs["x"][num_tokens:]) == 0
    assert torch.equal(kwargs["weight_scale"][0, 0::2], weight_scale[: width // 2])
    assert torch.equal(kwargs["weight_scale"][0, 1::2], weight_scale[width // 2 :])
    assert torch.equal(kwargs["activation_scale"][:num_tokens], activation_scale)
    assert torch.all(kwargs["activation_scale"][num_tokens:] == 1)
    assert kwargs["group_index"].tolist() == [padded_tokens]
    assert kwargs["group_index"].dtype == torch.int64
    assert kwargs["group_index"].device == x.device
    assert kwargs["clamp_limit"] == limit
    assert kwargs["glu_alpha"] == 1.0
    assert kwargs["glu_bias"] == 0.0
    assert kwargs["activate_left"] is True
    assert kwargs["quant_mode"] == kwargs["swiglu_mode"] == 1
    assert torch.equal(actual, output[:num_tokens])
    assert torch.equal(actual_scale, scale[:num_tokens])
