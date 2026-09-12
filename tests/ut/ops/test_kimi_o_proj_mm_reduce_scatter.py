# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ops import linear_op
from vllm_ascend.ops.linear import AscendRowParallelLinear


def _make_layer(weight):
    return SimpleNamespace(
        weight=weight,
        bias=None,
        custom_op=None,
        quant_method=UnquantizedLinearMethod(),
        input_is_parallel=True,
        input_size_per_partition=weight.shape[1],
        reduce_results=False,
        return_bias=True,
        skip_bias_add=False,
        prefix="model.layers.0.self_attn.o_proj",
    )


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 8, 17])
@pytest.mark.parametrize("last_rank", [False, True])
def test_fused_o_proj_matches_padded_partial_sum(monkeypatch, world_size, num_tokens, last_rank):
    rank = world_size - 1 if last_rank else 0
    generator = torch.Generator().manual_seed(42)
    # Strided activations exercise the fused operator's contiguous-input rule.
    inputs = torch.randint(-2, 3, (world_size, num_tokens, 512), generator=generator).to(torch.bfloat16)[..., ::2]
    weights = torch.randint(-2, 3, (world_size, 32, 256), generator=generator).to(torch.bfloat16) / 8
    padding = (-num_tokens) % world_size
    partials = [
        torch.nn.functional.pad(torch.nn.functional.linear(x, weight), (0, 0, 0, padding))
        for x, weight in zip(inputs, weights)
    ]
    expected = torch.stack(partials).sum(0).chunk(world_size, dim=0)[rank]
    backend = MagicMock()
    backend.get_hccl_comm_name.return_value = "kimi_tp"
    device_group = MagicMock()
    device_group._get_backend.return_value = backend
    tp_group = SimpleNamespace(device_group=device_group, rank_in_group=rank, world_size=world_size)
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: tp_group)
    layer = _make_layer(weights[rank])
    op = linear_op.KimiOProjMMReduceScatterOp(layer)
    calls = []

    def fused_mm(x, weight, hcom, size, **kwargs):
        calls.append((hcom, size, kwargs))
        assert x.is_contiguous()
        assert x.shape == (num_tokens + padding, 256)
        assert weight.data_ptr() == layer.weight.data_ptr()
        torch.testing.assert_close(weight, layer.weight.t())
        local_partial = x @ weight
        torch.testing.assert_close(local_partial, partials[rank])
        contributions = list(partials)
        contributions[rank] = local_partial
        return torch.stack(contributions).sum(0).chunk(size, dim=0)[rank], torch.empty(0)

    monkeypatch.setattr(linear_op.torch_npu, "npu_quant_mm_reduce_scatter", fused_mm, raising=False)

    output, bias = op.apply(inputs[rank])

    torch.testing.assert_close(output, expected)
    assert bias is None
    assert calls == [("kimi_tp", world_size, {"reduce_op": "sum", "comm_mode": "ai_cpu"})]
    backend.get_hccl_comm_name.assert_called_once_with(rank)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
def test_fused_o_proj_rejects_non_bf16_weights(dtype):
    with pytest.raises(ValueError, match="unquantized BF16"):
        linear_op.KimiOProjMMReduceScatterOp(_make_layer(torch.zeros(32, 256, dtype=dtype)))


@pytest.mark.parametrize(
    "attribute,value,error",
    [
        ("quant_method", object(), "unquantized BF16"),
        ("custom_op", object(), "original TP group"),
        ("bias", torch.zeros(32), "bias-free"),
    ],
)
def test_fused_o_proj_rejects_incompatible_projection(attribute, value, error):
    layer = _make_layer(torch.zeros(32, 256, dtype=torch.bfloat16))
    setattr(layer, attribute, value)
    with pytest.raises(ValueError, match=error):
        linear_op.KimiOProjMMReduceScatterOp(layer)


def test_row_parallel_linear_dispatches_fused_projection(monkeypatch):
    monkeypatch.setattr("vllm_ascend.ops.linear.get_parallel_op", lambda *_: (None, 0, 2))
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 2)
    device_group = MagicMock()
    device_group._get_backend.return_value.get_hccl_comm_name.return_value = "kimi_tp"
    tp_group = SimpleNamespace(device_group=device_group, rank_in_group=0, world_size=2)
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: tp_group)
    layer = AscendRowParallelLinear(
        512, 32, bias=False, params_dtype=torch.bfloat16, reduce_results=False, prefix="model.layers.0.self_attn.o_proj"
    )
    layer.custom_op = linear_op.KimiOProjMMReduceScatterOp(layer)
    expected = torch.ones(2, 32, dtype=torch.bfloat16)
    fused = MagicMock(return_value=(expected, torch.empty(0)))
    monkeypatch.setattr(linear_op.torch_npu, "npu_quant_mm_reduce_scatter", fused, raising=False)

    output, bias = layer(torch.ones(3, 256, dtype=torch.bfloat16), is_prefill=True)

    assert output is expected
    assert bias is None
    fused.assert_called_once()
    assert fused.call_args.args[0].shape == (4, 256)
