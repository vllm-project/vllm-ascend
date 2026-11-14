# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""CPU coverage for Ascend Engram's WKV TP/SP contract.

Collective transport and quantization kernels are outside these tests. The
numerical cases exercise the real ColumnParallelLinear forward/weight_loader
with a CPU linear method and a mocked TP all-gather.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "vllm.models.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm.model_executor.layers import linear as linear_mod

from vllm_ascend.models.deepseek_v41.engram import layer as layer_mod
from vllm_ascend.models.deepseek_v41.engram.common import engram_gate


def _cpu_linear(module, inputs, bias):
    return torch.nn.functional.linear(inputs, module.weight, bias)


def _config():
    # 160 output features / TP8 = 20: one shard crosses the key/value boundary.
    return SimpleNamespace(
        hidden_size=32,
        hc_mult=4,
        rms_norm_eps=1e-6,
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=3,
    )


@pytest.mark.parametrize("sp", [None, False, True], ids=["default", "tp", "sp"])
def test_wkv_constructor_preserves_parallel_and_checkpoint_contract(monkeypatch, sp):
    config = _config()
    quant_config = object()
    factory = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(layer_mod, "ColumnParallelLinear", factory)
    kwargs = {} if sp is None else {"use_sequence_parallel": sp}

    layer = layer_mod.AscendEngram(config, quant_config, "model.layers.14.engram", **kwargs)

    factory.assert_called_once_with(
        12,
        160,
        bias=False,
        gather_output=True,
        disable_tp=False if sp is None else sp,
        quant_config=quant_config,
        prefix="model.layers.14.engram.wkv",
        return_bias=False,
    )
    assert layer.wkv is factory.return_value
    assert layer.q_weight.shape == layer.k_weight.shape == (4, 32)
    assert layer.q_weight.dtype == layer.k_weight.dtype == torch.bfloat16


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 1, 7], ids=["empty", "single", "mixed-mask"])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_wkv_shards_restore_full_gate_input(monkeypatch, tp_size, sp, num_tokens, rotated, dtype):
    """Protect shard order, SP row ownership, and the unchanged gate interface."""
    config = _config()
    generator = torch.Generator().manual_seed(17)
    full_weight = torch.randn(160, 12, generator=generator, dtype=torch.float32).to(dtype)
    key_weight = torch.randn(4, 32, generator=generator).bfloat16()
    query_weight = torch.randn(4, 32, generator=generator).bfloat16()
    rotation = torch.eye(4).roll(1, dims=0) if rotated else None
    token_mask = torch.ones(num_tokens, dtype=torch.bool)
    if num_tokens > 1:
        token_mask[1::2] = False

    for rank in range(tp_size):
        # Under SP each rank owns different rows; collecting across them is invalid.
        rows = (torch.randn(num_tokens, 12, generator=generator) + (rank if sp else 0)).to(dtype)
        hidden = torch.randn(num_tokens, 4, 32, generator=generator).to(dtype)
        effective_tp = 1 if sp else tp_size
        effective_rank = 0 if sp else rank
        shard_size = 160 // effective_tp

        def gather(
            local_output,
            rows=rows,
            effective_rank=effective_rank,
            shard_size=shard_size,
            effective_tp=effective_tp,
        ):
            expected_local = torch.nn.functional.linear(
                rows, full_weight.narrow(0, effective_rank * shard_size, shard_size)
            )
            torch.testing.assert_close(local_output, expected_local, rtol=0, atol=0)
            # Simulate peer rank outputs in rank order, not actual HCCL transport.
            return torch.cat(
                [torch.nn.functional.linear(rows, shard) for shard in full_weight.chunk(effective_tp, dim=0)],
                dim=-1,
            )

        all_gather = Mock(side_effect=gather)
        monkeypatch.setattr(linear_mod, "tensor_model_parallel_all_gather", all_gather)

        def make_projection(
            input_size,
            output_size,
            *,
            effective_tp=effective_tp,
            effective_rank=effective_rank,
            shard_size=shard_size,
            **kwargs,
        ):
            assert (input_size, output_size) == (12, 160)
            assert kwargs["disable_tp"] is sp
            # Bypass platform-specific allocation/quantization initialization.
            # Forward and weight_loader below are the real upstream methods.
            projection = object.__new__(linear_mod.ColumnParallelLinear)
            torch.nn.Module.__init__(projection)
            projection.tp_size = effective_tp
            projection.tp_rank = effective_rank
            projection.gather_output = kwargs["gather_output"]
            projection.skip_bias_add = False
            projection.return_bias = kwargs["return_bias"]
            projection.bias = None
            projection.weight = torch.nn.Parameter(
                torch.empty(shard_size, input_size, dtype=dtype), requires_grad=False
            )
            projection.weight.output_dim = 0
            projection.quant_method = SimpleNamespace(apply=_cpu_linear)
            linear_mod.ColumnParallelLinear.weight_loader(projection, projection.weight, full_weight)
            torch.testing.assert_close(
                projection.weight,
                full_weight.narrow(0, effective_rank * shard_size, shard_size),
                rtol=0,
                atol=0,
            )
            return projection

        monkeypatch.setattr(layer_mod, "ColumnParallelLinear", make_projection)
        layer = layer_mod.AscendEngram(config, None, "model.layers.1.engram", use_sequence_parallel=sp)
        with torch.no_grad():
            layer.q_weight.copy_(query_weight)
            layer.k_weight.copy_(key_weight)

        full_output = torch.nn.functional.linear(rows, full_weight)
        key, value = full_output.split([128, 32], dim=-1)
        expected = engram_gate(
            hidden,
            key.reshape(num_tokens, 4, 32),
            value,
            query_weight.float() * key_weight.float(),
            rotation,
            token_mask,
            config.rms_norm_eps,
        )
        actual = layer(hidden, rows, token_mask, rotation)

        assert actual.shape == hidden.shape
        assert actual.dtype == dtype
        tolerance = 0.02 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual[~token_mask], hidden[~token_mask], rtol=0, atol=0)
        assert all_gather.call_count == int(effective_tp > 1)


def test_wkv_projection_errors_are_not_swallowed(monkeypatch):
    error = ValueError("projection output dimension is incompatible with TP")
    monkeypatch.setattr(layer_mod, "ColumnParallelLinear", Mock(side_effect=error))

    with pytest.raises(ValueError) as caught:
        layer_mod.AscendEngram(_config(), None, "model.layers.1.engram")

    assert caught.value is error


def test_fused_gate_receives_the_complete_projected_tensor(monkeypatch):
    config = _config()
    projection = Mock(return_value=torch.randn(3, 160))
    monkeypatch.setattr(layer_mod, "ColumnParallelLinear", lambda *args, **kwargs: projection)
    layer = layer_mod.AscendEngram(config, None, "model.layers.1.engram")
    profile = SimpleNamespace(supports=Mock(return_value=True))
    monkeypatch.setattr(layer_mod, "get_current_hardware_profile", lambda: profile)
    expected = object()
    fused = Mock(return_value=expected)
    monkeypatch.setattr(layer_mod, "fused_engram_gate", fused)
    hidden = SimpleNamespace(device=SimpleNamespace(type="npu"))
    rows = torch.randn(3, 12)
    mask = torch.tensor([True, False, True])

    assert layer.forward(hidden, rows, mask, None) is expected

    projection.assert_called_once_with(rows)
    profile.supports.assert_called_once_with(layer_mod.HardwareCapability.ENGRAM_UNROTATED_GATE)
    fused.assert_called_once_with(
        hidden, projection.return_value, layer.q_weight, layer.k_weight, mask, config.rms_norm_eps
    )
