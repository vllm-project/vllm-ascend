# SPDX-License-Identifier: Apache-2.0
"""D128 native norm/gate qualification; run this file directly for an A/B benchmark."""

import argparse
import json
import statistics
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.ops.kda_rms_norm_gated import kda_rms_norm_gated
from vllm_ascend.ops.triton.kda.kda import rms_norm_gated as triton_norm_gate


def _inputs(tokens, heads, weight_dtype, strided_x=True, strided_gate=True):
    torch.manual_seed(20260921)
    width = heads * 128
    x_store = torch.randn(tokens, width + (32 if strided_x else 0)).bfloat16().to("npu")
    x = x_store[:, :width].view(1, tokens, heads, 128)
    # Actual K3 output-gate view: [beta(H), raw_gate(H*128), output_gate(H*128)].
    bfg = torch.randn(tokens, heads + 2 * width).bfloat16().to("npu")
    gate = bfg[:, heads + width :].view(tokens, heads, 128)
    if not strided_gate:
        gate = gate.contiguous()
    weight = torch.randn(128).to(dtype=weight_dtype, device="npu")
    storage = torch.full((1, tokens + 3, heads, 128), -123, dtype=torch.bfloat16, device="npu")
    return x, gate, weight, storage[:, :tokens], storage


def _reference(x, gate, weight, epsilon, sigmoid_only=False):
    value = x.float()
    rstd = 1 / torch.sqrt(value.square().mean(-1, keepdim=True) + epsilon)
    gated = (value * rstd) * weight.float()
    if not sigmoid_only:
        gated *= gate.float()
    return (gated * torch.sigmoid(gate.float())).bfloat16()


def _check(actual, expected):
    assert torch.isfinite(actual).all()
    # One BF16 representable step; no task-level or FP8 error budget applies.
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=1e-6)
    difference = actual.float() - expected.float()
    rrms = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)
    assert rrms <= 0.002
    return {"rrms": float(rrms), "max_abs": float(difference.abs().max())}


@pytest.mark.parametrize("tokens,heads", [(1, 1), (1, 12), (7, 12), (129, 12), (515, 12), (17, 128)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("sigmoid_only", [False, True])
@torch.inference_mode()
def test_kda_norm_gate_strided_input_tail_and_output_guard(tokens, heads, weight_dtype, sigmoid_only):
    x, gate, weight, out, storage = _inputs(tokens, heads, weight_dtype)
    result = kda_rms_norm_gated(x, gate, weight, eps=1e-6, out=out, sigmoid_only=sigmoid_only)
    assert result.data_ptr() == out.data_ptr()
    expected = _reference(x.cpu(), gate.cpu(), weight.cpu(), 1e-6, sigmoid_only)
    _check(result.cpu(), expected)
    assert torch.all(storage[:, tokens:].cpu() == -123)
    old = triton_norm_gate(x, gate, weight, None, "sigmoid" if sigmoid_only else "swish", eps=1e-6)
    _check(result.cpu(), old.cpu())


@pytest.mark.parametrize("sigmoid_only", [False, True])
@torch.inference_mode()
def test_kda_norm_gate_zero_and_saturated_gate(sigmoid_only):
    x, gate, weight, out, _ = _inputs(13, 12, torch.float32)
    x[:, :3].zero_()
    gate[3].fill_(-80)
    gate[4].fill_(80)
    result = kda_rms_norm_gated(x, gate, weight, eps=1e-5, out=out, sigmoid_only=sigmoid_only)
    expected = _reference(x.cpu(), gate.cpu(), weight.cpu(), 1e-5, sigmoid_only)
    _check(result.cpu(), expected)
    assert torch.count_nonzero(result[:, :3]) == 0


@pytest.mark.parametrize("tokens", [129, 16384])
@pytest.mark.parametrize("strided_gate", [False, True])
@pytest.mark.parametrize("sigmoid_only", [False, True])
@torch.inference_mode()
def test_kda_norm_gate_graph_reloads_changed_inputs(tokens, strided_gate, sigmoid_only):
    x, gate, weight, out, storage = _inputs(tokens, 12, torch.bfloat16, False, strided_gate)
    kda_rms_norm_gated(x, gate, weight, out=out, sigmoid_only=sigmoid_only)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = kda_rms_norm_gated(x, gate, weight, out=out, sigmoid_only=sigmoid_only)
    selected = torch.unique(torch.linspace(0, tokens - 1, min(tokens, 129), dtype=torch.int64)).to("npu")
    for scale in (1.0, 0.001, 100.0):
        x.copy_(torch.randn_like(x) * scale)
        gate.copy_(torch.randn_like(gate))
        weight.copy_(torch.randn_like(weight))
        graph.replay()
        expected = _reference(
            x.index_select(1, selected).cpu(), gate.index_select(0, selected).cpu(), weight.cpu(), 1e-6, sigmoid_only
        )
        _check(result.index_select(1, selected).cpu(), expected)
        # Compare all rows with the existing fused implementation as well.
        old = triton_norm_gate(x, gate, weight, None, "sigmoid" if sigmoid_only else "swish", eps=1e-6)
        _check(result.cpu(), old.cpu())
        assert torch.all(storage[:, tokens:].cpu() == -123)


@pytest.mark.parametrize("strided_gate", [False, True])
@torch.inference_mode()
def test_kimi_module_dispatches_sigmoid_to_native(strided_gate):
    from vllm_ascend.ops.layernorm import AscendFusedRMSNormGated

    x, gate, weight, out, storage = _inputs(16384, 12, torch.bfloat16, False, strided_gate)
    gate[7].zero_()  # sigmoid(0)=0.5; SiLU(0)=0 exposes an activation mismatch.
    # Exercise the actual module boundary without creating a model/config or
    # depending on upstream CustomOp constructor-registration side effects.
    layer = AscendFusedRMSNormGated.__new__(AscendFusedRMSNormGated)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.register_parameter("bias", None)
    layer.eps, layer.activation = 1e-6, "sigmoid"
    layer._forward_method = layer.forward_oot
    with patch("vllm_ascend.ops.layernorm.kda_rms_norm_gated", wraps=kda_rms_norm_gated) as native:
        result = layer.forward_prefill(x, gate, out=out)
    native.assert_called_once_with(x, gate, layer.weight, eps=1e-6, out=out, sigmoid_only=True)
    assert result.data_ptr() == out.data_ptr()
    selected = torch.tensor([0, 7, 127, 128, 511, 1023, 8191, 16383], device="npu")
    expected = _reference(
        x.index_select(1, selected).cpu(), gate.index_select(0, selected).cpu(), weight.cpu(), 1e-6, True
    )
    _check(result.index_select(1, selected).cpu(), expected)
    assert torch.all(storage[:, 16384:].cpu() == -123)


def _time(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


@torch.inference_mode()
def benchmark(tokens=16384, heads=12, warmup=10, iterations=30, activation="sigmoid"):
    sigmoid_only = activation == "sigmoid"
    for strided_x, strided_gate in ((False, True), (False, False), (True, True)):
        x, gate, weight, out, storage = _inputs(tokens, heads, torch.bfloat16, strided_x, strided_gate)
        native = lambda x=x, gate=gate, weight=weight, out=out: kda_rms_norm_gated(
            x, gate, weight, eps=1e-6, out=out, sigmoid_only=sigmoid_only
        )
        original = lambda x=x, gate=gate, weight=weight, out=out: triton_norm_gate(
            x, gate, weight, None, activation, eps=1e-6, out=out
        )
        native()
        selected = torch.unique(torch.linspace(0, tokens - 1, min(tokens, 129), dtype=torch.int64)).to("npu")
        actual = out.index_select(1, selected).cpu()
        expected = _reference(
            x.index_select(1, selected).cpu(), gate.index_select(0, selected).cpu(), weight.cpu(), 1e-6, sigmoid_only
        )
        errors = _check(actual, expected)
        assert torch.all(storage[:, tokens:].cpu() == -123)
        for mode in ("eager", "graph"):
            if mode == "graph":
                original()
                native()
                torch.npu.synchronize()
                old_graph, new_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
                with torch.npu.graph(old_graph):
                    for _ in range(iterations):
                        original()
                with torch.npu.graph(new_graph):
                    for _ in range(iterations):
                        native()
                old_call, new_call = old_graph.replay, new_graph.replay
            else:
                old_call, new_call = original, native
            old_samples, new_samples = [], []
            for repeat in range(3):
                calls = ((old_call, old_samples), (new_call, new_samples))
                for function, samples in calls if repeat % 2 == 0 else reversed(calls):
                    # One replay launches the whole captured group, avoiding
                    # Python replay gaps being mistaken for device runtime.
                    if mode == "graph":
                        samples.append(_time(function, min(warmup, 3), 1) / iterations)
                    else:
                        samples.append(_time(function, warmup, iterations))
            old_ms, new_ms = statistics.median(old_samples), statistics.median(new_samples)
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "heads": heads,
                        "x_stride": list(x.stride()),
                        "gate_stride": list(gate.stride()),
                        "weight_dtype": str(weight.dtype),
                        "mode": mode,
                        "activation": activation,
                        "calls_per_graph": iterations if mode == "graph" else 0,
                        "triton_ms": old_ms,
                        "ascendc_ms": new_ms,
                        "speedup": old_ms / new_ms,
                        "triton_samples_ms": old_samples,
                        "ascendc_samples_ms": new_samples,
                        "sampled_query_tokens": selected.numel(),
                        **errors,
                    }
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--activation", choices=("sigmoid", "swish", "silu"), default="sigmoid")
    benchmark(**vars(parser.parse_args()))
