# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from vllm_ascend.ops.triton.flash_attention_output import flash_attention_gate, register_flash_attention_gate_warmup


@pytest.mark.parametrize(
    "tokens,hidden",
    [
        (1, 1536),
        (4, 1536),
        (8, 1536),
        (32, 1536),
        (64, 1536),
        (128, 1536),
        (1024, 1536),
        (8, 768),
        (8, 3072),
        (8, 12288),
        (17, 129),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_gate_matches_materialized_sigmoid(tokens, hidden, dtype, strided):
    torch.manual_seed(42)
    a_base = torch.randn(tokens, hidden + (16 if strided else 0), device="npu", dtype=dtype)
    g_base = torch.randn(tokens, hidden + (8 if strided else 0), device="npu", dtype=dtype)
    projected, gate = a_base[:, :hidden], g_base[:, :hidden]
    padding = a_base[:, hidden:].clone()
    expected = projected * torch.sigmoid(gate)
    assert flash_attention_gate(projected, gate) is projected
    tolerance = {"atol": 0, "rtol": 0} if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(projected, expected, **tolerance)
    torch.testing.assert_close(a_base[:, hidden:], padding, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_nonfinite_and_saturated_gate(dtype):
    gate = torch.tensor(
        [float("inf"), -float("inf"), float("nan"), -100, 100, -0.0, 0.0], device="npu", dtype=dtype
    ).repeat(3, 1)
    a = torch.tensor([1.0, -float("inf"), float("nan"), -1.0, 2.0, -0.0, 0.0], device="npu", dtype=dtype).repeat(3, 1)
    expected = a * torch.sigmoid(gate)
    flash_attention_gate(a, gate)
    torch.testing.assert_close(a, expected, equal_nan=True)


@pytest.mark.parametrize("tokens", [1, 8, 32, 128])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_graph_replay_uses_updated_values(tokens, strided):
    hidden = 1536
    a = torch.zeros(tokens, hidden + (16 if strided else 0), device="npu", dtype=torch.bfloat16)[:, :hidden]
    gate = torch.zeros(tokens, hidden + (8 if strided else 0), device="npu", dtype=torch.bfloat16)[:, :hidden]
    flash_attention_gate(a, gate)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        flash_attention_gate(a, gate)
    for seed in range(3):
        torch.manual_seed(seed)
        a.normal_()
        gate.normal_()
        expected = a * torch.sigmoid(gate)
        graph.replay()
        torch.testing.assert_close(a, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_all_bf16_gate_patterns_preserve_intermediate_rounding():
    gate = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(256, 256).to("npu")
    projected = torch.full_like(gate, 0.8671875)
    expected = projected * torch.sigmoid(gate)
    flash_attention_gate(projected, gate)
    torch.testing.assert_close(projected, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("shape", [(0, 1536), (8, 0)])
def test_empty(shape):
    a = torch.empty(shape, device="npu", dtype=torch.bfloat16)
    assert flash_attention_gate(a, a) is a


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("gate_prefix", [0, 576, 1])
@torch.inference_mode()
def test_registry_warmup_prevents_runtime_compilation(dtype, gate_prefix, monkeypatch):
    """All tile thresholds and fused views must hit precompiled variants.

    Run with a fresh TRITON_CACHE_DIR to exercise cold compile-only startup.
    No operator is executed before compiler entry points are forbidden.
    """
    hidden = 1536
    registry = JitWarmupRegistry(None)
    with registry.activate():
        register_flash_attention_gate_warmup(dtype, hidden, gate_prefix)
    registry.warmup()

    def forbid_compile(*args, **kwargs):
        raise AssertionError("Output gate compiled after registry warmup")

    for name in ("triton", "triton.compiler", "triton.compiler.compiler", "triton.runtime.jit"):
        module = importlib.import_module(name)
        if callable(getattr(module, "compile", None)):
            monkeypatch.setattr(module, "compile", forbid_compile)
    for tokens in (1, 5, 6, 42, 43, 682, 683, 1024):
        a = torch.randn(tokens, hidden, device="npu", dtype=dtype)
        g_base = torch.randn(tokens, hidden + gate_prefix, device="npu", dtype=dtype)
        g = g_base[:, gate_prefix:]
        gate_before = g_base.clone()
        expected = a * torch.sigmoid(g)
        assert flash_attention_gate(a, g) is a
        tolerance = {"atol": 0, "rtol": 0} if dtype == torch.bfloat16 else {}
        torch.testing.assert_close(a, expected, **tolerance)
        torch.testing.assert_close(g_base, gate_before, atol=0, rtol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        flash_attention_gate(a, g)
    a.normal_()
    g.normal_()
    expected = a * torch.sigmoid(g)
    graph.replay()
    torch.testing.assert_close(a, expected, **tolerance)
