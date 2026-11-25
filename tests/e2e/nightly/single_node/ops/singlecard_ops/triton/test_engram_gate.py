# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and replay coverage for post-wkv Engram fusion on Ascend.

TRITON_INTERPRET=1 runs arithmetic coverage on CPU; NPU runs also exercise
the Ascend compiler, device execution and ACL graph replay.
"""

import os
import statistics

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.engram_gate import fused_engram_gate
from vllm_ascend.ops.triton.engram_gate_a3 import fused_engram_gate_a3

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _inputs(tokens, dim, dtype, strided, device):
    generator = torch.Generator().manual_seed(17)
    step = 2 if strided else 1
    hidden = torch.randn(tokens * step, 4, dim, generator=generator).to(dtype=dtype, device=device)[::step]
    kv = torch.randn(tokens * step, 5 * dim, generator=generator).to(dtype=dtype, device=device)[::step]
    q = torch.randn(4, dim, generator=generator).to(dtype=dtype, device=device)
    k = torch.randn(4, dim, generator=generator).to(dtype=dtype, device=device)
    mask = (torch.arange(tokens * step, device=device) % 3 != 1)[::step]
    # Non-identity orthogonal block makes a wrong-basis gate observable.
    rotation = torch.linalg.qr(torch.randn(32, 32, generator=generator)).Q.to(device)
    return hidden, kv, q, k, mask, rotation


def _inverse_rotate(hidden, rotation):
    return (hidden.float().unflatten(-1, (-1, 32)) @ rotation.T).flatten(-2)


def _reference(hidden, kv, q, k, mask, original):
    tokens, branches, dim = hidden.shape
    key, value = kv.split([branches * dim, dim], -1)
    key = key.view(tokens, branches, dim).float()
    rstd = torch.rsqrt(original.square().mean(-1) + 1e-6)
    rstd *= torch.rsqrt(key.square().mean(-1) + 1e-6)
    dot = (original * (q.float() * k.float()) * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    return (hidden.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)


@pytest.mark.parametrize(
    "tokens,dim,strided,rotated",
    [
        (1, 96, False, True),
        (132, 5120, False, True),
        (432, 5120, True, True),
        (7, 96, True, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_engram_gate_matches_unfused(tokens, dim, strided, rotated, dtype):
    device = "cpu" if INTERPRET else "npu"
    hidden, kv, q, k, mask, rotation = _inputs(tokens, dim, dtype, strided, device)
    original = _inverse_rotate(hidden, rotation) if rotated else hidden.float()
    if rotated:
        actual = fused_engram_gate_a3(hidden, kv, q, k, rotation, mask, 1e-6)
    else:
        actual = fused_engram_gate(hidden, kv, q, k, mask, 1e-6)
    # CPU FP32 reference is independent of the device's reduction implementation.
    expected = _reference(*(x.cpu() for x in (hidden, kv, q, k, mask, original)))
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    actual_cpu = actual.cpu()
    torch.testing.assert_close(actual_cpu, expected, rtol=tolerance, atol=tolerance)
    error = (actual_cpu.float() - expected.float()).abs()
    print(
        f"tokens={tokens} dim={dim} dtype={dtype} rotated={rotated} strided={strided}: "
        f"max_abs={error.max().item():.8f} mean_abs={error.mean().item():.8f}"
    )
    torch.testing.assert_close(actual[~mask], hidden[~mask], rtol=0, atol=0)
    if rotated:
        wrong_basis = _reference(hidden, kv, q, k, mask, hidden.float())
        assert (actual[mask].float() - wrong_basis[mask].float()).abs().max() > 0.05


@pytest.mark.skipif(INTERPRET, reason="ACL graphs require NPU execution")
@torch.inference_mode()
def test_engram_gate_replay_reads_updated_inputs():
    hidden, kv, q, k, mask, rotation = _inputs(132, 5120, torch.bfloat16, False, "npu")

    def operation():
        return fused_engram_gate_a3(hidden, kv, q, k, rotation, mask, 1e-6)

    for _ in range(3):
        operation()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = operation()
    for scale in (0.25, -0.5):
        hidden.mul_(scale)
        kv.add_(0.125)
        mask.logical_not_()
        graph.replay()
        original = _inverse_rotate(hidden, rotation)
        expected = _reference(*(x.cpu() for x in (hidden, kv, q, k, mask, original)))
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(INTERPRET, reason="Dispatch coverage requires real NPU tensors")
@pytest.mark.parametrize(
    "hardware,rotated,expected_call",
    [
        ("A3", True, "fused_engram_gate_a3"),
        ("A5", False, "fused_engram_gate"),
        ("A3", False, "engram_gate"),
        ("A5", True, "engram_gate"),
        ("A2", True, "engram_gate"),
        ("A2", False, "engram_gate"),
    ],
)
@torch.inference_mode()
def test_engram_gate_dispatch_preserves_hardware_paths(monkeypatch, hardware, rotated, expected_call):
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.device.hardware_profile import get_hardware_profile
    from vllm_ascend.models.deepseek_v41.engram import layer as layer_module

    hidden, kv, q, k, mask, rotation = _inputs(2, 96, torch.bfloat16, False, "npu")
    layer = object.__new__(layer_module.AscendEngram)
    torch.nn.Module.__init__(layer)
    # Supply an already projected tensor so this test exercises gate dispatch
    # and arithmetic without loading a checkpoint or allocating a GEMM weight.
    layer.wkv = torch.nn.Identity()
    layer.hc_mult, layer.dim, layer.eps = 4, 96, 1e-6
    layer.q_weight = torch.nn.Parameter(q, requires_grad=False)
    layer.k_weight = torch.nn.Parameter(k, requires_grad=False)
    profile = get_hardware_profile(getattr(AscendDeviceType, hardware))
    monkeypatch.setattr(layer_module, "get_current_hardware_profile", lambda: profile)
    calls = []
    for name in ("fused_engram_gate", "fused_engram_gate_a3", "engram_gate"):
        function = getattr(layer_module, name)

        def record_call(*args, _name=name, _function=function, **kwargs):
            calls.append(_name)
            return _function(*args, **kwargs)

        monkeypatch.setattr(layer_module, name, record_call)
    actual = layer(hidden, kv, mask, rotation if rotated else None)
    original = _inverse_rotate(hidden, rotation) if rotated else hidden.float()
    expected = _reference(*(x.cpu() for x in (hidden, kv, q, k, mask, original)))
    assert calls == [expected_call]
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(INTERPRET, reason="Device timing requires NPU execution")
@torch.inference_mode()
def test_engram_gate_profile_shape_latency():
    """Time post-wkv ACL replays so Python launch gaps do not skew decode."""
    hidden, kv, q, k, mask, rotation = _inputs(132, 5120, torch.bfloat16, False, "npu")

    def unfused():
        return _reference(hidden, kv, q, k, mask, _inverse_rotate(hidden, rotation))

    def fused():
        return fused_engram_gate_a3(hidden, kv, q, k, rotation, mask, 1e-6)

    timings = {}
    for name, operation in (
        ("unfused_post_wkv", unfused),
        ("fused_post_wkv", fused),
    ):
        for _ in range(10):
            operation()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            _output = operation()
        for _ in range(10):
            graph.replay()
        samples = []
        for _ in range(5):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(30):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 30)
        timings[name] = statistics.median(samples)
        print(f"{name}_acl_replay: median={timings[name]:.3f} us samples={samples}")
    assert timings["fused_post_wkv"] < timings["unfused_post_wkv"]
