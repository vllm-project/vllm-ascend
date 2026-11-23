# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate parity, including distinct scoring/residual bases and strided inputs."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v41.engram.common import restore_engram_hidden
from vllm_ascend.models.deepseek_v41.engram.layer import AscendEngram
from vllm_ascend.ops.triton import engram_gate as gate_ops
from vllm_ascend.ops.triton.engram_gate import fused_engram_gate

HC = 4
ROTATION_BLOCK = 32
EPS = 1e-6


@pytest.mark.parametrize("device_type", [AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [4096, 4128, 5120])
@pytest.mark.parametrize("rotated", [False, True])
def test_gate_launch_policy(monkeypatch, device_type, dtype, dim, rotated):
    # Verify wrapper dispatch without compiling or executing an NPU kernel.
    kernel = MagicMock()
    monkeypatch.setattr(gate_ops, "_engram_gate_kernel", kernel)
    monkeypatch.setattr(gate_ops, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    hidden = torch.empty(1, HC, dim, dtype=dtype)
    kv = torch.empty(1, (HC + 1) * dim, dtype=dtype)
    weight = torch.empty(HC, dim, dtype=dtype)
    mask = torch.ones(1, dtype=torch.bool)
    score = torch.empty_like(hidden, dtype=torch.float32) if rotated else None
    fused_engram_gate(hidden, kv, weight, weight, mask, EPS, gate_hidden=score)
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    expected_tiled = device_type == AscendDeviceType.A5 and dtype == torch.float32 and dim > 4096
    assert launch.call_args.kwargs["TILED"] is expected_tiled
    expected_block = 1024 if expected_tiled else 1 << (dim - 1).bit_length()
    assert launch.call_args.kwargs["BLOCK"] == expected_block


def reference(hidden, kv, qw, kw, mask, rotation):
    # Independently express the reference; residual must stay in its original basis.
    dim = hidden.shape[-1]
    h = hidden.float()
    if rotation is not None:
        blocks = h.reshape(*h.shape[:-1], dim // ROTATION_BLOCK, ROTATION_BLOCK)
        if rotation.ndim == 2:
            h = (blocks @ rotation.T).reshape_as(h)
        else:
            h = (blocks.unsqueeze(-2) @ rotation.transpose(-1, -2)).squeeze(-2).reshape_as(h)
    key = kv[:, : HC * dim].reshape(hidden.shape).float()
    value = kv[:, HC * dim :].float()
    scale = torch.rsqrt(h.square().mean(-1) + EPS) * torch.rsqrt(key.square().mean(-1) + EPS)
    dot = (h * (qw.float() * kw.float()) * key).sum(-1) * scale * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    gate = gate.masked_fill(~mask[:, None], 0)
    return (hidden.float() + gate[..., None] * value[:, None, :]).to(hidden.dtype)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Engram gate routes on A3 (rotated) and A5 (unrotated)",
)
@pytest.mark.parametrize("tokens,dim", [(0, 5120), (1, 96), (7, 4096), (7, 4128), (7, 5120), (132, 5120)])
@pytest.mark.parametrize("rotation_kind", ["none", "shared", "per_block"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_engram_gate(tokens, dim, rotation_kind, dtype):
    torch.manual_seed(42)
    # Token and branch strides are deliberately non-contiguous, channels are contiguous.
    hidden = torch.randn(tokens * 2, HC * 2, dim, device="npu", dtype=dtype)[::2, ::2]
    kv = torch.randn(tokens * 2, (HC + 1) * dim, device="npu", dtype=dtype)[::2]
    qw = torch.randn(HC, dim * 2, device="npu", dtype=dtype)[:, ::2]
    kw = torch.randn_like(qw)
    mask = (torch.arange(tokens * 2, device="npu") % 3 != 0)[::2]
    rotation = None
    if rotation_kind != "none":
        shape: tuple[int, ...] = (ROTATION_BLOCK, ROTATION_BLOCK)
        if rotation_kind == "per_block":
            shape = (dim // ROTATION_BLOCK, *shape)
        rotation = torch.randn(shape, device="npu", dtype=torch.float32) / ROTATION_BLOCK**0.5
    gate_hidden = None if rotation is None else restore_engram_hidden(hidden, rotation)
    expected = reference(hidden, kv, qw, kw, mask, rotation)
    actual = fused_engram_gate(hidden, kv, qw, kw, mask, EPS, gate_hidden=gate_hidden)
    torch.npu.synchronize()
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual[~mask], hidden[~mask], atol=0, rtol=0)


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Rotated model gate routes on A3")
def test_rotated_engram_model_graph_replay():
    torch.manual_seed(7)
    tokens, dim = 7, 5120
    hidden = torch.randn(tokens, HC, dim, device="npu", dtype=torch.bfloat16)
    kv = torch.randn(tokens, (HC + 1) * dim, device="npu", dtype=torch.bfloat16)
    mask = torch.ones(tokens, device="npu", dtype=torch.bool)
    rotation = torch.randn(ROTATION_BLOCK, ROTATION_BLOCK, device="npu") / ROTATION_BLOCK**0.5
    # Exercise real model dispatch without constructing a checkpoint linear layer.
    layer = AscendEngram.__new__(AscendEngram)
    torch.nn.Module.__init__(layer)
    layer.dim, layer.hc_mult, layer.eps = dim, HC, EPS
    layer.wkv = torch.nn.Identity()
    layer.q_weight = torch.nn.Parameter(torch.randn(HC, dim, device="npu", dtype=torch.bfloat16))
    layer.k_weight = torch.nn.Parameter(torch.randn(HC, dim, device="npu", dtype=torch.bfloat16))
    with torch.inference_mode():
        for _ in range(3):
            layer(hidden, kv, mask, rotation)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = layer(hidden, kv, mask, rotation)
        for scale in (0.0, -0.5, 2.0):
            hidden.normal_().mul_(scale)
            kv.normal_()
            mask.copy_(torch.arange(tokens, device="npu") % 2 == 0)
            graph.replay()
            torch.npu.synchronize()
            expected = reference(hidden, kv, layer.q_weight, layer.k_weight, mask, rotation)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(actual[~mask], hidden[~mask], atol=0, rtol=0)
