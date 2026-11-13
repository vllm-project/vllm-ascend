# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.ops import gdn
from vllm_ascend.ops.layernorm import AscendRMSNormGated
from vllm_ascend.ops.triton.fla.gdn_gated_norm import gdn_gated_norm


@pytest.fixture(autouse=True)
def norm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.mark.parametrize("tokens", [1, 8, 17, 124])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", ["swish", "sigmoid"])
@pytest.mark.parametrize("before", [False, True])
@torch.inference_mode()
def test_strided_z_matches_original_norm_and_cpu(tokens, dtype, activation, before):
    torch.manual_seed(42)
    backing = torch.randn(tokens + 2, 128, 128, device="npu", dtype=dtype)
    z = backing[1 : tokens + 1, 80:128]
    xb = torch.randn(tokens + 2, 50, 128, device="npu", dtype=dtype)
    x = xb[1 : tokens + 1, 1:49]
    norm = AscendRMSNormGated(128, norm_before_gate=before, activation=activation, device="npu", dtype=dtype)
    norm.weight.copy_(torch.randn_like(norm.weight) * 0.2 + 1)
    with patch.object(gdn, "gdn_gated_norm", wraps=gdn_gated_norm) as fast:
        actual = gdn._gdn_output_norm(norm, x, z)
        assert fast.call_count == (0 if z.is_contiguous() else 1)
    original = norm(x.reshape(-1, 128), z.reshape(-1, 128)).reshape(x.shape)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    xf, zf = x.cpu().float(), z.cpu().float()
    gate = zf.sigmoid()
    if activation == "swish":
        gate *= zf
    if not before:
        xf *= gate
    expected = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    expected *= norm.weight.cpu().float()
    if before:
        expected *= gate
    tol = 0.02 if dtype == torch.bfloat16 else (0.002 if dtype == torch.float16 else 1e-5)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("fallback", ["group", "width", "contiguous", "subclass", "feature_stride", "grad"])
def test_unsupported_inputs_use_existing_norm(fallback):
    class CustomNorm(AscendRMSNormGated):
        pass

    width = 64 if fallback == "width" else 128
    cls = CustomNorm if fallback == "subclass" else AscendRMSNormGated
    norm = cls(width, group_size=64 if fallback == "group" else None, device="npu", dtype=torch.bfloat16)
    z = torch.randn(8, 96, width, device="npu", dtype=torch.bfloat16)[:, :48]
    if fallback == "contiguous":
        z = z.contiguous()
    if fallback == "feature_stride":
        z = torch.randn(8, 48, width * 2, device="npu", dtype=torch.bfloat16)[..., ::2]
    x = torch.randn_like(z)
    with torch.set_grad_enabled(fallback == "grad"):
        with patch.object(gdn, "gdn_gated_norm", side_effect=AssertionError("must use existing norm")):
            actual = gdn._gdn_output_norm(norm, x, z)
        expected = norm(x.reshape(-1, width), z.reshape(-1, width)).reshape(z.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_gdn_norm_graph_replay_reads_updated_z():
    norm = AscendRMSNormGated(128, device="npu", dtype=torch.bfloat16)
    x = torch.randn(8, 48, 128, device="npu", dtype=torch.bfloat16)
    z = torch.randn(8, 128, 128, device="npu", dtype=torch.bfloat16)[:, 80:]
    for _ in range(3):
        gdn._gdn_output_norm(norm, x, z)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = gdn._gdn_output_norm(norm, x, z)
    for _ in range(3):
        z.copy_(torch.randn_like(z))
        expected = norm(x.reshape(-1, 128), z.reshape(-1, 128)).reshape(x.shape)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
