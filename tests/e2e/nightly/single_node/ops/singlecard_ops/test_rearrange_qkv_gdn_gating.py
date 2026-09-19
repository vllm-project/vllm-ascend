# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend.ops import gdn as gdn_module
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

GDN_CONFIGS = ((1024, 1024, 3072), (128, 128, 256), (64, 64, 128))


class LayerStub:
    rearrange_mixed_qkv = AscendGatedDeltaNetAttention.rearrange_mixed_qkv
    _view_packed_qkv = AscendGatedDeltaNetAttention._view_packed_qkv


@pytest.fixture(scope="module", autouse=True)
def load_op():
    enable_custom_op()
    if get_ascend_device_type() not in {AscendDeviceType.A2, AscendDeviceType.A3}:
        pytest.skip("npu_rearrange_qkv_and_gdn_gating is only built for A2 and A3")


def reference_packed_qkv(mixed_qkv, dims):
    return torch.cat([part.reshape(-1) for part in mixed_qkv.split(dims, dim=-1)])


@pytest.mark.parametrize(("q_dim", "k_dim", "v_dim"), GDN_CONFIGS)
@pytest.mark.parametrize("tokens", [0, 1, 3, 17, 129, 1024])
@pytest.mark.parametrize("head_dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_matches_references(q_dim, k_dim, v_dim, tokens, head_dtype):
    num_heads = v_dim // 128
    mixed_qkv = torch.randn(tokens, q_dim + k_dim + v_dim, dtype=torch.bfloat16, device="npu")
    a = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device="npu")
    b = torch.randn_like(a)
    A_log = torch.randn(num_heads, dtype=head_dtype, device="npu")
    dt_bias = torch.randn_like(A_log)

    packed_qkv, g, beta = torch.ops._C_ascend.npu_rearrange_qkv_and_gdn_gating(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        q_dim,
        k_dim,
        v_dim,
    )

    expected_qkv = reference_packed_qkv(mixed_qkv, (q_dim, k_dim, v_dim))
    expected_g, expected_beta = fused_gdn_gating_pytorch(A_log=A_log, a=a, b=b, dt_bias=dt_bias)
    assert torch.equal(packed_qkv.view(torch.int16).cpu(), expected_qkv.view(torch.int16).cpu())
    torch.testing.assert_close(g.cpu(), expected_g.squeeze(0).cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        beta.float().cpu(),
        expected_beta.squeeze(0).float().cpu(),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("tokens", [17, 321, 4096])
@torch.inference_mode()
def test_matches_separate_ops(tokens):
    q_dim = k_dim = 1024
    v_dim = 3072
    num_heads = v_dim // 128
    mixed_qkv = torch.randn(tokens, q_dim + k_dim + v_dim, dtype=torch.bfloat16, device="npu")
    a = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device="npu")
    b = torch.randn_like(a)
    A_log = torch.randn(num_heads, dtype=torch.bfloat16, device="npu")
    dt_bias = torch.randn_like(A_log)

    packed_qkv, g, beta = torch.ops._C_ascend.npu_rearrange_qkv_and_gdn_gating(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        q_dim,
        k_dim,
        v_dim,
    )
    expected_qkv = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, q_dim, k_dim, v_dim)
    expected_g, expected_beta = fused_gdn_gating_patch(A_log=A_log, a=a, b=b, dt_bias=dt_bias)

    assert torch.equal(packed_qkv.view(torch.int16).cpu(), expected_qkv.view(torch.int16).cpu())
    torch.testing.assert_close(g.cpu(), expected_g.squeeze(0).cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        beta.float().cpu(),
        expected_beta.squeeze(0).float().cpu(),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("tokens", [1, 17, 321])
@torch.inference_mode()
def test_nondefault_stream_and_graph_replay(tokens):
    dims = (1024, 1024, 3072)
    num_heads = 24
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        mixed_qkv = torch.randn(tokens, sum(dims), dtype=torch.bfloat16, device="npu")
        a = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device="npu")
        b = torch.randn_like(a)
        A_log = torch.randn(num_heads, dtype=torch.float32, device="npu")
        dt_bias = torch.randn_like(A_log)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            captured = torch.ops._C_ascend.npu_rearrange_qkv_and_gdn_gating(
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                *dims,
            )
        for _ in range(3):
            mixed_qkv.normal_()
            a.normal_()
            b.normal_()
            graph.replay()
            stream.synchronize()
            expected = reference_packed_qkv(mixed_qkv.cpu(), dims)
            assert torch.equal(captured[0].view(torch.int16).cpu(), expected.view(torch.int16))
            expected_g, expected_beta = fused_gdn_gating_pytorch(A_log=A_log, a=a, b=b, dt_bias=dt_bias)
            torch.testing.assert_close(captured[1].cpu(), expected_g.squeeze(0).cpu(), rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(
                captured[2].float().cpu(),
                expected_beta.squeeze(0).float().cpu(),
                rtol=1e-2,
                atol=1e-2,
            )


@torch.inference_mode()
def test_gdn_dispatch_uses_device_adaptor():
    tokens = 37
    q_dim = k_dim = 1024
    v_dim = 3072
    num_heads = 24
    mixed_qkv = torch.randn(tokens, q_dim + k_dim + v_dim, dtype=torch.bfloat16, device="npu")
    a = torch.randn(tokens, num_heads, dtype=torch.bfloat16, device="npu")
    b = torch.randn_like(a)
    A_log = torch.randn(num_heads, dtype=torch.float32, device="npu")
    dt_bias = torch.randn_like(A_log)
    layer = LayerStub()
    layer.key_dim = q_dim
    layer.value_dim = v_dim
    layer.tp_size = 1
    layer.head_k_dim = 128
    layer.head_v_dim = 128
    with patch.object(gdn_module, "_ORIGINAL_REARRANGE_MIXED_QKV", side_effect=AssertionError("fallback")):
        outputs = AscendGatedDeltaNetAttention.rearrange_mixed_qkv_and_fused_gdn_gating(
            layer,
            mixed_qkv,
            A_log,
            a,
            b,
            dt_bias,
        )
    assert [output.shape for output in outputs] == [
        (1, tokens, 8, 128),
        (1, tokens, 8, 128),
        (1, tokens, 24, 128),
        (1, tokens, 24),
        (1, tokens, 24),
    ]


@pytest.mark.parametrize("kind", ["x_dtype", "head_shape", "head_dtype", "stride"])
def test_reject_unsafe_inputs(kind):
    mixed_qkv = torch.empty(2, 5120, dtype=torch.bfloat16, device="npu")
    a = torch.empty(2, 24, dtype=torch.bfloat16, device="npu")
    b = torch.empty_like(a)
    A_log = torch.empty(24, dtype=torch.float32, device="npu")
    dt_bias = torch.empty_like(A_log)

    if kind == "x_dtype":
        mixed_qkv = mixed_qkv.to(torch.float16)
    elif kind == "head_shape":
        b = torch.empty(2, 16, dtype=torch.bfloat16, device="npu")
    elif kind == "head_dtype":
        dt_bias = dt_bias.to(torch.float16)
    else:
        mixed_qkv = torch.empty(2, 10240, dtype=torch.bfloat16, device="npu")[:, ::2]

    with pytest.raises(RuntimeError, match="npu_rearrange_qkv_and_gdn_gating requires"):
        torch.ops._C_ascend.npu_rearrange_qkv_and_gdn_gating(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            1024,
            1024,
            3072,
        )


def test_meta():
    tokens = 17
    mixed_qkv = torch.empty(tokens, 5120, dtype=torch.bfloat16, device="meta")
    a = torch.empty(tokens, 24, dtype=torch.bfloat16, device="meta")
    b = torch.empty_like(a)
    A_log = torch.empty(24, dtype=torch.float32, device="meta")
    dt_bias = torch.empty_like(A_log)
    packed_qkv, g, beta = torch.ops._C_ascend.npu_rearrange_qkv_and_gdn_gating(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        1024,
        1024,
        3072,
    )
    assert packed_qkv.shape == (tokens * 5120,)
    assert g.shape == (tokens, 24) and g.dtype == torch.float32
    assert beta.shape == (tokens, 24) and beta.dtype == torch.bfloat16
