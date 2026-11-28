# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope import triton_split_qkv_rmsnorm_mrope
from vllm_ascend.ops.triton.qwen_prefill import pack_vision_qk, split_packed_qkvz, vision_add_norm
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize("tokens,heads", [(1, 16), (11, 4), (20048, 16), (37836, 16)])
def test_pack_vision_qk(tokens, heads):
    qkv = torch.randn(tokens, 3, heads, 72, device="npu", dtype=torch.bfloat16)
    packed = pack_vision_qk(qkv)
    assert packed.is_contiguous()
    assert torch.equal(packed[0], qkv[:, 0])
    assert torch.equal(packed[1], qkv[:, 1])


def test_gdn_split():
    packed = torch.randn(11, 12288, device="npu", dtype=torch.bfloat16)
    qkv, z = split_packed_qkvz(packed, 8192)
    assert qkv.is_contiguous() and z.is_contiguous()
    assert torch.equal(qkv, packed[:, :8192])
    assert torch.equal(z, packed[:, 8192:])


@pytest.mark.parametrize("tokens", [11, 20048, 37836])
@torch.inference_mode()
def test_native_vision_add_norm(tokens):
    params = dict(device="npu", dtype=torch.bfloat16)
    x, attn = torch.randn(2, tokens, 1, 1152, **params).unbind()
    norm = torch.nn.LayerNorm(1152, eps=1e-6, **params)
    norm.weight.uniform_(0.75, 1.25)
    norm.bias.uniform_(-0.25, 0.25)
    residual, normalized = vision_add_norm(x, attn, norm)
    assert torch.equal(residual, x + attn)
    torch.testing.assert_close(normalized, norm(x + attn), atol=0.03125, rtol=0.01)


@pytest.mark.parametrize("broadcast", [True, False])
def test_mrope_position_strides(broadcast):
    init_device_properties_triton()
    params = dict(device="npu", dtype=torch.bfloat16)
    cache = torch.randn(31, 64, **params)
    pos = torch.randint(31, (3, 22), device="npu")[:, ::2]
    if broadcast:
        pos = pos[:1].expand(3, -1)
    qkv = torch.randn(11, 9216, **params)
    q, k = torch.randn(2, 256, **params).unbind()
    args = dict(
        num_q_heads=16,
        num_kv_heads=2,
        head_size=256,
        eps=1e-6,
        mrope_section=[12, 10, 10],
        is_interleaved=True,
        rope_dim=64,
        has_gate=True,
    )
    ref = triton_split_qkv_rmsnorm_mrope(qkv, q + 1, k + 1, cache[pos], **args)
    out = triton_split_qkv_rmsnorm_mrope(qkv, q, k, cache, positions=pos, unit_offset=True, **args)
    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
