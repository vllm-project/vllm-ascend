# SPDX-License-Identifier: Apache-2.0
"""NPU comparison of fused DSpark context KV against the eager operator chain."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.dspark_context_kv import can_fuse_dspark_context_kv, fused_dspark_context_kv


@pytest.mark.parametrize("num_tokens", [1, 16, 256, 2048])
@pytest.mark.parametrize("paired_slots", [False, True])
def test_dspark_context_kv_matches_per_layer_reference(num_tokens, paired_slots):
    torch.manual_seed(4)
    num_layers, head_dim, nope_dim, block_size = 3, 512, 448, 128
    num_blocks = (num_tokens + block_size - 1) // block_size
    kv = torch.randn(num_tokens, num_layers * head_dim, device="npu", dtype=torch.bfloat16).view(
        num_tokens, num_layers, head_dim
    )
    # Slicing a layer-outermost cache gives a non-contiguous physical block stride.
    fused_storage = torch.full(
        (num_blocks, num_layers, block_size, 1, head_dim), -7, device="npu", dtype=torch.bfloat16
    )
    reference_storage = fused_storage.clone()
    layers, mappings, rope = [], [], ({}, {})
    for layer_idx in range(num_layers):
        weight = torch.randn(head_dim, device="npu", dtype=torch.bfloat16)
        angles = torch.randn(num_tokens, 32, device="npu", dtype=torch.float32)
        cos = angles.cos().repeat_interleave(2, -1)[:, None, None, :]
        sin = angles.sin().repeat_interleave(2, -1)[:, None, None, :]
        name = f"draft-{layer_idx}"
        rope[0][name], rope[1][name] = cos, sin
        cache = fused_storage[:, layer_idx]
        attn = SimpleNamespace(
            packed_cache_ops=None,
            nope_head_dim=nope_dim,
            rope_head_dim=head_dim - nope_dim,
            rotary_emb=SimpleNamespace(layername=name),
            kv_norm=SimpleNamespace(weight=weight, variance_epsilon=1e-6),
            dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=[cache], block_size=block_size)),
        )
        layers.append(SimpleNamespace(self_attn=attn))
        physical = torch.arange(num_tokens, device="npu", dtype=torch.int32)
        physical[::7] = -1
        if paired_slots:
            mapping = torch.stack((physical.clamp_min(0) // block_size, physical.clamp_min(0) % block_size), -1)
            mapping[physical < 0] = -1
        else:
            mapping = physical
        mappings.append(mapping)

        row = kv[:, layer_idx].float()
        normalized = (row * torch.rsqrt(row.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).to(kv.dtype)
        rotated = torch_npu.npu_rotary_mul(
            normalized[:, None, None, nope_dim:], cos, sin, rotary_mode="interleave"
        ).view(num_tokens, head_dim - nope_dim)
        expected = torch.cat((normalized[:, :nope_dim], rotated), -1)
        for token, slot in enumerate(physical.cpu().tolist()):
            if slot >= 0:
                reference_storage[slot // block_size, layer_idx, slot % block_size, 0] = expected[token]

    assert can_fuse_dspark_context_kv(kv, layers, mappings, rope)
    fused_dspark_context_kv(kv, layers, mappings, rope, list(range(num_layers)))
    torch.testing.assert_close(fused_storage, reference_storage, rtol=5e-3, atol=5e-2)
