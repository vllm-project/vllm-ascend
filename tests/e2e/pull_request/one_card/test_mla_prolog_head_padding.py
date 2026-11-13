# SPDX-License-Identifier: Apache-2.0

"""NPU regression for native MLA Q_B head padding before NZ conversion."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.mla_v1 import AscendMLAImpl

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend NPU")


@pytest.mark.parametrize("heads", [4, 6, 8, 12])
def test_native_padded_q_projection_matches_unfused(heads):
    prior_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    try:
        generator = torch.Generator().manual_seed(197 + heads)
        dtype = torch.bfloat16

        def random(*shape, scale=1.0):
            return (torch.randn(*shape, generator=generator) * scale).to(dtype).npu()

        x = random(3, 7168)
        w_a = random(2112, 7168, scale=0.02)
        w_b = random(heads * 192, 1536, scale=0.02)
        w_uk = random(heads, 128, 512, scale=0.02)
        gamma_q = random(1536, scale=0.1) + 1
        gamma_k = random(512, scale=0.1) + 1
        padded = 1 << (heads - 1).bit_length()
        impl = object.__new__(AscendMLAImpl)
        impl.fused_qkv_a_proj = SimpleNamespace(weight=w_a)
        impl.q_proj = SimpleNamespace(weight=w_b)
        impl.q_lora_rank = 1536
        impl.q_a_layernorm = SimpleNamespace(weight=gamma_q)
        impl.kv_a_layernorm = SimpleNamespace(weight=gamma_k)
        impl.mlapo_W_UK_T = w_uk
        impl.num_heads = heads
        impl.num_heads_padded = padded
        impl.head_padding = padded - heads
        impl.qk_head_dim = 192
        weights = impl._build_prolog_v3_weights()
        angles = random(3, 32).float()
        cos = angles.cos().repeat(1, 2).to(dtype)
        sin = angles.sin().repeat(1, 2).to(dtype)
        cache_k = torch.zeros((1, 128, 1, 512), dtype=dtype, device="npu")
        cache_r = torch.zeros((1, 128, 1, 64), dtype=dtype, device="npu")
        output = torch_npu.npu_mla_prolog_v3(
            token_x=x,
            weight_dq=weights["weight_dq"],
            weight_uq_qr=weights["weight_uq_qr"],
            weight_uk=weights["weight_uk"],
            weight_dkv_kr=weights["weight_dkv_kr"],
            rmsnorm_gamma_cq=gamma_q,
            rmsnorm_gamma_ckv=gamma_k,
            rope_sin=sin,
            rope_cos=cos,
            kv_cache=cache_k,
            kr_cache=cache_r,
            cache_index=torch.arange(3, device="npu", dtype=torch.int64),
            rmsnorm_epsilon_cq=1e-5,
            rmsnorm_epsilon_ckv=1e-5,
            cache_mode="PA_BSND",
            query_norm_flag=False,
            query_quant_mode=0,
            kv_cache_quant_mode=0,
            ckvkr_repo_mode=0,
            quant_scale_repo_mode=0,
            tile_size=128,
            weight_quant_mode=0,
        )
        # Independent CPU projection math, rounding each BF16 intermediate.
        projected = torch.nn.functional.linear(x.cpu().float(), w_a.cpu().float()).to(dtype)
        q_a = projected[:, :1536].float()
        q_a = (q_a * torch.rsqrt(q_a.square().mean(-1, keepdim=True) + 1e-5) * gamma_q.cpu().float()).to(dtype)
        q_b = torch.nn.functional.linear(q_a.float(), w_b.cpu().float()).to(dtype)
        q_b = q_b.reshape(3, heads, 192)
        expected_nope = torch.bmm(q_b[..., :128].float().transpose(0, 1), w_uk.cpu().float()).to(dtype).transpose(0, 1)
        # Use a separate rotary operator; this path has no packed Q_B weight.
        expected_rope = torch_npu.npu_interleave_rope(
            q_b[..., 128:].contiguous().npu().reshape(3, heads, 1, 64), cos[:, None, None, :], sin[:, None, None, :]
        ).reshape(3, heads, 64)
        actual_nope = output[0].reshape(3, padded, 512)[:, :heads]
        actual_rope = output[1].reshape(3, padded, 64)[:, :heads]
        torch.npu.synchronize()
        torch.testing.assert_close(actual_nope.cpu().float(), expected_nope.float(), rtol=0.02, atol=0.01)
        torch.testing.assert_close(actual_rope.cpu().float(), expected_rope.cpu().float(), rtol=0.02, atol=0.02)
    finally:
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": (prior_internal_format or b"disable").decode()})
