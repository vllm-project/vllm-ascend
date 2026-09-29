# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""BF16 Kimi K3 MLA prolog accuracy through the CANN operator."""

import pytest
import torch
import torch_npu


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    values = x.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5) * weight.float()).to(x.dtype)


@torch.inference_mode()
def test_kimi_k3_mla_prolog_bf16_decode_accuracy() -> None:
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(0):
        pytest.skip("requires an Ascend 950 NPU")
    mla_prolog = pytest.importorskip("cann_ops_transformer").mla_prolog

    torch.manual_seed(952)
    tokens, hidden, q_rank, kv_rank, heads, nope_dim, rope_dim = 2, 7168, 1536, 512, 96, 128, 64

    def sample(*shape: int) -> torch.Tensor:
        return (torch.randn(shape, dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)

    x = sample(tokens, hidden)
    dq = sample(hidden, q_rank)
    uq_qr = sample(q_rank, heads * (nope_dim + rope_dim))
    uk = sample(heads, nope_dim, kv_rank)
    dkv_kr = sample(hidden, kv_rank + rope_dim)
    gamma_q = torch.ones(q_rank, dtype=torch.bfloat16, device="npu")
    gamma_kv = torch.ones(kv_rank, dtype=torch.bfloat16, device="npu")
    kv_cache = torch.zeros((2, 128, 1, kv_rank), dtype=torch.bfloat16, device="npu")
    kr_cache = torch.zeros((2, 128, 1, rope_dim), dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([1, 130], dtype=torch.int64, device="npu")

    query, query_rope, *_ = mla_prolog(
        token_x=x,
        weight_dq=torch_npu.npu_format_cast(dq.contiguous(), 29),
        weight_uq_qr=torch_npu.npu_format_cast(uq_qr.contiguous(), 29),
        weight_uk=uk,
        weight_dkv_kr=torch_npu.npu_format_cast(dkv_kr.contiguous(), 29),
        rmsnorm_gamma_cq=gamma_q,
        rmsnorm_gamma_ckv=gamma_kv,
        rope_sin=None,
        rope_cos=None,
        kv_cache=kv_cache,
        kr_cache=kr_cache,
        cache_index=slots,
        cache_mode="PA_BSND",
        weight_quant_mode=0,
        kv_cache_quant_mode=0,
        query_quant_mode=0,
    )

    q_norm = _rms_norm(x @ dq, gamma_q)
    q_up = (q_norm @ uq_qr).view(tokens, heads, nope_dim + rope_dim)
    expected_query = torch.einsum("thd,hdk->thk", q_up[..., :nope_dim].float(), uk.float()).to(torch.bfloat16)
    kv_up = x @ dkv_kr
    expected_kv = _rms_norm(kv_up[..., :kv_rank], gamma_kv)

    torch.testing.assert_close(query, expected_query, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(query_rope, q_up[..., nope_dim:], rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(kv_cache.flatten(0, 1)[slots, 0], expected_kv, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(kr_cache.flatten(0, 1)[slots, 0], kv_up[..., kv_rank:], rtol=5e-2, atol=5e-2)
