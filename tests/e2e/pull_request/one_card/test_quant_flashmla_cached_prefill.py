# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.mla_v1 import AscendMLAImpl


def test_quant_flashmla_prefill_reads_fp8_history_across_pages(monkeypatch):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("Requires Ascend 950 for QuantFlashMLA FP8 cache")
    device = "npu:0"
    heads, latent_dim, rope_dim, head_dim = 8, 512, 64, 128
    width = latent_dim + rope_dim
    block_size = 128
    generator = torch.Generator().manual_seed(0)
    fp8 = (torch.randn(3, block_size, 1, width, generator=generator) * 0.25).to(torch.float8_e4m3fn)
    # Transfer the stored bytes to the device, including the token-fused strides.
    fused = fp8.view(torch.int8).to(device).view(fp8.dtype)
    cache = fused[..., :latent_dim], fused[..., latent_dim:]
    table = torch.tensor([[2, 0]], device=device, dtype=torch.int32)
    lengths = torch.tensor([4], device=device, dtype=torch.int32)
    starts = torch.tensor([126], device=device, dtype=torch.int32)
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            chunked_context=SimpleNamespace(seq_tot=[4], starts=[starts], chunk_actual_seq_lengths_kv_list=[[4]]),
            actual_seq_lengths_q=[2],
        )
    )

    def project(latent):
        return (latent[:, : 2 * head_dim].repeat(1, heads),)

    impl = SimpleNamespace(
        num_heads=heads,
        head_padding=0,
        qk_nope_head_dim=head_dim,
        qk_rope_head_dim=rope_dim,
        v_head_dim=head_dim,
        scale=1.0 / (head_dim + rope_dim) ** 0.5,
        use_quant_flashmla=True,
        fa_quant_layer=True,
        support_fp8_attention=True,
        fak_descale_float=torch.tensor([0.5], device=device),
        get_context_block_table=lambda _: table,
        get_context_seq_len_npu=lambda *_: lengths,
        _reorg_kvcache=lambda k, v, **_: (k, v),
        kv_b_proj=project,
    )
    q_nope = torch.randn(2, heads, head_dim, generator=generator).bfloat16().to(device)
    q_pe = torch.randn(2, heads, rope_dim, generator=generator).bfloat16().to(device)
    prefix = torch.zeros_like(q_nope)
    prefix_lse = torch.zeros(heads, 2, dtype=torch.float32, device=device)

    def prefill():
        output, _ = AscendMLAImpl._compute_prefill_context(
            impl, q_nope, q_pe, cache, rope_dim, metadata, prefix, prefix_lse
        )
        return output.cpu()

    actual = prefill()
    raw = fp8.view(torch.int8)
    expected_rows = torch.cat((raw[2, 126:, 0], raw[0, :2, 0])).to(device)
    monkeypatch.setattr(mla_v1, "load_mla_cache_context_rows", lambda *_: expected_rows)
    torch.testing.assert_close(actual, prefill(), rtol=0, atol=0)
