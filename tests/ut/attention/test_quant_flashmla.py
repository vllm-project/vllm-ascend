# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.ops  # noqa: F401  # Register ops before importing attention/device modules.
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.flashmla import FlashMLAMetadataBuilder, QuantFlashMLAConfig, QuantFlashMLAImpl
from vllm_ascend.attention.mla_v1 import AscendMLAImpl


def _inputs(tokens: int = 2, heads: int = 8):
    q = torch.empty((tokens, heads, 576), dtype=torch.float8_e4m3fn)
    cache = torch.empty((2, 128, 1, 576), dtype=torch.float8_e4m3fn)
    q_descale = torch.ones((tokens, heads), dtype=torch.float32)
    k_descale = torch.ones(1, dtype=torch.float32)
    block_table = torch.zeros((tokens, 1), dtype=torch.int32)
    cache_seqlens = torch.ones(tokens, dtype=torch.int32)
    cu_seqlens_q = torch.arange(tokens + 1, dtype=torch.int32)
    seqused_q = torch.ones(tokens, dtype=torch.int32)
    metadata = torch.zeros((2, 64), dtype=torch.int32)
    return (
        q,
        cache,
        q_descale,
        k_descale,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        seqused_q,
        metadata,
    )


def test_quant_flashmla_uses_delivered_metadata_and_attention_abi():
    metadata_op = MagicMock(return_value=torch.zeros((2, 64), dtype=torch.int32))
    attention_op = MagicMock(
        return_value=(
            torch.empty((8, 2, 512), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.float32),
        )
    )
    adapter = QuantFlashMLAImpl(QuantFlashMLAConfig(8, 0.125, mask_mode=0), attention_op, metadata_op)
    q, cache, q_scale, k_scale, table, lengths, cu, used, metadata = _inputs()

    assert adapter.build_metadata(lengths, cu, used) is metadata_op.return_value
    metadata_op.assert_called_once_with(
        lengths,
        8,
        1,
        QuantFlashMLAConfig.FP8_QUANT_MODE,
        cu_seqlens_q=cu,
        seqused_q=used,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        head_dim_qk=576,
        head_dim_v=512,
        mask_mode=0,
        layout_q="TND",
    )

    output, lse = adapter.attention(
        q,
        cache,
        q_descale=q_scale,
        k_descale=k_scale,
        block_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        seqused_q=used,
        metadata=metadata,
    )
    assert output is attention_op.return_value[0]
    assert lse is attention_op.return_value[1]
    attention_op.assert_called_once_with(
        q,
        cache,
        q_scale,
        k_scale,
        table,
        lengths,
        QuantFlashMLAConfig.FP8_QUANT_MODE,
        cu_seqlens_q=cu,
        seqused_q=used,
        attn_mask=None,
        metadata=metadata,
        head_dim_v=512,
        softmax_scale=0.125,
        mask_mode=0,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_out="NTD",
        return_softmax_lse=False,
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("q_descale", torch.ones((2, 8), dtype=torch.bfloat16), "q_descale"),
        ("k_descale", torch.ones(2, dtype=torch.float32), "k_descale"),
        ("metadata", torch.zeros(64, dtype=torch.int32), "metadata"),
    ],
)
def test_quant_flashmla_rejects_invalid_scale_or_metadata_contract(field, replacement, message):
    adapter = QuantFlashMLAImpl(
        QuantFlashMLAConfig(8, 0.125, mask_mode=0),
        MagicMock(),
        MagicMock(),
    )
    q, cache, q_scale, k_scale, table, lengths, cu, used, metadata = _inputs()
    kwargs = {
        "q_descale": q_scale,
        "k_descale": k_scale,
        "block_table": table,
        "cache_seqlens": lengths,
        "cu_seqlens_q": cu,
        "seqused_q": used,
        "metadata": metadata,
    }
    kwargs[field] = replacement
    with pytest.raises(ValueError, match=message):
        adapter.attention(q, cache, **kwargs)


def test_quant_flashmla_external_forward_passes_descales(monkeypatch):
    q_nope = torch.zeros((2, 8, 512), dtype=torch.float8_e4m3fn)
    q_pe = torch.zeros((2, 8, 64), dtype=torch.bfloat16)
    q_descale = torch.arange(16, dtype=torch.float32).view(2, 8)
    k_descale = torch.tensor([0.25], dtype=torch.float32)
    query = torch.empty((2, 8, 576), dtype=torch.float8_e4m3fn)
    latent = torch.empty((8, 2, 512), dtype=torch.bfloat16)
    adapter = MagicMock()
    adapter.attention.return_value = latent, torch.empty(0)
    flash = SimpleNamespace(
        query=query,
        adapter=adapter,
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        cache_lens=torch.ones(2, dtype=torch.int32),
        cu=torch.arange(3, dtype=torch.int32),
        used_q=torch.ones(2, dtype=torch.int32),
        schedule=torch.zeros((2, 64), dtype=torch.int32),
        attn_mask=None,
    )
    impl = SimpleNamespace(
        use_quant_flashmla=True,
        kv_lora_rank=512,
        num_heads=8,
        fak_descale_float=k_descale,
        _v_up_proj=lambda value: value,
    )
    cache = torch.empty((2, 128, 1, 576), dtype=torch.float8_e4m3fn)
    metadata = SimpleNamespace(external_flashmla=flash)
    preprocessed = SimpleNamespace(ql_nope=q_nope, q_pe=q_pe, dequant_scale_q_nope=q_descale)
    monkeypatch.setattr(mla_v1, "record_attention_compute_start", lambda: None)

    assert AscendMLAImpl._forward_external_flashmla(impl, preprocessed, cache, metadata) is latent
    passed_q_descale = adapter.attention.call_args.kwargs["q_descale"]
    assert passed_q_descale.shape == q_descale.shape
    assert passed_q_descale.dtype == q_descale.dtype
    assert passed_q_descale.data_ptr() == q_descale.data_ptr()
    torch.testing.assert_close(passed_q_descale, q_descale)
    torch.testing.assert_close(adapter.attention.call_args.kwargs["k_descale"], k_descale)


def test_quant_flashmla_metadata_buffers_keep_rope_bf16():
    metadata_op = MagicMock(
        side_effect=lambda cache_seqlens, *_args, **_kwargs: torch.empty(
            (2, 64), dtype=torch.int32, device=cache_seqlens.device
        )
    )
    adapter = QuantFlashMLAImpl(QuantFlashMLAConfig(8, 0.125), MagicMock(), metadata_op)
    impl = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(),
            model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=4096),
        ),
        num_heads=8,
        scale=0.125,
        dtype=torch.float8_e4m3fn,
        use_mla_rope=True,
        use_quant_flashmla=True,
    )
    builder = FlashMLAMetadataBuilder(
        impl,
        torch.device("cpu"),
        max_num_reqs=2,
        attn_mask=torch.zeros((2048, 2048), dtype=torch.int8),
        adapter=adapter,
        max_query_len=1,
    )

    buffers = builder._allocate(tokens=2, rows=2, columns=1, causal=True)
    assert buffers.query.dtype == torch.float8_e4m3fn
    assert buffers.cos is not None and buffers.cos.dtype == torch.bfloat16
    assert buffers.sin is not None and buffers.sin.dtype == torch.bfloat16


@pytest.mark.parametrize("return_quantized_current", [False, True])
def test_quant_flashmla_no_rope_writer_scales_only_latent_cache(monkeypatch, return_quantized_current):
    normalized = torch.full((2, 1, 512), 2.0, dtype=torch.bfloat16)
    layernorm = MagicMock(return_value=normalized)
    scatter = MagicMock()
    monkeypatch.setattr(mla_v1.DeviceOperator, "reshape_and_cache", scatter)
    impl = SimpleNamespace(
        kv_a_layernorm=layernorm,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        use_quant_flashmla=True,
        fak_descale_reciprocal=torch.tensor([4.0], dtype=torch.float32),
    )
    positional = torch.full((2, 64), 3.0, dtype=torch.bfloat16)
    kv_input = torch.cat((torch.zeros((2, 512), dtype=torch.bfloat16), positional), dim=-1)
    fused = torch.empty((1, 128, 1, 576), dtype=torch.float8_e4m3fn)
    cache = fused[..., :512], fused[..., 512:]
    slots = torch.tensor([2, 7], dtype=torch.int64)

    k_pe, k_nope = AscendMLAImpl._exec_kv_no_rope(
        impl, kv_input, cache, slots, return_quantized_current=return_quantized_current
    )
    cached_latent = scatter.call_args.kwargs["key"]
    cached_positional = scatter.call_args.kwargs["value"]
    assert cached_latent.dtype == cached_positional.dtype == fused.dtype
    torch.testing.assert_close(cached_latent.float(), torch.full_like(cached_latent.float(), 8.0))
    torch.testing.assert_close(cached_positional.view(2, 64).bfloat16(), positional)
    if return_quantized_current:
        assert k_pe is cached_positional
        assert k_nope is cached_latent
    else:
        torch.testing.assert_close(k_pe.view(2, 64), positional)
        torch.testing.assert_close(k_nope, normalized)
