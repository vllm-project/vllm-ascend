# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.ops  # noqa: F401  # Register ops before importing attention/device modules.
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.flashmla import FlashMLAMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.attention.quant_flashmla import (
    QuantFlashMLAConfig,
    QuantFlashMLAImpl,
)


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


@pytest.mark.parametrize("tokens", [1, 7, 129])
def test_quant_flashmla_writer_stages_then_scatters(monkeypatch, tokens):
    pages = (tokens + 127) // 128
    nope = torch.empty((pages, 128, 1, 512), dtype=torch.float8_e4m3fn)
    rope = torch.empty((pages, 128, 1, 64), dtype=torch.float8_e4m3fn)
    indices = torch.arange(pages * 128, dtype=torch.int64)
    current_nope = torch.full((tokens, 1, 1, 512), 2.0, dtype=torch.bfloat16)
    current_rope = torch.full((tokens, 1, 1, 64), 3.0, dtype=torch.bfloat16)
    rope_scale = torch.ones(64, dtype=torch.float32)
    latent_scale = torch.full((512,), 4.0, dtype=torch.float32)

    def writer(*args, **kwargs):
        assert kwargs["k_rope_scale"] is rope_scale
        assert kwargs["c_kv_scale"] is latent_scale
        # Match the delivered operator: FP8 caches, BF16 original outputs.
        args[5].flatten(0, 1)[:tokens].copy_(current_rope.view(tokens, 1, 64))
        args[6].flatten(0, 1)[:tokens].copy_(current_nope.view(tokens, 1, 512).float() * 4)
        return args[5], args[6], current_rope, current_nope

    rmsnorm_rope = MagicMock(side_effect=writer)
    scatter = MagicMock()
    monkeypatch.setattr(mla_v1.torch_npu, "npu_kv_rmsnorm_rope_cache", rmsnorm_rope)
    monkeypatch.setattr(mla_v1.DeviceOperator, "reshape_and_cache", scatter)
    impl = SimpleNamespace(
        _flashmla_writer_buffers=(nope, rope, indices),
        _quant_flashmla_writer_scales=(rope_scale, latent_scale),
        _validate_prolog_v3_cache=lambda _cache: True,
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-6),
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    kv_no_split = torch.empty((tokens, 576), dtype=torch.bfloat16)
    cos = torch.empty((tokens, 64), dtype=torch.bfloat16)
    sin = torch.empty_like(cos)
    fused = torch.empty((pages + 1, 128, 1, 576), dtype=torch.float8_e4m3fn)
    cache = fused[..., :512], fused[..., 512:]
    slots = torch.arange(tokens, dtype=torch.int64) + 127

    result = AscendMLAImpl._exec_quant_flashmla_kv_writer(impl, kv_no_split, cos, sin, cache, slots)
    assert result[0] is current_rope
    assert result[1] is current_nope
    assert rmsnorm_rope.call_args.kwargs["is_output_kv"] is True
    scatter.assert_called_once()
    kwargs = scatter.call_args.kwargs
    assert kwargs["key"].dtype == kwargs["value"].dtype == torch.float8_e4m3fn
    assert kwargs["key_cache"] is cache[0]
    assert kwargs["value_cache"] is cache[1]
    assert kwargs["slot_mapping"] is slots
    torch.testing.assert_close(kwargs["key"].float(), torch.full((tokens, 1, 512), 8.0))
    torch.testing.assert_close(kwargs["value"].float(), torch.full((tokens, 1, 64), 3.0))


def test_quant_flashmla_writer_scales_refresh_in_place():
    impl = SimpleNamespace(
        use_quant_flashmla=True,
        _quant_flashmla_writer_scales=None,
        _flashmla_writer_buffers=None,
        fak_descale_reciprocal=torch.tensor([4.0]),
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_kv_heads=1,
        dtype=torch.float8_e4m3fn,
        W_UV=torch.empty(1),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
            cache_config=SimpleNamespace(block_size=128),
        ),
    )
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    scales = impl._quant_flashmla_writer_scales
    addresses = tuple(x.data_ptr() for x in scales)
    buffers = impl._flashmla_writer_buffers
    torch.testing.assert_close(scales[0], torch.ones(64))
    torch.testing.assert_close(scales[1], torch.full((512,), 4.0))
    impl.fak_descale_reciprocal = torch.tensor([8.0])
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    assert impl._flashmla_writer_buffers is buffers
    assert tuple(x.data_ptr() for x in impl._quant_flashmla_writer_scales) == addresses
    torch.testing.assert_close(scales[1], torch.full((512,), 8.0))


def test_quant_flashmla_no_rope_writer_scales_only_latent_cache(monkeypatch):
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

    k_pe, k_nope = AscendMLAImpl._exec_kv_no_rope(impl, kv_input, cache, slots)
    torch.testing.assert_close(k_nope, normalized)
    torch.testing.assert_close(k_pe.view(2, 64), positional)
    cached_latent = scatter.call_args.kwargs["key"]
    assert cached_latent.dtype == fused.dtype
    torch.testing.assert_close(cached_latent.float(), torch.full_like(cached_latent.float(), 8.0))
    # The positional cache is deliberately not multiplied by k's descale reciprocal.
    assert scatter.call_args.kwargs["value"] is k_pe
