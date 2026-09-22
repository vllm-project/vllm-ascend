# SPDX-License-Identifier: Apache-2.0
"""A5 TND FlashAttention calls and scheduling for expanded MLA operands."""

import torch

C8_METADATA_MIN_QUERY_LENGTH = 65


def flash_attn_prefill(query, key, value, cumulative, lengths, scale, attn_mask=None):
    """Run a packed BF16 D64 prefill batch without cached history."""
    from cann_ops_transformer.ops import flash_attn, flash_attn_metadata

    maximum = max(lengths)
    kwargs = dict(
        cu_seqlens_q=cumulative,
        cu_seqlens_kv=cumulative,
        max_seqlen_q=maximum,
        max_seqlen_kv=maximum,
        mask_mode=3 if attn_mask is not None else 0,
        layout_q="TND",
        layout_kv="TND",
        layout_out="TND",
    )
    metadata = flash_attn_metadata(
        query.shape[1],
        key.shape[1],
        query.shape[2],
        head_dim_v=value.shape[2],
        batch_size=len(lengths),
        **kwargs,
    )
    return flash_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        metadata=metadata,
        attn_mask=attn_mask,
        softmax_scale=scale,
        **kwargs,
    )[0]


def native_flash_adapters(
    num_heads: int,
    scale: float,
    attn_mask: torch.Tensor,
    *,
    c8: bool = False,
    fake_quant: bool = False,
):
    """Use a matched CANN 192/128 binding/runtime; no custom backend or switch."""
    from cann_ops_transformer.ops import flash_attn, flash_attn_metadata

    if c8 and fake_quant:
        raise ValueError("select either direct FP8 or fake-quantized BF16 attention")
    if c8 or fake_quant:
        from vllm_ascend.ops.flash_attn_c8_quant import fake_quant_flash_attn_c8, quantize_flash_attn_c8

    def kwargs(call):
        return dict(
            cu_seqlens_q=call.query_cu,
            cu_seqlens_kv=call.kv_cu,
            seqused_q=call.query_used,
            seqused_kv=call.kv_used,
            max_seqlen_q=max(call.query_lengths),
            max_seqlen_kv=max(call.kv_lengths),
            mask_mode=call.mask_mode,
            layout_q="TND",
            layout_kv="TND",
            layout_out="TND",
        )

    def schedule(call):
        metadata_kwargs = kwargs(call)
        if c8:
            # Native C8 uses M128/N128. The generic metadata producer selects
            # M64/N256 for short queries; only its tile-selection bound changes.
            # Actual cumulative/used lengths and runtime max lengths stay exact.
            metadata_kwargs["max_seqlen_q"] = max(metadata_kwargs["max_seqlen_q"], C8_METADATA_MIN_QUERY_LENGTH)
        return flash_attn_metadata(
            num_heads, num_heads, 192, head_dim_v=128, batch_size=len(call.query_lengths), **metadata_kwargs
        )

    def attention(query, key_nope, value, key_rope, call):
        if c8:
            q, k, v, qr, kr, sq, sk, sv = quantize_flash_attn_c8(query, key_nope, value, key_rope=key_rope)
            return torch.ops._C_ascend.flash_attn_c8(
                q,
                k,
                v,
                qr,
                kr,
                sq,
                sk,
                sv,
                call.query_cu,
                call.kv_cu,
                call.schedule,
                softmax_scale=scale,
                mask_mode=call.mask_mode,
                max_seqlen_q=max(call.query_lengths),
                max_seqlen_kv=max(call.kv_lengths),
                seqused_q=call.query_used,
                attn_mask=attn_mask if call.mask_mode else None,
                return_softmax_lse=True,
            )
        if fake_quant:
            query, key, value = fake_quant_flash_attn_c8(query, key_nope, value, key_rope)
        else:
            key = torch.cat((key_nope, key_rope.reshape(-1, 1, 64).expand(-1, num_heads, -1)), dim=-1)
            value = value.contiguous()
        return flash_attn(
            query,
            key,
            value,
            metadata=call.schedule,
            attn_mask=attn_mask if call.mask_mode else None,
            softmax_scale=scale,
            return_softmax_lse=True,
            **kwargs(call),
        )

    return schedule, attention
