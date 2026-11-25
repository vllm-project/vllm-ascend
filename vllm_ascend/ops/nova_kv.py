# SPDX-License-Identifier: Apache-2.0
"""NOVA-KV cache updates, eager prefill and exact-window maintenance."""

import torch_npu

from vllm_ascend.ops.nova_kv_prefill import bounded_fia_prefill, exact_window_reader, planar_cold_reader
from vllm_ascend.ops.triton.nova_kv.cache import write_planar_packets


def nova_quant_cache(key, value, codec, views, slots, *, chunk_size):
    """Encode and scatter into allocator-owned planar cache pages."""
    encoded_key, encoded_value, scales = codec.encode(key, value, chunk_size=chunk_size)
    write_planar_packets(
        views, slots.long().contiguous(), encoded_key.contiguous(), encoded_value.contiguous(), scales.contiguous()
    )


def nova_prefill_attention(
    query,
    codec,
    views,
    block_table,
    window,
    scale,
    *,
    key,
    value,
    old_length,
    request_slot,
    causal_mask=None,
    query_tile=8192,
    kv_tile=32768,
):
    """Read retained history before the caller advances its exact window."""
    count, heads, groups, dim = query.shape
    if old_length == 0:
        result, _ = torch_npu.npu_fused_infer_attention_score(
            query.permute(1, 2, 0, 3).reshape(1, heads * groups, count, dim).contiguous(),
            key.transpose(0, 1).unsqueeze(0).contiguous(),
            value.transpose(0, 1).unsqueeze(0).contiguous(),
            atten_mask=causal_mask if count > 1 else None,
            input_layout="BNSD",
            actual_seq_lengths=[count],
            actual_seq_lengths_kv=[count],
            num_heads=heads * groups,
            num_key_value_heads=heads,
            scale=scale,
            sparse_mode=3 if count > 1 else 0,
            pre_tokens=2147483647,
            next_tokens=0,
            inner_precise=1,
        )
        return result.reshape(heads, groups, count, dim).permute(2, 0, 1, 3), None
    cold = planar_cold_reader(*views, block_table, codec)
    exact = exact_window_reader(window.key, window.value, request_slot, window.sink, window.ring, old_length)
    return bounded_fia_prefill(
        query,
        key,
        value,
        old_length,
        cold,
        exact,
        window.sink,
        window.ring - 1,
        query_tile=query_tile,
        kv_tile=kv_tile,
        scale=scale,
    )


def nova_exact_window_update(
    window, key, value, *, request_slots=None, positions=None, request_slot=None, old_length=None
):
    """Write decode rows or retain the non-overlapping rows of a prefill chunk."""
    if request_slot is not None:
        if old_length is None or request_slots is not None or positions is not None:
            raise ValueError("invalid NOVA prefill window metadata")
        window.seed_chunk(key, value, request_slot, old_length)
    else:
        if request_slots is None or positions is None or old_length is not None:
            raise ValueError("invalid NOVA decode window metadata")
        window.write_rows(key, value, request_slots, positions)
