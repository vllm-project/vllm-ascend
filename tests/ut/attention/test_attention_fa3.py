from importlib import import_module, util
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu


def _fa3_available() -> bool:
    try:
        if util.find_spec("flash_attn_npu_v3") is None:
            return False
        mod = import_module("flash_attn_npu_v3")
        return hasattr(mod, "flash_attn_with_kvcache")
    except ImportError:
        return False


@pytest.mark.skipif(not _fa3_available(), reason="flash_attn_npu_v3 is not installed")
@pytest.mark.parametrize(
    "decode_lengths,prefill_lengths",
    [([], [4, 7]), ([1, 1], []), ([1, 3, 1], []), ([1, 1], [4, 7]), ([1, 3, 1], [4, 7])],
)
def test_fa3_forward_batch_metadata(monkeypatch, decode_lengths, prefill_lengths):
    from vllm_ascend.attention.fa3_v1 import AscendFAImpl

    lengths = decode_lengths + prefill_lengths
    offsets = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32)
    num_tokens = sum(lengths)
    num_decodes = len(decode_lengths)
    block_tables = torch.arange(len(lengths) * 2, dtype=torch.int32).reshape(-1, 2).npu()
    seq_lens = torch.tensor(lengths, dtype=torch.int32) + 16
    metadata = SimpleNamespace(
        actual_seq_lengths_q=offsets[1:].tolist(),
        num_decodes=num_decodes,
        num_decode_tokens=sum(decode_lengths),
        num_prefills=len(prefill_lengths),
        block_tables=block_tables,
        query_start_loc=offsets.npu(),
        seq_lens=seq_lens,
    )
    # Distinct values check the partition and concatenation order, including padding.
    query = torch.arange((num_tokens + 2) * 4, dtype=torch.float32).reshape(-1, 1, 4).npu()
    output = torch.full_like(query, -1)
    calls = []

    def check_partition(q, blocks, query_offsets, kv_lengths, causal, max_seq_len):
        start = sum(len(part) for part in calls)
        part = decode_lengths if start == 0 and decode_lengths else prefill_lengths
        expected_offsets = torch.tensor([0, *np.cumsum(part)], dtype=torch.int32)
        torch.testing.assert_close(query_offsets.cpu(), expected_offsets)
        torch.testing.assert_close(blocks, block_tables[start : start + len(part)])
        torch.testing.assert_close(kv_lengths.cpu(), seq_lens[start : start + len(part)])
        assert q.shape[0] == sum(part)
        assert causal == (start >= num_decodes or any(length > 1 for length in part))
        assert max_seq_len == max(seq_lens[start : start + len(part)])
        calls.append(part)
        return q.clone()

    impl = AscendFAImpl.__new__(AscendFAImpl)
    monkeypatch.setattr(impl, "_flash_attn_with_kvcache", check_partition)
    impl.forward_impl(query, None, None, (), metadata, output)
    assert len(calls) == bool(decode_lengths) + bool(prefill_lengths)
    torch.testing.assert_close(output[:num_tokens], query[:num_tokens])
    torch.testing.assert_close(output[num_tokens:], torch.full_like(output[num_tokens:], -1))


@pytest.mark.skipif(not _fa3_available(), reason="flash_attn_npu_v3 is not installed")
@pytest.mark.parametrize("decode_lengths,prefill_lengths", [([1, 3, 1], [4, 7]), ([3, 2], [])])
def test_fa3_forward_speculative_causal_reference(decode_lengths, prefill_lengths):
    from vllm_ascend.attention.fa3_v1 import AscendFAImpl

    lengths = decode_lengths + prefill_lengths
    num_heads, num_kv_heads, head_size, block_size = 4, 2, 128, 128
    offsets = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32)
    seq_lens = torch.tensor(lengths, dtype=torch.int32) + 8
    generator = torch.Generator().manual_seed(17991)
    query = torch.randn(sum(lengths), num_heads, head_size, generator=generator).bfloat16()
    key = torch.randn(len(lengths), block_size, num_kv_heads, head_size, generator=generator).bfloat16()
    value = torch.randn(len(lengths), block_size, num_kv_heads, head_size, generator=generator).bfloat16()
    metadata = SimpleNamespace(
        actual_seq_lengths_q=offsets[1:].tolist(),
        num_decodes=len(decode_lengths),
        num_decode_tokens=sum(decode_lengths),
        num_prefills=len(prefill_lengths),
        block_tables=torch.arange(len(lengths), dtype=torch.int32).reshape(-1, 1).npu(),
        query_start_loc=offsets.npu(),
        seq_lens=seq_lens,
    )
    reference = []
    for row, q_len in enumerate(lengths):
        kv_len = int(seq_lens[row])
        q = query[offsets[row] : offsets[row + 1]].float().transpose(0, 1)
        k = key[row, :kv_len].float().repeat_interleave(num_heads // num_kv_heads, dim=1).transpose(0, 1)
        v = value[row, :kv_len].float().repeat_interleave(num_heads // num_kv_heads, dim=1).transpose(0, 1)
        mask = torch.arange(kv_len)[None, :] <= torch.arange(q_len)[:, None] + kv_len - q_len
        scores = (q @ k.transpose(-1, -2)) / head_size**0.5
        reference.append((scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v).transpose(0, 1))

    impl = AscendFAImpl.__new__(AscendFAImpl)
    impl.num_kv_heads = num_kv_heads
    impl.head_size = head_size
    impl.key_cache = key.npu()
    impl.value_cache = value.npu()
    actual = impl.forward_impl(query.npu(), None, None, (), metadata, torch.empty_like(query, device="npu"))
    torch.testing.assert_close(actual.cpu().float(), torch.cat(reference), rtol=1e-2, atol=1e-2)


def ref_fused_infer_attention(
    query,
    key,
    value,
    block_table,
    block_size,
    actual_seq_lengths_q,
    actual_seq_lengths_kv,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    attn_mask,
    causal,
):
    if not causal:
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            sparse_mode=0,
        )
    else:
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            sparse_mode=3,
        )

    attn_output = attn_output.view(-1, num_heads, head_size)
    return attn_output


test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal)
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False),
    (torch.bfloat16, 5, 4, 1, 1024, 1024, 128, 128, True),
    (torch.float16, 7, 16, 8, 512, 512, 128, 128, False),
]


@pytest.mark.skipif(not _fa3_available(), reason="flash_attn_npu_v3 is not installed")
@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal", test_cases
)
def test_fa_custom_ops_tnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, block_size, is_causal
):
    q_min_range = -1.0
    q_max_range = 1.0
    kv_min_range = -1.0
    kv_max_range = 1.0
    block_size = 128
    num_blocks = 64

    q_sequences = sorted(
        torch.randint(low=1, high=q_seqlen + 1, size=(batch_size,)).tolist(), reverse=False
    )  # actual_seq_lengths in fia need in ascending order
    kv_sequences = [torch.randint(low=q, high=kv_seqlen + 1, size=(1,)).item() for q in q_sequences]

    t_q_sum = sum(q_sequences)

    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(t_q_sum, num_heads, head_size)).to(data_type).npu()

    key_cache = None
    value_cache = None
    block_tables = []

    key_cache = (
        (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size))
        .to(data_type)
        .npu()
    )
    value_cache = (
        (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size))
        .to(data_type)
        .npu()
    )
    max_num_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
    for i in range(batch_size):
        block_table = [max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int32).npu()

    q_seqlen_list = q_sequences
    kv_seqlen_list = kv_sequences

    scale = 1.0 / (head_size**0.5)
    window_size_left = -1
    window_size_right = -1
    is_rotary_interleaved = False
    num_splits = 0
    kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int32).npu()
    rotary_cos = None
    rotary_sin = None
    cache_batch_idx = None
    leftpad_k = None
    new_q_seqlen_list = None

    new_q_seqlen_list = [0]
    pre_seq_sum = 0
    for i in range(batch_size):
        pre_seq_sum += q_seqlen_list[i]
        new_q_seqlen_list.append(pre_seq_sum)
    new_q_seqlen_list = torch.tensor(new_q_seqlen_list, dtype=torch.int32).npu()

    from flash_attn_npu_v3 import flash_attn_with_kvcache  # type: ignore[import-not-found]

    out_out = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        None,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=kv_seqlen_list,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        page_table=block_tables,
        cu_seqlens_q=new_q_seqlen_list,
        cu_seqlens_k_new=None,
        max_seqlen_q=q_seqlen,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=None,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=is_rotary_interleaved,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=None,
        sm_margin=0,
        return_softmax_lse=False,
    )

    ref_out = torch.empty((t_q_sum, num_heads, head_size), dtype=data_type)

    if is_causal:
        attn_mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(query.device)
    else:
        attn_mask = None

    q_cumsum = torch.tensor(np.cumsum(q_sequences), dtype=torch.int32, device=query.device)

    ref_out = ref_fused_infer_attention(
        query,
        key_cache.permute(0, 2, 1, 3).contiguous(),  # [num_blocks, num_kv_heads, block_size, head_size]
        value_cache.permute(0, 2, 1, 3).contiguous(),
        block_tables,
        block_size,
        q_cumsum,
        kv_seqlen_list,
        num_heads,
        kv_heads,
        head_size,
        scale,
        attn_mask,
        is_causal,
    )

    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(out_out.cpu(), ref_out.cpu(), rtol=rtol, atol=atol)
