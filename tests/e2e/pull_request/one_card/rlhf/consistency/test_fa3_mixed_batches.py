"""Real-NPU regression for FA3 mixed prefill and speculative decode batches."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "q_lens,kv_lens,num_decodes",
    [
        ([3, 5], [5, 133], 1),
        ([1, 1], [129, 257], 2),
        ([3], [5], 1),
        ([10], [10], 0),
        ([1, 129], [129, 257], 1),
        ([3, 1, 7], [129, 257, 133], 2),
    ],
)
def test_fa3_mixed_paged_attention(q_lens, kv_lens, num_decodes):
    pytest.importorskip("flash_attn_npu_v3")
    # Import lazily: FA3 is an optional dependency; initialize the ops package
    # before attention modules to avoid their circular import during collection.
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.attention.fa3_v1 import AscendFAImpl

    torch.manual_seed(42)
    block_size, heads, kv_heads, head_size, blocks = 128, 8, 2, 128, 12
    # Use dense K/V to isolate metadata semantics from physical-page layout.
    k_cpu = torch.randn(blocks, block_size, kv_heads, head_size, dtype=torch.bfloat16)
    v_cpu = torch.randn_like(k_cpu)
    k, v = k_cpu.npu(), v_cpu.npu()
    table = torch.tensor([[7, 2, 10], [9, 1, 6], [11, 4, 3]][: len(q_lens)], dtype=torch.int32)
    q_cpu = torch.randn(sum(q_lens), heads, head_size, dtype=torch.bfloat16)
    expected = []
    start = 0
    for i, (nq, nk) in enumerate(zip(q_lens, kv_lens)):
        keys = (
            k_cpu[table[i].long()].reshape(-1, kv_heads, head_size)[:nk].float().repeat_interleave(heads // kv_heads, 1)
        )
        values = (
            v_cpu[table[i].long()].reshape(-1, kv_heads, head_size)[:nk].float().repeat_interleave(heads // kv_heads, 1)
        )
        logits = torch.einsum("qhd,khd->hqk", q_cpu[start : start + nq].float(), keys) * head_size**-0.5
        mask = torch.arange(nk)[None, :] > torch.arange(nq)[:, None] + nk - nq
        logits.masked_fill_(mask[None], float("-inf"))
        expected.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), values))
        start += nq
    cu = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    meta = SimpleNamespace(
        actual_seq_lengths_q=cu[1:].tolist(),
        num_decodes=num_decodes,
        num_decode_tokens=sum(q_lens[:num_decodes]),
        num_prefills=len(q_lens) - num_decodes,
        block_tables=table.npu(),
        query_start_loc=cu.npu(),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
    )
    impl = AscendFAImpl.__new__(AscendFAImpl)
    impl.key_cache, impl.value_cache = k, v
    impl.num_kv_heads, impl.head_size, impl.scale = kv_heads, head_size, head_size**-0.5
    query = q_cpu.npu()
    actual = impl.forward_impl(query, None, None, (k, v), meta, torch.empty_like(query))
    torch.testing.assert_close(actual.float().cpu(), torch.cat(expected), atol=0.02, rtol=0.02)
