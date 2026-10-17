"""Real-NPU regression for FA3 strided paged caches."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("page_step", [1, 2, 3])
@pytest.mark.parametrize(
    "q_lens,kv_lens,num_decodes",
    [
        ([1, 1], [129, 257], 2),
        ([10], [10], 0),
    ],
)
def test_fa3_strided_paged_attention(page_step, q_lens, kv_lens, num_decodes):
    pytest.importorskip("flash_attn_npu_v3")
    # Import lazily: FA3 is an optional dependency; initialize the ops package
    # before attention modules to avoid their circular import during collection.
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.attention.fa3_v1 import AscendFAImpl

    torch.manual_seed(42)
    block_size, heads, kv_heads, head_size, blocks = 128, 8, 2, 128, 12
    # Padding and interleaved K/V exercise the physical page stride. Neither
    # the requests nor their logical blocks use sequential physical pages.
    raw = torch.randn(blocks, page_step + 1, block_size, kv_heads, head_size, dtype=torch.bfloat16)
    if page_step == 1:
        k_cpu, v_cpu = raw[:, 0].contiguous(), raw[:, 1].contiguous()
        k, v = k_cpu.npu(), v_cpu.npu()
    else:
        raw = raw[:, :page_step].contiguous()
        k_cpu, v_cpu = raw[:, 0], raw[:, 1]
        raw_npu = raw.npu()
        k, v = raw_npu[:, 0], raw_npu[:, 1]
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


@pytest.mark.parametrize("page_step", [1, 2, 3])
def test_fa3_page_view_shares_storage(page_step):
    pytest.importorskip("flash_attn_npu_v3")
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.attention.fa3_v1 import _as_fa3_paged_cache

    storage = torch.randn(4, page_step, 128, 2, 128)
    cache = storage[:, page_step - 1]
    view, step = _as_fa3_paged_cache(cache)
    assert step == page_step
    assert view.data_ptr() == cache.data_ptr()
    assert view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    torch.testing.assert_close(view[::step], cache)


def test_fa3_rejects_unsupported_page_layout():
    pytest.importorskip("flash_attn_npu_v3")
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.attention.fa3_v1 import _as_fa3_paged_cache

    cache = torch.empty(4, 128, 2, 128).transpose(1, 2)
    with pytest.raises(ValueError, match="dense inner pages"):
        _as_fa3_paged_cache(cache)


@pytest.mark.parametrize("page_step", [1, 2, 3])
def test_fa3_empty_cache(page_step):
    pytest.importorskip("flash_attn_npu_v3")
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.attention.fa3_v1 import _as_fa3_paged_cache

    cache = torch.empty(0, page_step, 128, 2, 128)[:, 0]
    view, step = _as_fa3_paged_cache(cache)
    assert view is cache
    assert step == 1


@pytest.mark.parametrize("page_step", [1, 2, 3])
def test_fa3_preserves_page_table_sentinels(monkeypatch, page_step):
    pytest.importorskip("flash_attn_npu_v3")
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.attention import fa3_v1

    cache = torch.empty(4, page_step, 128, 2, 128, device="npu", dtype=torch.bfloat16)[:, 0]
    impl = fa3_v1.AscendFAImpl.__new__(fa3_v1.AscendFAImpl)
    impl.key_cache = impl.value_cache = cache
    impl.num_kv_heads, impl.head_size = 2, 128
    table = torch.tensor([[0, 3, -1, -2]], device="npu", dtype=torch.int32)
    original = table.clone()
    captured = {}

    def capture(query, key, value, **kwargs):
        captured["table"] = kwargs["page_table"]
        return query

    # Inspect the actual adapter boundary, without asking the FA3 kernel to
    # consume an invalid block ID in the active sequence.
    monkeypatch.setattr(fa3_v1, "_fa3_fn", capture)
    query = torch.empty(1, 2, 128, device="npu", dtype=torch.bfloat16)
    impl._flash_attn_with_kvcache(
        query,
        table,
        torch.tensor([0, 1], device="npu", dtype=torch.int32),
        torch.tensor([1], device="npu", dtype=torch.int32),
        False,
        1,
    )
    expected = torch.tensor([[0, 3 * page_step, -1, -2]], dtype=torch.int32)
    torch.testing.assert_close(captured["table"].cpu(), expected)
    torch.testing.assert_close(table, original)
