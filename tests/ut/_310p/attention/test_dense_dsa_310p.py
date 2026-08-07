# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend._310p.attention.dense_dsa import (
    dense_causal_current_attention,
    dense_decode_swa_attention,
    dense_dspark_swa_attention,
    gather_paged_swa_cache,
    infer_blocks_per_phys_block,
    infer_blocks_per_phys_block_from_shape,
    write_paged_swa_cache,
)


def test_fresh_prefill_attention_matches_manual_causal_reference() -> None:
    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, -1.0]],
            [[-0.5, 1.0], [0.25, 0.75]],
        ],
        dtype=torch.float32,
    )
    kv = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32)
    offsets = torch.tensor([0, 3], dtype=torch.int32)
    sinks = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dense_causal_current_attention(q, kv, offsets, window_size=2, softmax_scale=0.5, sinks=sinks)

    expected = torch.empty_like(q)
    keys_all = kv[:, 0]
    for i in range(3):
        start = max(0, i + 1 - 2)
        keys = keys_all[start : i + 1]
        logits = q[i] @ keys.T * 0.5
        probs = torch.softmax(torch.cat((logits, sinks[:, None]), dim=-1), dim=-1)[:, : keys.shape[0]]
        expected[i] = probs @ keys

    torch.testing.assert_close(actual, expected)


def test_infers_hybrid_block_split_from_slot_mapping() -> None:
    # Physical block 2 is exposed as four logical blocks [8, 9, 10, 11].
    block_table = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)
    positions = torch.arange(6, dtype=torch.int64)
    slot_mapping = torch.stack(
        (
            torch.full((6,), 2, dtype=torch.int32),
            torch.arange(6, dtype=torch.int32),
        ),
        dim=-1,
    )

    factor = infer_blocks_per_phys_block(
        block_table,
        slot_mapping,
        positions,
        torch.tensor([0, 6], dtype=torch.int32),
        block_size=8,
    )

    assert factor == 4


def test_infers_hybrid_block_split_from_static_table_shape() -> None:
    block_table = torch.zeros((8, 64), dtype=torch.int32)

    assert infer_blocks_per_phys_block_from_shape(block_table, block_size=32, max_model_len=128) == 16


def test_gather_paged_cache_decodes_hybrid_logical_blocks() -> None:
    cache = torch.empty((4, 8, 1, 1), dtype=torch.float32)
    for block in range(cache.shape[0]):
        for offset in range(cache.shape[1]):
            cache[block, offset, 0, 0] = block * 100 + offset

    # Physical blocks 2 and 3, each split into four logical blocks of size 2.
    block_table_row = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int32)
    actual = gather_paged_swa_cache(
        cache,
        block_table_row,
        start=0,
        end=10,
        block_size=8,
        blocks_per_phys_block=4,
    )

    expected = torch.tensor(
        [[200], [201], [202], [203], [204], [205], [206], [207], [300], [301]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


def test_gather_paged_cache_preserves_unsplit_layout() -> None:
    cache = torch.arange(3 * 4, dtype=torch.float32).view(3, 4, 1, 1)
    actual = gather_paged_swa_cache(
        cache,
        torch.tensor([1, 2], dtype=torch.int32),
        start=0,
        end=6,
        block_size=4,
    )

    expected = torch.tensor([[4], [5], [6], [7], [8], [9]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_vectorized_decode_matches_paged_reference() -> None:
    cache = torch.zeros((4, 8, 1, 2), dtype=torch.float32)
    for block in range(cache.shape[0]):
        for offset in range(cache.shape[1]):
            cache[block, offset, 0] = torch.tensor([block * 10 + offset, 1.0])

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    seq_lens = torch.tensor([6, 10], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    sinks = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dense_decode_swa_attention(
        q,
        cache,
        block_table,
        seq_lens,
        block_size=8,
        blocks_per_phys_block=4,
        window_size=4,
        softmax_scale=0.25,
        sinks=sinks,
    )

    expected = torch.empty_like(q)
    for request_idx, seq_len in enumerate(seq_lens.tolist()):
        keys = gather_paged_swa_cache(
            cache,
            block_table[request_idx],
            start=max(0, seq_len - 4),
            end=seq_len,
            block_size=8,
            blocks_per_phys_block=4,
        )
        logits = q[request_idx] @ keys.T * 0.25
        probs = torch.softmax(torch.cat((logits, sinks[:, None]), dim=-1), dim=-1)[:, : keys.shape[0]]
        expected[request_idx] = probs @ keys

    torch.testing.assert_close(actual, expected)


def test_vectorized_speculative_decode_matches_paged_reference() -> None:
    cache = torch.zeros((4, 8, 1, 2), dtype=torch.float32)
    for block in range(cache.shape[0]):
        for offset in range(cache.shape[1]):
            cache[block, offset, 0] = torch.tensor([block * 10 + offset, 1.0])

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, -1.0]],
            [[-0.5, 1.0], [0.25, 0.75]],
            [[1.0, 1.0], [-0.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    seq_lens = torch.tensor([6, 10], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    sinks = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dense_decode_swa_attention(
        q,
        cache,
        block_table,
        seq_lens,
        block_size=8,
        blocks_per_phys_block=4,
        window_size=4,
        softmax_scale=0.25,
        sinks=sinks,
    )

    expected = torch.empty_like(q)
    query_len = 2
    for request_idx, seq_len in enumerate(seq_lens.tolist()):
        context_len = seq_len - query_len
        for query_idx in range(query_len):
            visible_end = context_len + query_idx + 1
            keys = gather_paged_swa_cache(
                cache,
                block_table[request_idx],
                start=max(0, visible_end - 4),
                end=visible_end,
                block_size=8,
                blocks_per_phys_block=4,
            )
            row = request_idx * query_len + query_idx
            logits = q[row] @ keys.T * 0.25
            probs = torch.softmax(torch.cat((logits, sinks[:, None]), dim=-1), dim=-1)[:, : keys.shape[0]]
            expected[row] = probs @ keys

    torch.testing.assert_close(actual, expected)


def test_dspark_attention_uses_full_noncausal_query_block() -> None:
    cache = torch.zeros((2, 4, 1, 2), dtype=torch.float32)
    flat_cache = cache.reshape(-1, 1, 2)
    flat_cache[0, 0] = torch.tensor([1.0, 0.0])
    flat_cache[1, 0] = torch.tensor([0.0, 1.0])
    # Current three-token DSpark block.  The first query must be allowed to
    # see slots 3 and 4 as well, unlike an ordinary causal speculative block.
    flat_cache[2, 0] = torch.tensor([1.0, 1.0])
    flat_cache[3, 0] = torch.tensor([2.0, 0.0])
    flat_cache[4, 0] = torch.tensor([0.0, 2.0])

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, -1.0]],
            [[-0.5, 1.0], [0.25, 0.75]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [[0, 1, 2, 3, 4, -1]],
            [[0, 1, 2, 3, 4, -1]],
            [[0, 1, 2, 3, 4, -1]],
        ],
        dtype=torch.int32,
    )
    sinks = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dense_dspark_swa_attention(
        q,
        cache,
        indices,
        softmax_scale=0.25,
        sinks=sinks,
    )

    keys = flat_cache[:5, 0]
    expected = torch.empty_like(q)
    for row in range(q.shape[0]):
        logits = q[row] @ keys.T * 0.25
        probs = torch.softmax(
            torch.cat((logits, sinks[:, None]), dim=-1),
            dim=-1,
        )[:, : keys.shape[0]]
        expected[row] = probs @ keys

    torch.testing.assert_close(actual, expected)

    # A causal first query would only see context slots 0, 1 and its own slot
    # 2, so explicitly guard that the DSpark result is different.
    causal_keys = flat_cache[:3, 0]
    causal_logits = q[0] @ causal_keys.T * 0.25
    causal_probs = torch.softmax(
        torch.cat((causal_logits, sinks[:, None]), dim=-1),
        dim=-1,
    )[:, : causal_keys.shape[0]]
    causal_first = causal_probs @ causal_keys
    assert not torch.allclose(actual[0], causal_first)


def test_paged_cache_write_ignores_invalid_slots_without_boolean_indexing() -> None:
    cache = torch.arange(8 * 2, dtype=torch.float32).view(2, 4, 1, 2)
    original = cache.clone()
    kv = torch.tensor(
        [
            [[-100.0, -100.0]],
            [[70.0, 71.0]],
        ],
        dtype=torch.float32,
    )

    # The first invalid row uses the last cache row as its scratch slot,
    # while the second row performs a real write to that same row. Invalid
    # writes are ordered first, so the real value must win.
    write_paged_swa_cache(
        cache,
        kv,
        torch.tensor([-1, 7], dtype=torch.int64),
        block_size=4,
    )

    expected = original
    expected.reshape(-1, 1, 2)[7] = kv[1]
    torch.testing.assert_close(cache, expected)


def test_paged_cache_write_supports_block_offset_slot_mapping() -> None:
    cache = torch.zeros((2, 4, 1, 2), dtype=torch.float32)
    kv = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float32)
    slots = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)

    write_paged_swa_cache(cache, kv, slots, block_size=4)

    flat = cache.reshape(-1, 1, 2)
    torch.testing.assert_close(flat[1], kv[0])
    torch.testing.assert_close(flat[6], kv[1])
