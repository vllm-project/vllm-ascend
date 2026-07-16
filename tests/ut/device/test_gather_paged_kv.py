#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for _gather_paged_kv_to_dense (block-aligned variable gather).

The function gathers paged block KV cache into dense TND for
npu_fusion_attention. It was rewritten from max-padding (materialise
(num_seqs, max_seq_len, ...) then mask) to a block-aligned variable gather
(each seq contributes only the blocks it owns) to avoid OOM and bandwidth
waste when a batch mixes short and very long sequences.

These tests run on CPU with plain torch ops (the function uses no torch_npu
op), asserting numerical equivalence against the original max-padding
reference and correct TND shape / trim behaviour.
"""

import pytest
import torch

from vllm_ascend.device.utils import _gather_paged_kv_to_dense


def _ref_gather_padding(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original max-padding implementation, kept here as a numerical oracle."""
    block_size = key_cache.shape[1]
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long)
    num_blocks = (max_seq_len + block_size - 1) // block_size
    bt = block_table[: len(seq_lens), :num_blocks].long()
    flat_block_ids = bt.reshape(-1)
    max_tokens_padded = num_blocks * block_size
    dense_shape = (len(seq_lens), max_tokens_padded, num_kv_heads, head_size)
    gathered_key = key_cache.index_select(0, flat_block_ids).reshape(dense_shape)
    gathered_value = value_cache.index_select(0, flat_block_ids).reshape(dense_shape)
    positions = torch.arange(max_tokens_padded, dtype=torch.long)
    valid_mask = positions.unsqueeze(0) < seq_lens_tensor.unsqueeze(1)
    return gathered_key[valid_mask].contiguous(), gathered_value[valid_mask].contiguous()


def _make_cache(num_blocks, block_size, num_kv_heads, head_size, seed):
    """Block-stamped cache: block i is filled with value i so gathered tokens
    reveal which block they came from (order/ownership check)."""
    g = torch.Generator().manual_seed(seed)
    key_cache = torch.empty(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float32)
    value_cache = torch.empty_like(key_cache)
    for b in range(num_blocks):
        key_cache[b] = b + 0.01 * torch.randn(block_size, num_kv_heads, head_size, generator=g)
        value_cache[b] = 1000 + b + 0.01 * torch.randn(block_size, num_kv_heads, head_size, generator=g)
    return key_cache, value_cache


def _block_table_for(seq_lens, block_size, num_blocks_total):
    """Assign each seq ceil(slen/block_size) distinct blocks; pad unused columns
    with a valid (block 0) id — mirroring real block_table layout where tail
    columns may point at an arbitrary valid block but are never selected by the
    gather (only the first ceil(slen/block_size) columns per seq are used)."""
    import random

    rng = random.Random(42)
    pool = list(range(1, num_blocks_total))  # reserve block 0 as the pad target
    rng.shuffle(pool)
    max_cols = (max(seq_lens) + block_size - 1) // block_size
    bt = torch.zeros(len(seq_lens), max_cols, dtype=torch.int32)
    pos = 0
    for i, slen in enumerate(seq_lens):
        nb = (slen + block_size - 1) // block_size
        for c in range(nb):
            bt[i, c] = pool[pos % len(pool)]
            pos += 1
    return bt


# (seq_lens, block_size, num_kv_heads, head_size) — covers the bug scenario
# (one very long + many short), block-aligned lengths, single seq, equal lens.
CASES = [
    # Bug scenario: 1x long (109K-ish scaled down) + many short.
    ([1097, 13, 11, 12, 15, 6, 7], 128, 4, 512),
    # Mixed short/long, non-block-aligned tails.
    ([100, 200, 50, 333], 64, 8, 256),
    # Block-aligned lengths (no in-block tail padding).
    ([128, 256, 384], 128, 4, 128),
    # Single sequence.
    ([500], 64, 4, 512),
    # All equal length.
    ([300, 300, 300, 300], 128, 8, 256),
    # One empty-ish (seq_len=1) alongside a long one.
    ([1, 2000], 128, 4, 512),
]


@pytest.mark.parametrize("seq_lens,block_size,num_kv_heads,head_size", CASES)
def test_matches_padding_reference(seq_lens, block_size, num_kv_heads, head_size):
    """New block-aligned gather must be numerically identical to the old
    max-padding gather (same TND output, just less peak memory/bandwidth)."""
    num_blocks_total = sum((s + block_size - 1) // block_size for s in seq_lens) + 8
    key_cache, value_cache = _make_cache(num_blocks_total, block_size, num_kv_heads, head_size, seed=7)
    block_table = _block_table_for(seq_lens, block_size, num_blocks_total)

    new_k, new_v = _gather_paged_kv_to_dense(key_cache, value_cache, block_table, seq_lens, num_kv_heads, head_size)
    ref_k, ref_v = _ref_gather_padding(key_cache, value_cache, block_table, seq_lens, num_kv_heads, head_size)

    assert new_k.shape == ref_k.shape, f"key shape {new_k.shape} != ref {ref_k.shape}"
    assert torch.equal(new_k, ref_k), "key TND differs from padding reference"
    assert torch.equal(new_v, ref_v), "value TND differs from padding reference"


@pytest.mark.parametrize("seq_lens,block_size,num_kv_heads,head_size", CASES)
def test_output_shape_is_sum_seq_lens(seq_lens, block_size, num_kv_heads, head_size):
    """TND output must be (sum(seq_lens), num_kv_heads, head_size) — every
    in-block tail-padded token trimmed, no padding leaks into TND."""
    num_blocks_total = sum((s + block_size - 1) // block_size for s in seq_lens) + 8
    key_cache, value_cache = _make_cache(num_blocks_total, block_size, num_kv_heads, head_size, seed=11)
    block_table = _block_table_for(seq_lens, block_size, num_blocks_total)

    new_k, new_v = _gather_paged_kv_to_dense(key_cache, value_cache, block_table, seq_lens, num_kv_heads, head_size)

    expected_rows = sum(seq_lens)
    assert new_k.shape == (expected_rows, num_kv_heads, head_size)
    assert new_v.shape == (expected_rows, num_kv_heads, head_size)
    # contiguous TND (consumed by npu_fusion_attention)
    assert new_k.is_contiguous() and new_v.is_contiguous()


def test_bug_scenario_long_short_mix_no_padding_bloat():
    """The exact shape that triggered the OOM: one ~100K-token request plus
    short prompts. The old code materialised (num_seqs, max_seq_len, ...) =
    a huge padded tensor; the new code must output only sum(seq_lens) rows.

    Scaled-down (block/token counts) so it runs on CPU; the shape invariant is
    what matters — output rows == sum(seq_lens), independent of max(seq_lens).
    """
    seq_lens = [100000 // 128] + [5, 7, 9, 11]  # one long, four short
    # make the long one slightly non-block-aligned too
    seq_lens[0] = 100000 // 128 * 128 + 10
    block_size, num_kv_heads, head_size = 128, 4, 512
    num_blocks_total = sum((s + block_size - 1) // block_size for s in seq_lens) + 8
    key_cache, value_cache = _make_cache(num_blocks_total, block_size, num_kv_heads, head_size, seed=3)
    block_table = _block_table_for(seq_lens, block_size, num_blocks_total)

    new_k, new_v = _gather_paged_kv_to_dense(key_cache, value_cache, block_table, seq_lens, num_kv_heads, head_size)

    expected_rows = sum(seq_lens)
    # Output must NOT be padded to max(seq_lens) * num_seqs (the old bug).
    assert new_k.shape[0] == expected_rows
    assert new_k.shape[0] < max(seq_lens) * len(seq_lens), "output is still max-padded"


def test_trim_drops_in_block_tail_padding():
    """When seq_len is not a multiple of block_size, the tail block's padding
    tokens must be trimmed (not appear in TND). Verify with seq_len=100,
    block_size=128: only 100 rows, and they match the first 100 tokens of the
    seq's single block."""
    block_size, num_kv_heads, head_size = 128, 4, 256
    seq_lens = [100]
    key_cache = torch.zeros(2, block_size, num_kv_heads, head_size, dtype=torch.float32)
    key_cache[1] = 1.0  # block 1 is the seq's block
    value_cache = torch.full_like(key_cache, 7.0)
    block_table = torch.tensor([[1, 0]], dtype=torch.int32)  # seq0 -> block 1

    new_k, new_v = _gather_paged_kv_to_dense(key_cache, value_cache, block_table, seq_lens, num_kv_heads, head_size)

    assert new_k.shape == (100, num_kv_heads, head_size)
    assert torch.all(new_k == 1.0), "tail padding (tokens 100-127) leaked into output"
    assert torch.all(new_v == 7.0)


def test_acceptance_gather_reads_only_owned_blocks_not_max_padded(monkeypatch):
    """Acceptance test — guards against regression to the OOM bug.

    The OOM root cause was max-padding: ``index_select`` was called with
    ``num_seqs * ceil(max(seq_len)/block_size)`` block ids, reading ~Nx more
    blocks than needed (the padding blocks were read out then discarded by the
    mask — wasting both memory and bandwidth). Because the new and old
    implementations produce numerically identical TND output, output-only tests
    cannot detect a regression to max-padding.

    This test spies ``Tensor.index_select`` to assert the block-id index length
    equals the variable owned total ``sum(ceil(seq_len/block_size))``, proving
    the gather no longer materialises a max-padded intermediate. A regression to
    max-padding would make this fail (``seen == [max_padded, max_padded]``
    instead of ``[owned, owned]``).
    """
    # bug-shaped batch: one long request + several short ones
    seq_lens = [10000, 5, 7, 9, 11]
    block_size, num_kv_heads, head_size = 128, 4, 512
    num_blocks_total = sum((s + block_size - 1) // block_size for s in seq_lens) + 8
    key_cache, value_cache = _make_cache(num_blocks_total, block_size, num_kv_heads, head_size, seed=2)
    block_table = _block_table_for(seq_lens, block_size, num_blocks_total)

    owned = sum((s + block_size - 1) // block_size for s in seq_lens)
    max_padded = len(seq_lens) * ((max(seq_lens) + block_size - 1) // block_size)

    seen: list[int] = []
    orig_index_select = torch.Tensor.index_select

    def spy(self, dim, index):
        if dim == 0:
            seen.append(int(index.numel()))
        return orig_index_select(self, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", spy)

    _gather_paged_kv_to_dense(key_cache, value_cache, block_table, seq_lens, num_kv_heads, head_size)

    # key + value: two gathers, each reading exactly the owned block count.
    assert seen == [owned, owned], (
        f"index_select read {seen} blocks; expected [{owned}, {owned}] (owned). "
        f"A max-padding regression would read [{max_padded}, {max_padded}]."
    )
    # the bug scenario must show large padding waste, else this acceptance is moot
    assert max_padded > owned * 3, (
        f"scenario too uniform to validate (max_padded={max_padded}, owned={owned})"
    )
