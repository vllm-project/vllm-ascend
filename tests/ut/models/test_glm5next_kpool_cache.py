# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM-5.3-Flash kpool indexer's cache write path."""

import pytest
import torch

from vllm_ascend.models.glm5next.ops.kpool_cache import write_decode, write_prefill
from vllm_ascend.models.glm5next.ops.kpool_compress import compress_pool

HEAD_DIM = 128
POOL_SIZE = 4
POOLS_PER_BLOCK = 2
NUM_BLOCKS = 16
# How far past its own block unification pads a page in the padded fixtures.
PAGE_PAD_FACTOR = 8


def _caches(dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """An index-K cache and a tail ring, both filled with a recognizable value.

    Starting from a non-zero fill means a slot that should have been left alone
    can be told apart from one that was written with zeros.
    """
    index_cache = torch.full((NUM_BLOCKS, POOLS_PER_BLOCK, HEAD_DIM), 7.0, dtype=dtype)
    tail_cache = torch.full((NUM_BLOCKS, 2, POOL_SIZE, HEAD_DIM), 7.0, dtype=dtype)
    return index_cache, tail_cache


def _pool_slot(position: int, block_table: list[int]) -> int:
    """The pool-granular slot a completing token writes to.

    ``tokens_per_state`` makes vLLM emit a slot only when a pool completes, so
    every other token maps to -1 on both the prefill and the decode path.
    """
    if position % POOL_SIZE != POOL_SIZE - 1:
        return -1
    pool = position // POOL_SIZE
    return block_table[pool // POOLS_PER_BLOCK] * POOLS_PER_BLOCK + pool % POOLS_PER_BLOCK


def _slot_mappings(positions: list[int], block_table: list[int], tail_block: int) -> tuple[torch.Tensor, torch.Tensor]:
    pool_slots = torch.tensor([_pool_slot(p, block_table) for p in positions], dtype=torch.int32)
    tail_slots = torch.tensor([tail_block * POOL_SIZE + p % POOL_SIZE for p in positions], dtype=torch.int32)
    return pool_slots, tail_slots


def _assert_pools_match(actual: torch.Tensor, expected: torch.Tensor, msg: str = "") -> None:
    """Compare every pool slot outside the null block.

    Block 0 is vLLM's reserved null block and is where the write path dumps the
    writes it has to skip, so its contents are deliberately undefined.
    """
    torch.testing.assert_close(actual[1:], expected[1:], msg=msg or None)


def _reference_index_cache(
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    position_bias: torch.Tensor,
    block_table: list[int],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compress a whole sequence pool by pool, reading positions on the host."""
    expected = torch.full((NUM_BLOCKS, POOLS_PER_BLOCK, HEAD_DIM), 7.0, dtype=dtype)
    entries = expected.view(-1, HEAD_DIM)
    for position in range(keys.shape[0]):
        slot = _pool_slot(position, block_table)
        if slot < 0:
            continue
        members = slice(position - POOL_SIZE + 1, position + 1)
        entries[slot] = compress_pool(keys[members], gate_scores[members], position_bias)
    return expected


@pytest.fixture
def sequence() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    length = 40
    keys = torch.randn(length, HEAD_DIM, dtype=torch.bfloat16)
    gate_scores = torch.randn(length, HEAD_DIM, dtype=torch.bfloat16)
    position_bias = torch.randn(POOL_SIZE, HEAD_DIM)
    return keys, gate_scores, position_bias


def test_prefill_compresses_every_complete_pool(sequence):
    keys, gate_scores, position_bias = sequence
    block_table = [3, 5, 8, 11, 12]
    index_cache, tail_cache = _caches()
    pool_slots, tail_slots = _slot_mappings(list(range(keys.shape[0])), block_table, tail_block=1)

    write_prefill(
        index_cache,
        tail_cache,
        keys,
        gate_scores,
        position_bias,
        pool_slots=pool_slots,
        tail_slots=tail_slots,
        pool_size=POOL_SIZE,
        pools_per_block=POOLS_PER_BLOCK,
    )

    expected = _reference_index_cache(keys, gate_scores, position_bias, block_table)
    _assert_pools_match(index_cache, expected)


def test_prefill_seeds_the_ring_with_the_last_pool_of_tokens(sequence):
    """Seeding writes each request's last ``pool_size`` tokens at their phase.

    That is a superset of the incomplete trailing pool -- with 38 tokens it also
    re-seeds positions 34 and 35, which belong to the already-compressed pool 8.
    Harmless: pool 9 completes at position 39, by which point positions 38 and
    39 have overwritten those two slots.
    """
    keys, gate_scores, position_bias = sequence
    # 38 tokens leaves positions 36 and 37 in an incomplete trailing pool.
    keys, gate_scores = keys[:38], gate_scores[:38]
    index_cache, tail_cache = _caches()
    pool_slots, tail_slots = _slot_mappings(list(range(38)), [3, 5, 8, 11, 12], tail_block=1)

    write_prefill(
        index_cache,
        tail_cache,
        keys,
        gate_scores,
        position_bias,
        pool_slots=pool_slots,
        tail_slots=tail_slots,
        pool_size=POOL_SIZE,
        pools_per_block=POOLS_PER_BLOCK,
    )

    ring = tail_cache[1]
    for position in range(38 - POOL_SIZE, 38):
        phase = position % POOL_SIZE
        torch.testing.assert_close(ring[0, phase], keys[position], msg=f"key at {position}")
        torch.testing.assert_close(ring[1, phase], gate_scores[position], msg=f"gate at {position}")

    # Block 1 is this request's ring and block 0 is the null block the skipped
    # writes are dumped into; no other request's ring was touched.
    assert torch.all(tail_cache[2:] == 7.0)


def test_prefill_leaves_the_cache_alone_when_no_pool_completes(sequence):
    keys, gate_scores, position_bias = sequence
    index_cache, tail_cache = _caches()
    pool_slots, tail_slots = _slot_mappings([0, 1], [3], tail_block=1)

    write_prefill(
        index_cache,
        tail_cache,
        keys[:2],
        gate_scores[:2],
        position_bias,
        pool_slots=pool_slots,
        tail_slots=tail_slots,
        pool_size=POOL_SIZE,
        pools_per_block=POOLS_PER_BLOCK,
    )

    assert torch.all(index_cache[1:] == 7.0)


@pytest.mark.parametrize("prefill_length", [16, 20, 24])
@pytest.mark.parametrize("tokens_per_step", [1, 2, 4, 5])
def test_decode_through_the_ring_matches_a_single_prefill(sequence, prefill_length, tokens_per_step):
    """The whole point of the ring: pools that straddle steps still compress.

    Prefilling a prefix and then decoding the rest must leave the index cache
    bit-identical to prefilling the entire sequence at once, whatever the step
    size -- including step sizes that are not a whole number of pools, where a
    pool completes in the middle of a draft-verify batch.
    """
    keys, gate_scores, position_bias = sequence
    length = keys.shape[0]
    block_table = [3, 5, 8, 11, 12]
    index_cache, tail_cache = _caches()

    pool_slots, tail_slots = _slot_mappings(list(range(prefill_length)), block_table, tail_block=1)
    write_prefill(
        index_cache,
        tail_cache,
        keys[:prefill_length],
        gate_scores[:prefill_length],
        position_bias,
        pool_slots=pool_slots,
        tail_slots=tail_slots,
        pool_size=POOL_SIZE,
        pools_per_block=POOLS_PER_BLOCK,
    )

    for start in range(prefill_length, length, tokens_per_step):
        positions = list(range(start, min(start + tokens_per_step, length)))
        step_pool_slots, step_tail_slots = _slot_mappings(positions, block_table, tail_block=1)
        write_decode(
            index_cache,
            tail_cache,
            keys[positions].unsqueeze(0),
            gate_scores[positions].unsqueeze(0),
            position_bias,
            pool_slots=step_pool_slots.unsqueeze(0),
            tail_slots=step_tail_slots.unsqueeze(0),
            positions=torch.tensor(positions, dtype=torch.int32).unsqueeze(0),
            pool_size=POOL_SIZE,
            pools_per_block=POOLS_PER_BLOCK,
        )

    expected = _reference_index_cache(keys, gate_scores, position_bias, block_table)
    _assert_pools_match(index_cache, expected)


def _padded_caches(dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """Caches whose pages unification padded out past the block they hold.

    The indexer page grows along its pool axis and the ring page along its
    state axis, which is how vLLM folds ``page_size_padded`` into the shape.
    """
    index_cache = torch.full((NUM_BLOCKS, POOLS_PER_BLOCK * PAGE_PAD_FACTOR, HEAD_DIM), 7.0, dtype=dtype)
    tail_cache = torch.full((NUM_BLOCKS, 2, POOL_SIZE * PAGE_PAD_FACTOR, HEAD_DIM), 7.0, dtype=dtype)
    return index_cache, tail_cache


def test_padding_a_page_does_not_move_where_a_write_lands(sequence):
    """A slot is numbered by the spec's block size, not the padded page width.

    Unifying page sizes across the cache groups pads both of these pages out to
    the much wider MLA one, so a page carries its block's entries at the front
    and padding behind them. Indexing the allocation flat instead would walk
    pages at the unpadded width and land every write in a page nothing reads --
    which no operator rejects, so the only symptom would be worse answers.
    """
    keys, gate_scores, position_bias = sequence
    block_table = [3, 5, 8, 11, 12]
    length = keys.shape[0]
    prefill_length = 20

    plain, padded = _caches(), _padded_caches()
    for index_cache, tail_cache in (plain, padded):
        pool_slots, tail_slots = _slot_mappings(list(range(prefill_length)), block_table, tail_block=1)
        write_prefill(
            index_cache,
            tail_cache,
            keys[:prefill_length],
            gate_scores[:prefill_length],
            position_bias,
            pool_slots=pool_slots,
            tail_slots=tail_slots,
            pool_size=POOL_SIZE,
            pools_per_block=POOLS_PER_BLOCK,
        )
        # Decode too: it reads the ring back to finish a boundary pool, so a
        # misread page would corrupt exactly the pools that span two steps.
        for position in range(prefill_length, length):
            step_pool_slots, step_tail_slots = _slot_mappings([position], block_table, tail_block=1)
            write_decode(
                index_cache,
                tail_cache,
                keys[position].view(1, 1, HEAD_DIM),
                gate_scores[position].view(1, 1, HEAD_DIM),
                position_bias,
                pool_slots=step_pool_slots.view(1, 1),
                tail_slots=step_tail_slots.view(1, 1),
                positions=torch.tensor([[position]], dtype=torch.int32),
                pool_size=POOL_SIZE,
                pools_per_block=POOLS_PER_BLOCK,
            )

    _assert_pools_match(padded[0][:, :POOLS_PER_BLOCK], plain[0])
    torch.testing.assert_close(padded[1][1:, :, :POOL_SIZE], plain[1][1:])
    # And the padding itself is never touched.
    assert (padded[0][:, POOLS_PER_BLOCK:] == 7.0).all()
    assert (padded[1][:, :, POOL_SIZE:] == 7.0).all()

    # The whole point: this is the answer an unpadded page produces.
    _assert_pools_match(
        padded[0][:, :POOLS_PER_BLOCK], _reference_index_cache(keys, gate_scores, position_bias, block_table)
    )


def test_decode_keeps_concurrent_requests_apart(sequence):
    """Two requests at different pool phases, stepped together in one batch."""
    keys, gate_scores, position_bias = sequence
    block_tables = [[3, 5, 8], [11, 12, 14]]
    tail_blocks = [1, 2]
    # Different prefill lengths put the two requests in different pool phases,
    # so a step completes a pool for one request and not the other.
    prefill_lengths = [16, 18]

    index_cache, tail_cache = _caches()
    for request, (block_table, tail_block, prefill_length) in enumerate(
        zip(block_tables, tail_blocks, prefill_lengths)
    ):
        pool_slots, tail_slots = _slot_mappings(list(range(prefill_length)), block_table, tail_block)
        write_prefill(
            index_cache,
            tail_cache,
            keys[:prefill_length] + request,
            gate_scores[:prefill_length],
            position_bias,
            pool_slots=pool_slots,
            tail_slots=tail_slots,
            pool_size=POOL_SIZE,
            pools_per_block=POOLS_PER_BLOCK,
        )

    for step in range(8):
        positions = [prefill_lengths[0] + step, prefill_lengths[1] + step]
        batch_pool_slots = torch.stack(
            [_slot_mappings([p], bt, tb)[0] for p, bt, tb in zip(positions, block_tables, tail_blocks)]
        )
        batch_tail_slots = torch.stack(
            [_slot_mappings([p], bt, tb)[1] for p, bt, tb in zip(positions, block_tables, tail_blocks)]
        )
        write_decode(
            index_cache,
            tail_cache,
            torch.stack([keys[p] + r for r, p in enumerate(positions)]).unsqueeze(1),
            torch.stack([gate_scores[p] for p in positions]).unsqueeze(1),
            position_bias,
            pool_slots=batch_pool_slots,
            tail_slots=batch_tail_slots,
            positions=torch.tensor(positions, dtype=torch.int32).unsqueeze(1),
            pool_size=POOL_SIZE,
            pools_per_block=POOLS_PER_BLOCK,
        )

    for request, block_table in enumerate(block_tables):
        expected = _reference_index_cache(
            keys[: prefill_lengths[request] + 8] + request,
            gate_scores[: prefill_lengths[request] + 8],
            position_bias,
            block_table,
        )
        for pool in range((prefill_lengths[request] + 8) // POOL_SIZE):
            block = block_table[pool // POOLS_PER_BLOCK]
            offset = pool % POOLS_PER_BLOCK
            torch.testing.assert_close(
                index_cache[block, offset],
                expected[block, offset],
                msg=f"request {request} pool {pool}",
            )


def test_decode_ignores_padded_requests(sequence):
    """A padded row must not disturb a real request sharing the batch."""
    keys, gate_scores, position_bias = sequence
    block_table = [3, 5, 8]
    index_cache, tail_cache = _caches()
    pool_slots, tail_slots = _slot_mappings(list(range(16)), block_table, tail_block=1)
    write_prefill(
        index_cache,
        tail_cache,
        keys[:16],
        gate_scores[:16],
        position_bias,
        pool_slots=pool_slots,
        tail_slots=tail_slots,
        pool_size=POOL_SIZE,
        pools_per_block=POOLS_PER_BLOCK,
    )
    before = index_cache.clone()

    real_pool_slots, real_tail_slots = _slot_mappings([19], block_table, tail_block=1)
    for step, position in enumerate(range(16, 20)):
        step_pool_slots, step_tail_slots = _slot_mappings([position], block_table, tail_block=1)
        write_decode(
            index_cache,
            tail_cache,
            torch.stack([keys[position], keys[0]]).unsqueeze(1),
            torch.stack([gate_scores[position], gate_scores[0]]).unsqueeze(1),
            position_bias,
            # The padded row carries position 0 and a negative slot mapping,
            # which is how vLLM pads a decode batch.
            pool_slots=torch.tensor([[step_pool_slots[0]], [-1]], dtype=torch.int32),
            tail_slots=torch.tensor([[step_tail_slots[0]], [-1]], dtype=torch.int32),
            positions=torch.tensor([[position], [0]], dtype=torch.int32),
            pool_size=POOL_SIZE,
            pools_per_block=POOLS_PER_BLOCK,
        )

    # Only the pool ending at position 19 changed; the padded row wrote nothing.
    written = int(real_pool_slots[0])
    changed = (index_cache.view(-1, HEAD_DIM) != before.view(-1, HEAD_DIM)).any(dim=-1)
    # Slot 0 is the reserved null block, where skipped writes are dumped.
    changed[0] = False
    assert changed.nonzero().flatten().tolist() == [written]


def test_writes_are_free_of_host_reads(sequence, monkeypatch):
    """A host read here would abort ACL-graph capture on the decode path."""
    keys, gate_scores, position_bias = sequence

    def fail(self, *args, **kwargs):
        raise AssertionError("host read on the kpool cache write path")

    monkeypatch.setattr(torch.Tensor, "item", fail)
    monkeypatch.setattr(torch.Tensor, "tolist", fail)

    index_cache, tail_cache = _caches()
    write_decode(
        index_cache,
        tail_cache,
        keys[:8].reshape(2, 4, HEAD_DIM),
        gate_scores[:8].reshape(2, 4, HEAD_DIM),
        position_bias,
        pool_slots=torch.tensor([[-1, -1, -1, 4], [-1, -1, -1, 6]], dtype=torch.int32),
        tail_slots=torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.int32),
        positions=torch.tensor([[16, 17, 18, 19], [20, 21, 22, 23]], dtype=torch.int32),
        pool_size=POOL_SIZE,
        pools_per_block=POOLS_PER_BLOCK,
    )
