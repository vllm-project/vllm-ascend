# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_kpool_tail_compress import (
    glm5_next_kpool_tail_compress_and_write_cache_triton as compress,
)

HEAD_DIM = 128
BF16_RTOL = 1e-2
BF16_ATOL = 1e-2


@pytest.mark.parametrize("pool,capacity", [(4, 4), (4, 12), (8, 8)])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("prefix_tokens", [0, 64])
@torch.inference_mode()
def test_tail_prefill_all_chunk_boundaries_decode_padding_and_graph(pool, capacity, use_graph, prefix_tokens):
    """Full absolute history is the oracle; it never reads the candidate ring."""
    generator = torch.Generator().manual_seed(137)
    dim, token_capacity, history_size = HEAD_DIM, 80, 512
    tail_backing = torch.full((12, 2, capacity + 3, dim + 2), -7.0, device="npu")
    tail = tail_backing[:, :, :capacity, :dim]
    cache_backing = torch.full((5, 130, 1, dim + 2), -7.0, dtype=torch.bfloat16, device="npu")
    cache = cache_backing[:, :128, :, :dim]
    expected_tail, expected_cache = tail_backing.cpu(), cache_backing.cpu()
    keys = torch.randn(3, history_size, dim, generator=generator)
    gates = torch.randn(3, history_size, dim, generator=generator) * 0.1
    ape = torch.randn(pool, dim, generator=generator) * 0.1
    tail_ids, cache_ids = [9, 5, 2], [3, 1, 4]
    # A prefix hit leaves a fresh ring; a pool-aligned resume must not read it.
    starts = [prefix_tokens] * 3
    graph, captured = None, None

    def run(lengths):
        nonlocal graph, captured
        positions, input_k, input_g, tail_slots, output_slots = [], [], [], [], []
        for req, (start, length) in enumerate(zip(starts, lengths)):
            for pos in range(start, start + length):
                positions.append(pos)
                input_k.append(keys[req, pos])
                input_g.append(gates[req, pos])
                tail_slots.append(tail_ids[req] * capacity + pos % capacity)
                slot = cache_ids[req] * 128 + pos // pool if (pos + 1) % pool == 0 else -1
                output_slots.append(slot)
                expected_tail[tail_ids[req], 0, pos % capacity, :dim] = keys[req, pos]
                expected_tail[tail_ids[req], 1, pos % capacity, :dim] = gates[req, pos]
                if slot >= 0:
                    begin = pos - pool + 1
                    pooled = (torch.softmax(gates[req, begin : pos + 1] + ape, dim=0) * keys[req, begin : pos + 1]).sum(
                        0
                    )
                    expected_cache[cache_ids[req], pos // pool, 0, :dim] = pooled.bfloat16()
        count = len(positions)
        assert count < token_capacity
        ends = torch.tensor(lengths, dtype=torch.int32).cumsum(0).int()
        k = torch.zeros(token_capacity, dim)
        g = torch.zeros_like(k)
        if count:
            k[:count], g[:count] = torch.stack(input_k), torch.stack(input_g)
        # Positive stale padding slots must never write into block zero.
        args = (
            tail,
            cache,
            k.npu(),
            g.npu(),
            ape.npu(),
            torch.tensor(positions + [0] * (token_capacity - count), device="npu"),
            ends.npu(),
            torch.tensor([s + n for s, n in zip(starts, lengths)], dtype=torch.int32).npu(),
            torch.tensor(tail_slots + [0] * (token_capacity - count), device="npu"),
            torch.tensor([[i] for i in tail_ids], dtype=torch.int32).npu(),
            torch.tensor(output_slots + [0] * (token_capacity - count), device="npu"),
            pool,
        )
        if use_graph:
            if graph is None:
                captured = args
                compress(*captured)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    compress(*captured)
            else:
                for target, value in zip(captured[2:-1], args[2:-1]):
                    target.copy_(value)
            graph.replay()
        else:
            compress(*args)
        torch.testing.assert_close(tail_backing.cpu(), expected_tail, rtol=0, atol=0)
        torch.testing.assert_close(cache_backing.cpu(), expected_cache, rtol=BF16_RTOL, atol=BF16_ATOL)
        for req, length in enumerate(lengths):
            starts[req] += length

    run([1, 0, 2])
    run([3 * pool + 1, 2 * pool + 1, pool + 1])
    for residue in range(pool):
        lengths = [(residue - start) % pool + 1 for start in starts]
        run(lengths)
        run([1, 1, 1])
    run([0, 0, 0])
    run([pool, pool + 1, pool - 1])


@pytest.mark.parametrize("empty", ["tokens", "requests"])
def test_empty_compression_preserves_caches(empty):
    tail = torch.ones(1, 2, 4, HEAD_DIM, device="npu")
    cache = torch.ones(1, 16, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    count = 0 if empty == "tokens" else 1
    keys = torch.zeros(count, HEAD_DIM, device="npu")
    ends = torch.tensor([count] if empty == "tokens" else [], dtype=torch.int32, device="npu")
    compress(
        tail,
        cache,
        keys,
        keys,
        torch.zeros(4, HEAD_DIM, device="npu"),
        torch.zeros(count, dtype=torch.int64, device="npu"),
        ends,
        ends,
        torch.full((count,), -1, device="npu"),
        torch.zeros(1, 1, dtype=torch.int32, device="npu"),
        torch.full((count,), -1, device="npu"),
        4,
    )
    torch.testing.assert_close(tail.cpu(), torch.ones_like(tail, device="cpu"), rtol=0, atol=0)
    torch.testing.assert_close(cache.cpu(), torch.ones_like(cache, device="cpu"), rtol=0, atol=0)


@torch.inference_mode()
def test_mtp_rejection_replay_preserves_open_pool_history():
    """Rejected lookahead must not overwrite history needed by replay.

    Pool 4-7 is verified while candidates 8 and 9 are also written.  Only
    position 6 is retained, so position 7 is replayed with a replacement key.
    A four-row ring would let rejected 8/9 overwrite committed 4/5; capacity
    pool+lookahead keeps 4/5/6 available for the replacement compression.
    """
    pool, capacity, dim = 4, 7, HEAD_DIM
    generator = torch.Generator().manual_seed(911)
    keys = torch.randn(10, dim, generator=generator)
    gates = torch.randn(10, dim, generator=generator) * 0.1
    replacement_k = torch.randn(dim, generator=generator)
    replacement_g = torch.randn(dim, generator=generator) * 0.1
    ape = torch.randn(pool, dim, generator=generator) * 0.1
    tail_block, indexer_block = 2, 1
    tail = torch.zeros(4, 2, capacity, dim, device="npu")
    cache = torch.zeros(3, 16, 1, dim, dtype=torch.bfloat16, device="npu")
    table = torch.tensor([[tail_block]], dtype=torch.int32, device="npu")

    def run(positions, current_k, current_g, output_slots):
        positions_cpu = torch.tensor(positions, dtype=torch.int64)
        tail_slots = tail_block * capacity + positions_cpu % capacity
        compress(
            tail,
            cache,
            current_k.npu(),
            current_g.npu(),
            ape.npu(),
            positions_cpu.npu(),
            torch.tensor([len(positions)], dtype=torch.int32, device="npu"),
            torch.tensor([positions[-1] + 1], dtype=torch.int32, device="npu"),
            tail_slots.npu(),
            table,
            torch.tensor(output_slots, dtype=torch.int64, device="npu"),
            pool,
        )

    # Committed incomplete history.
    run([4, 5], keys[4:6], gates[4:6], [-1, -1])
    # Target verification tentatively writes 6..9.  Acceptance later keeps 6
    # and rejects 7..9; the kernel intentionally does not know that yet.
    pooled_slot = indexer_block * cache.shape[1] + 1
    run([6, 7, 8, 9], keys[6:10], gates[6:10], [-1, pooled_slot, -1, -1])
    # Logical rollback replays position 7 with the target replacement token.
    run([7], replacement_k[None], replacement_g[None], [pooled_slot])
    torch.npu.synchronize()

    committed_k = torch.stack((keys[4], keys[5], keys[6], replacement_k))
    committed_g = torch.stack((gates[4], gates[5], gates[6], replacement_g))
    expected = (torch.softmax(committed_g + ape, dim=0) * committed_k).sum(0).bfloat16()
    torch.testing.assert_close(
        cache[indexer_block, 1, 0].cpu(),
        expected,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )


@pytest.mark.parametrize("shape", [(1, 4, HEAD_DIM), (1, 3, 4, HEAD_DIM), (1, 2, 2, HEAD_DIM)])
def test_invalid_tail_layout_raises(shape):
    tail = torch.zeros(shape, device="npu")
    cache = torch.zeros(1, 16, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    k = torch.zeros(1, HEAD_DIM, device="npu")
    ends = torch.ones(1, dtype=torch.int32, device="npu")
    slots = torch.zeros(1, dtype=torch.int64, device="npu")
    with pytest.raises(ValueError):
        compress(
            tail,
            cache,
            k,
            k,
            torch.zeros(4, HEAD_DIM, device="npu"),
            slots,
            ends,
            ends,
            slots,
            torch.zeros(1, 1, dtype=torch.int32, device="npu"),
            slots,
            4,
        )
