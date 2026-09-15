# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_kpool_state_compress import glm5_next_kpool_state_compress_and_write_cache_triton

HEAD_DIM = 128
BF16_RTOL = 1e-2
BF16_ATOL = 1e-2


@pytest.mark.parametrize("pool,capacity", [(4, 4), (4, 16), (8, 8)])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_paged_state_long_prefill_padding_and_rollback(pool, capacity, use_graph):
    generator = torch.Generator().manual_seed(13)
    dim = HEAD_DIM
    # Physical pages deliberately have padding and request IDs are reordered.
    state_backing = torch.full((32, capacity + 2, 2 * dim), -7.0, device="npu")
    cache_backing = torch.full((3, 18, 1, dim), -7.0, dtype=torch.bfloat16, device="npu")
    state, cache = state_backing[:, :capacity], cache_backing[:, :16]
    expected_state_backing, expected_cache_backing = state_backing.cpu(), cache_backing.cpu()
    expected_state, expected_cache = expected_state_backing[:, :capacity], expected_cache_backing[:, :16]
    ape = torch.randn(pool, dim, generator=generator) * 0.1
    state_table = torch.tensor([[5, 9, 3, 12, 7, 16, 2, 20], [11, 8, 1, 18, 4, 22, 15, 23]], dtype=torch.int32)
    cache_blocks = [1, 2]
    history: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [{}, {}]
    graph, captured_args = None, None
    graph_capacity = 32

    def run(starts, lengths, invalidate=False):
        nonlocal graph, captured_args
        positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lengths)])
        num_tokens = positions.numel()
        keys = torch.randn(num_tokens, dim, generator=generator)
        gates = torch.randn(num_tokens, dim, generator=generator) * 0.1
        ends = torch.tensor(lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
        seq_lens = torch.tensor([s + n for s, n in zip(starts, lengths)], dtype=torch.int32)
        state_slots, indexer_slots = [], []
        cursor = 0
        for req, (start, length) in enumerate(zip(starts, lengths)):
            for local in range(length):
                row, pos = cursor + local, start + local
                history[req][pos] = (keys[row], gates[row])
                state_slots.append(int(state_table[req, pos // capacity]) * capacity + pos % capacity)
                slot = cache_blocks[req] * 16 + pos // pool if (pos + 1) % pool == 0 else -1
                indexer_slots.append(slot)
                if slot >= 0:
                    window = [history[req][p] for p in range(pos - pool + 1, pos + 1)]
                    pooled = (
                        torch.softmax(torch.stack([g for _, g in window]) + ape, dim=0)
                        * torch.stack([k for k, _ in window])
                    ).sum(0)
                    expected_cache[cache_blocks[req], pos // pool, 0] = pooled.bfloat16()
            cursor += length
        if invalidate:
            state_slots[0] = -1
            state_slots[1] = state.shape[0] * capacity
            indexer_slots[0] = cache.shape[0] * cache.shape[1]
        # Each logical position has its own scheduler-provided physical slot.
        cursor = 0
        for req, (start, length) in enumerate(zip(starts, lengths)):
            for local in range(length):
                row, pos = cursor + local, start + local
                if 0 <= state_slots[row] < state.shape[0] * capacity:
                    expected_state[state_slots[row] // capacity, pos % capacity] = torch.cat((keys[row], gates[row]))
            cursor += length
        # Graph-capacity rows have no owning request, even if a stale slot is positive.
        padded_keys = torch.nn.functional.pad(keys, (0, 0, 0, graph_capacity - num_tokens)).npu()
        padded_gates = torch.nn.functional.pad(gates, (0, 0, 0, graph_capacity - num_tokens)).npu()
        args = (
            state,
            cache,
            padded_keys,
            padded_gates,
            ape.npu(),
            torch.cat((positions, torch.zeros(graph_capacity - num_tokens))).long().npu(),
            ends.npu(),
            seq_lens.npu(),
            torch.tensor(state_slots + [0] + [-1] * (graph_capacity - num_tokens - 1), device="npu"),
            state_table.npu(),
            torch.tensor(indexer_slots + [0] + [-1] * (graph_capacity - num_tokens - 1), device="npu"),
            pool,
        )
        if use_graph:
            if graph is None:
                captured_args = args
                glm5_next_kpool_state_compress_and_write_cache_triton(*args)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    glm5_next_kpool_state_compress_and_write_cache_triton(*captured_args)
            else:
                for target, value in zip(captured_args[2:-1], args[2:-1]):
                    target.copy_(value)
            graph.replay()
        else:
            glm5_next_kpool_state_compress_and_write_cache_triton(*args)
        torch.testing.assert_close(state_backing.cpu(), expected_state_backing, rtol=0, atol=0)
        torch.testing.assert_close(cache_backing.cpu(), expected_cache_backing, rtol=BF16_RTOL, atol=BF16_ATOL)

    run([0, 0], [2, 3], invalidate=True)
    run([0, 3], [2, 13])  # Rewrite invalid req0 slots; req1 reads history before a long prefill.
    run([2, 16], [3, 3])  # Verification writes future candidates into separate paged slots.
    run([3, 17], [1, 3])  # Reject candidates and overwrite their positions.
    state_table[1, : 16 // capacity] = -1  # Evicted old pages must not be consulted.
    run([4, 20], [pool, pool])


@pytest.mark.parametrize("empty", ["tokens", "requests"])
def test_empty_compression_preserves_caches(empty):
    state = torch.ones(1, 4, 2 * HEAD_DIM, device="npu")
    cache = torch.ones(1, 16, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    count = 0 if empty == "tokens" else 1
    keys = torch.zeros(count, HEAD_DIM, device="npu")
    ends = torch.tensor([count] if empty == "tokens" else [], dtype=torch.int32, device="npu")
    glm5_next_kpool_state_compress_and_write_cache_triton(
        state,
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
    torch.testing.assert_close(state.cpu(), torch.ones_like(state, device="cpu"), rtol=0, atol=0)
    torch.testing.assert_close(cache.cpu(), torch.ones_like(cache, device="cpu"), rtol=0, atol=0)


@pytest.mark.parametrize("block_size,pages", [(2, 1), (4, 0)])
def test_invalid_state_layout_raises(block_size, pages):
    state = torch.zeros(1, block_size, 2 * HEAD_DIM, device="npu")
    cache = torch.zeros(1, 16, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    keys = torch.zeros(1, HEAD_DIM, device="npu")
    ends = torch.ones(1, dtype=torch.int32, device="npu")
    slots = torch.zeros(1, dtype=torch.int64, device="npu")
    with pytest.raises(ValueError, match="nonempty state page table and block size >= pool size"):
        glm5_next_kpool_state_compress_and_write_cache_triton(
            state,
            cache,
            keys,
            keys,
            torch.zeros(4, HEAD_DIM, device="npu"),
            slots,
            ends,
            ends,
            slots,
            torch.zeros(1, pages, dtype=torch.int32, device="npu"),
            slots,
            4,
        )
