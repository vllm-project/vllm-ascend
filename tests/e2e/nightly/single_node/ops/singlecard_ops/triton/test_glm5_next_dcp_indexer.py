# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3/A5 pooled DCP kernels against an independent full-history reference."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile  # noqa: E402
from vllm_ascend.ops.triton.glm5_next_dcp_indexer import (  # noqa: E402
    glm5_next_dcp_indexer,
    select_local_pool_candidates,
)
from vllm_ascend.ops.triton.sharded_indexer import pack_local_candidates  # noqa: E402

POOL, BLOCK, DIM, MAX_LEN = 4, 32, 128, 4096


def make_inputs(world, rank, interleave, quantized=False, step=0):
    """Exact arithmetic, negative scores, ties, permuted pages and padded rows."""
    capacity = (MAX_LEN + world * interleave - 1) // (world * interleave) * (interleave // POOL)
    pages = (capacity + BLOCK - 1) // BLOCK
    generator = torch.Generator().manual_seed(20883 + step)
    table = torch.randperm(2 * pages, generator=generator).view(2, pages).int() + 1
    # Exercise a table whose columns are a strided view.
    table_owner = torch.zeros(2, pages * 2, dtype=torch.int32)
    table_owner[:, ::2] = table
    cache = torch.zeros(2 * pages + 1, BLOCK, 1, DIM)
    scales = torch.ones(2 * pages + 1, BLOCK, 1)
    for req in range(2):
        for local in range(capacity):
            global_pool = (local // (interleave // POOL) * world + rank) * (interleave // POOL)
            global_pool += local % (interleave // POOL)
            page, slot = int(table[req, local // BLOCK]), local % BLOCK
            cache[page, slot, 0, :2] = pool_key(req, global_pool, step)
            if quantized:
                scales[page, slot] = 0.5 if global_pool % 2 else 0.25
                cache[page, slot] /= scales[page, slot]
    cache = cache.to(torch.float8_e4m3fn if quantized else torch.bfloat16)
    query = torch.zeros(9, 4, DIM, dtype=torch.bfloat16)
    query[:, :, :2] = torch.randint(-3, 4, (9, 4, 2), generator=generator).to(torch.bfloat16)
    # Include a zero-weight row to force tied cutoffs across owners.
    weights = torch.tensor([[1, -2, 3, -1]], dtype=torch.bfloat16).expand(9, 4).clone()
    weights[2].zero_()
    # Advance then roll back a draft across both pool and owner boundaries.
    offset = [0, 4, 0][step]
    positions = torch.tensor([0, 1, 2, 3, 4, 1023, 1024, 1025, 9999], dtype=torch.int64) + offset
    metadata = SimpleNamespace(
        positions=positions,
        cum_query_lens=torch.tensor([5, 8], dtype=torch.int32),
        seq_lens=torch.tensor([(5 + offset) // POOL, (1026 + offset) // POOL], dtype=torch.int32),
        block_table=table_owner[:, ::2],
    )
    return query, cache, weights, scales if quantized else None, metadata, capacity


def pool_key(req, pool, step):
    return torch.tensor([(pool * 13 + req * 7 + step * 3) % 31 - 15, (pool * 3 + req + step) % 17 - 8])


def reference(query, weights, metadata, topk, step):
    output = torch.full((query.shape[0], 1, topk + POOL - 1), -1, dtype=torch.int32)
    for row in range(int(metadata.cum_query_lens[-1])):
        req = 0 if row < int(metadata.cum_query_lens[0]) else 1
        visible = min((int(metadata.positions[row]) + 1) // POOL, int(metadata.seq_lens[req]))
        qbar = (query[row].double() * weights[row].double()[:, None]).sum(0)
        keys = torch.stack([pool_key(req, pool, step) for pool in range(visible)]) if visible else torch.empty(0, 2)
        scores = keys.double() @ qbar[:2]
        selected = scores.argsort(descending=True, stable=True)[: topk // POOL]
        tokens = (selected[:, None] * POOL + torch.arange(POOL)).flatten().int()
        output[row, 0, : tokens.numel()] = tokens
        tail_start = (int(metadata.positions[row]) + 1) // POOL * POOL
        tail = torch.arange(tail_start, int(metadata.positions[row]) + 1, dtype=torch.int32)
        column = min(tail_start, topk)
        output[row, 0, column : column + tail.numel()] = tail
    return output


def to_device(inputs):
    query, cache, weights, scales, metadata, capacity = inputs
    # Keep both non-contiguous column stride and stable storage on replay.
    table_owner = torch.zeros(metadata.block_table.shape[0], metadata.block_table.shape[1] * 2, dtype=torch.int32).npu()
    table_owner[:, ::2].copy_(metadata.block_table)
    device_metadata = SimpleNamespace(
        positions=metadata.positions.npu(),
        cum_query_lens=metadata.cum_query_lens.npu(),
        seq_lens=metadata.seq_lens.npu(),
        block_table=table_owner[:, ::2],
    )
    return (
        query.npu(),
        cache.npu(),
        weights.npu(),
        scales.npu() if scales is not None else None,
        device_metadata,
        capacity,
    )


def update_inputs(device, host):
    for index in (0, 1, 2, 3):
        if device[index] is not None:
            device[index].copy_(host[index])
    for name in ("positions", "cum_query_lens", "seq_lens", "block_table"):
        getattr(device[4], name).copy_(getattr(host[4], name))


def require_dtype(quantized):
    if quantized and get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED:
        pytest.skip("FP8 indexer cache requires A5")


@pytest.mark.parametrize("world,interleave,topk", [(2, 4, 32), (4, 128, 2048), (8, 512, 2048)])
@pytest.mark.parametrize("quantized", [False, True])
def test_sharded_pool_selection_and_graph_replay(world, interleave, topk, quantized):
    torch.npu.set_device(0)
    require_dtype(quantized)
    hosts = [make_inputs(world, r, interleave, quantized) for r in range(world)]
    devices = [to_device(host) for host in hosts]
    storage = torch.full((world, 11, topk + POOL + 7), -123, dtype=torch.int32, device="npu")

    def run():
        packs = []
        for rank, (query, cache, weights, scales, metadata, capacity) in enumerate(devices):
            indices, scores = select_local_pool_candidates(
                query,
                cache,
                weights,
                metadata,
                world_size=world,
                rank=rank,
                interleave=interleave // POOL,
                pool_size=POOL,
                pool_topk=topk // POOL,
                capacity=capacity,
                scale_cache=scales,
            )
            packs.append(pack_local_candidates(indices, scores, world, rank, interleave // POOL))
        gathered = torch.cat(packs)
        results = []
        for rank, (query, cache, weights, scales, metadata, _) in enumerate(devices):
            group = SimpleNamespace(world_size=world, rank_in_group=rank, all_gather=lambda packed, dim: gathered)
            results.append(
                glm5_next_dcp_indexer(
                    query,
                    cache,
                    weights,
                    metadata,
                    group,
                    topk=topk,
                    pool_size=POOL,
                    interleave=interleave,
                    max_model_len=MAX_LEN,
                    output_buffer=storage[rank, 1:-1],
                    scale_cache=scales,
                )
            )
        return results

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        results = run()
    for step in range(3):
        hosts = [make_inputs(world, r, interleave, quantized, step) for r in range(world)]
        for device, host in zip(devices, hosts):
            update_inputs(device, host)
        graph.replay()
        query, _, weights, _, metadata, _ = hosts[0]
        expected = reference(query, weights, metadata, topk, step)
        for result in results:
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
        assert (storage[:, [0, -1]].cpu() == -123).all()
        assert (storage[:, 1:-1, topk + POOL - 1 :].cpu() == -1).all()
    graph.reset()
