# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate CANN top-k on CP query shards against full-query CANN execution.

Run only while the benchmark services are stopped. This tests causal metadata,
not generated-text quality. Both sides share the same quantized K/V cache.
"""

from types import SimpleNamespace

import cann_ops_transformer  # noqa: F401
import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.attention.indexer_kpool as metadata_module
from vllm_ascend.ops.cann_keypool import quantize_keypool_query
from vllm_ascend.ops.triton.glm5_next_cann_keypool import pack_keypool_indices


@torch.inference_mode()
def check(lengths, history, pool=4, topk=2048, padded=0):
    torch.manual_seed(173)
    heads, dim, page = 32, 128, 128 // pool
    total = sum(lengths)
    size = total + padded
    assert size % 2 == 0
    raw = torch.tensor([h + n for h, n in zip(history, lengths)], dtype=torch.int64, device="npu")
    qlens = torch.tensor(lengths, dtype=torch.int64, device="npu")
    ends = qlens.cumsum(0)
    starts = torch.cat([ends.new_zeros(1), ends])
    positions = torch.tensor(
        [p for h, n in zip(history, lengths) for p in range(h, h + n)] + [0] * padded, dtype=torch.int64, device="npu"
    )
    max_pools = max(h + n for h, n in zip(history, lengths)) // pool
    pages = (max_pools + page - 1) // page
    table = torch.randperm(len(lengths) * pages, device="npu", dtype=torch.int32).reshape(len(lengths), pages)
    keys = torch.randn(len(lengths) * pages, page, 1, dim, device="npu")
    values_scale = keys.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 448
    # Production stores FP8 rows and FP32 scales in each padded cache page.
    page_bytes = page * (dim + 4) + 128
    backing = torch.zeros(len(lengths) * pages * page_bytes, dtype=torch.uint8, device="npu")
    cache = torch.as_strided(backing, keys.shape, (page_bytes, dim, dim, 1)).view(torch.float8_e4m3fn)
    scale = torch.as_strided(
        backing.view(torch.float32), (*keys.shape[:-1], 1), (page_bytes // 4, 1, 1, 1), storage_offset=page * dim // 4
    )
    cache.copy_((keys / values_scale).to(torch.float8_e4m3fn))
    scale.copy_(values_scale)
    q = torch.randn(size, heads, dim, device="npu", dtype=torch.bfloat16)
    weights = torch.rand(size, heads, device="npu", dtype=torch.bfloat16) / heads

    def run(query, head_weights, pos, cumulative, pool_lens, pool_tail):
        q8, qs = quantize_keypool_query(query)
        indices, _ = torch.ops.cann_ops_transformer.pool_key_indexer(
            q8,
            cache,
            head_weights,
            pool_tail,
            actual_seq_q=cumulative,
            actual_seq_k=pool_lens,
            block_table=table,
            q_descale=qs,
            k_descale=scale[..., 0],
            layout_q="TND",
            layout_k="PA_BBND",
            topk=topk,
            pool_size=pool,
            mask_mode=3,
            quant_mode=0,
        )
        meta = SimpleNamespace(positions=pos, cum_query_lens=cumulative)
        return pack_keypool_indices(indices, meta, topk, pool)

    reference = run(q[:total], weights[:total], positions[:total], ends, raw // pool, raw % pool).cpu()
    builder = object.__new__(metadata_module.AscendIndexerKPoolMetadataBuilder)
    builder.use_dsa_cp = True
    builder.dsa_cp_size = 2
    builder.compress_ratio = pool
    builder._max_num_metadata_reqs = len(lengths)
    builder._dsa_cp_query_buffers = {}
    builder.device = torch.device("npu")
    builder.logical_block_size = 128
    builder.kernel_row_block_size = page
    builder._max_num_batched_tokens = size
    builder._metadata_buffers = {}
    builder._captured_state_tables = {}
    builder._state_table_scratch = {}
    builder._state_table_shapes = {}
    common = SimpleNamespace(
        slot_mapping=torch.zeros(size, device="npu", dtype=torch.int64),
        positions=positions,
        num_input_tokens=size,
        block_table_tensor=table,
        num_reqs=len(lengths),
        num_actual_tokens=total,
        query_start_loc=starts.to(torch.int32),
        seq_lens=raw.to(torch.int32),
    )
    shards = []
    for rank in range(2):
        lo, hi = rank * (size // 2), (rank + 1) * (size // 2)
        builder.dsa_cp_rank = rank
        meta = builder.build(0, common).query_metadata
        actual = max(min(hi, total) - lo, 0)
        output = run(q[lo:hi], weights[lo:hi], positions[lo:hi], meta.cum_query_lens, meta.seq_lens, meta.pool_tail)
        output[actual:] = -1
        shards.append(output.cpu())
    output = torch.cat(shards)
    # Sorting removes top-k tie ordering while still checking the complete set.
    torch.testing.assert_close(output[:total].sort(-1).values, reference.sort(-1).values, rtol=0, atol=0)
    assert bool((output[total:] == -1).all())
    valid = output[:total, :, :] >= 0
    assert bool((output[:total][valid] <= positions[:total, None, None].cpu().expand_as(output[:total])[valid]).all())


@pytest.mark.skipif("950" not in torch.npu.get_device_name(0), reason="FP8 PoolKeyIndexer requires Ascend A5")
@pytest.mark.parametrize(
    "lengths,history,padded",
    [
        ([64], [0], 0),
        ([18, 14], [234, 513], 0),
        ([1, 1, 14, 2], [16, 0, 35, 1063], 0),
        ([20, 0, 12], [235929, 0, 235931], 0),
        ([2, 3], [7, 1021], 3),
        ([1], [235929], 1),
        ([128, 128], [235929, 245761], 0),
    ],
)
def test_cp_indexer_matches_full_cann_query(lengths, history, padded):
    check(lengths, history, padded=padded)


@pytest.mark.skipif("950" not in torch.npu.get_device_name(0), reason="FP8 CANN indexer requires Ascend A5")
@pytest.mark.parametrize("page_padding", [0, 128])
@pytest.mark.parametrize("capture", [False, True])
@torch.inference_mode()
def test_fp8_cann_writer_preserves_scale_planes_and_padding(page_padding, capture):
    from vllm_ascend.ops.triton.glm5_next_cann_keypool import write_keypool_caches

    dim, rows, blocks = 128, 32, 3
    page_bytes = rows * (dim + 4) + page_padding
    raw = torch.full((blocks * page_bytes,), 0x55, dtype=torch.uint8, device="npu")
    cache = torch.as_strided(raw, (blocks, rows, 1, dim), (page_bytes, dim, dim, 1)).view(torch.float8_e4m3fn)
    scales = torch.as_strided(
        raw.view(torch.float32),
        (blocks, rows, 1, 1),
        (page_bytes // 4, 1, 1, 1),
        storage_offset=rows * dim // 4,
    )
    host = torch.linspace(-4, 4, 3 * dim).reshape(3, dim).bfloat16()
    host[0].zero_()
    pooled = host.npu()
    tail = torch.full((1, 4, 2 * dim), 42.0, device="npu")
    empty = torch.empty((0, 4), dtype=torch.int64, device="npu")
    tail_metadata = SimpleNamespace(cann_inputs=SimpleNamespace(write_indices=empty, state_indices=empty, page_rows=4))
    metadata = SimpleNamespace(pooled_slot_mapping=torch.tensor([1, rows + 1, -1], dtype=torch.int64, device="npu"))
    state = torch.zeros((1, 4, 2 * dim), device="npu")

    def run():
        write_keypool_caches(tail, cache, None, None, pooled, metadata, tail_metadata, state, scale_cache=scales)

    run()  # Compile before capture.
    if capture:
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        host.mul_(2)
        pooled.copy_(host)
        graph.replay()
    torch.npu.synchronize()
    expected_scales = host[:2].float().abs().amax(-1).div(448)
    expected_scales[0] = 1.0
    mask = torch.zeros(blocks * page_bytes, dtype=torch.bool)
    for block in range(2):
        actual_scale = scales[block, 1, 0, 0].cpu()
        torch.testing.assert_close(actual_scale, expected_scales[block], rtol=1e-6, atol=0)
        actual = cache[block, 1, 0].float().cpu() * actual_scale
        torch.testing.assert_close(actual, host[block].float(), rtol=0, atol=float(host[block].abs().max()) * 0.07)
        offset = block * page_bytes + dim
        mask[offset : offset + dim] = True
        offset = block * page_bytes + rows * dim + 4
        mask[offset : offset + 4] = True
    assert bool((raw.cpu()[~mask] == 0x55).all())
    assert bool((tail == 42).all())
