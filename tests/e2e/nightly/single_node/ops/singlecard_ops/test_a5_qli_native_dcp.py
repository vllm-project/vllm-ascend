# SPDX-License-Identifier: Apache-2.0
"""Native A5 DCP visibility and graph-padding contract; requires rebuilt extension."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def native_extension():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an A5 NPU")
    if "950" not in torch.npu.get_device_name(0):
        pytest.skip("Requires an A5 NPU")
    package = Path(next(iter(importlib.util.find_spec("vllm_ascend").submodule_search_locations)))
    torch.ops.load_library(str(next(package.glob("vllm_ascend_C*.so"))))
    # Match platform startup order before importing the device adaptor.
    import vllm_ascend.ops  # noqa: F401


@pytest.mark.parametrize("interleave", [1, 8, 768])
@pytest.mark.parametrize("rank", [0, 7])
def test_native_dcp_visibility_padding_and_changed_length_replay(interleave, rank):
    from vllm_ascend.ops.triton.sharded_indexer import build_local_key_lengths

    torch.manual_seed(42)
    world, heads, dim, page, rows, active, topk = 8, 32, 128, 128, 8, 4, 2048
    local_length = max(page, 2 * interleave)
    blocks = (local_length + page - 1) // page
    query = torch.randn(rows, heads, dim).to(torch.float8_e4m3fn).npu()
    # Key and scales use a shared physical page with non-contiguous block strides.
    page_bytes = page * (dim + 4)
    storage = torch.zeros(blocks * page_bytes, dtype=torch.uint8, device="npu")
    key = storage.view(query.dtype).as_strided((blocks, page, 1, dim), (page_bytes, dim, dim, 1))
    key.copy_(torch.randn(key.shape).to(query.dtype))
    scales = storage.view(torch.float32).as_strided((blocks, page, 1), (page_bytes // 4, 1, 1), page * dim // 4)
    scales.fill_(0.1)
    weights = torch.ones(rows, heads, device="npu")
    query_scale = torch.full_like(weights, 0.1)
    starts = torch.tensor([0, active], dtype=torch.int32, device="npu")
    local = torch.tensor([local_length], dtype=torch.int32, device="npu")
    global_len = torch.tensor([world * interleave + active], dtype=torch.int32, device="npu")
    table = torch.arange(blocks, dtype=torch.int32, device="npu").view(1, -1)
    dcp = dict(
        enable_dcp=True,
        cp_world=world,
        cp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        causal_seqlens_kv_global=global_len,
    )

    def call():
        build_local_key_lengths(global_len, local, world, rank, interleave)
        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            heads,
            1,
            dim,
            topk,
            1,
            cu_seqlens_q=starts,
            seqused_k=local,
            batch_size=1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
            **dcp,
        )
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            query,
            key,
            weights,
            query_scale,
            scales,
            topk,
            1,
            cu_seqlens_q=starts,
            seqused_k=local,
            block_table=table,
            metadata=metadata,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
            return_value=1,
            output_idx_offset=None,
            **dcp,
        )

    def check(outputs, causal_end):
        indices, scores = [x.cpu() for x in outputs]
        assert indices.dtype == torch.int32 and scores.dtype == torch.bfloat16
        assert (indices[active:] == -1).all() and torch.isneginf(scores[active:]).all()
        local_ids = torch.arange(local_length)
        global_ids = (local_ids // interleave * world + rank) * interleave + local_ids % interleave
        for row in range(active):
            expected = local_ids[global_ids < causal_end - active + row + 1]
            valid = indices[row, 0] >= 0
            torch.testing.assert_close(indices[row, 0, valid].sort().values.long(), expected)
            assert torch.isfinite(scores[row, 0, valid]).all()
            assert torch.isneginf(scores[row, 0, ~valid]).all()

    for _ in range(3):
        result = call()
    check(result, world * interleave + active)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = call()
    for length in (0, world * interleave - 1, world * interleave + 13):
        global_len.fill_(length)
        captured[0].fill_(0)
        captured[1].fill_(0)
        graph.replay()
        torch.npu.synchronize()
        check(captured, length)


@pytest.mark.parametrize("local_length,rank", [(0, 7), (4097, 0), (4097, 7), (32769, 7)])
def test_native_dcp_matches_local_candidates_and_cpu_scores(local_length, rank):
    from vllm_ascend.device.device_op import A5DeviceAdaptor

    world, interleave, heads, dim, page, active, rows, topk = 8, 768, 32, 128, 128, 4, 8, 2048
    blocks = max(1, (local_length + page - 1) // page)
    token = torch.arange(blocks * page)
    # Binary fractions keep the BF16 comparison values exact, including ties.
    key_values = ((token * 17 % 33 - 8).float() / 8).to(torch.float8_e4m3fn).float()
    key_scales = 2.0 ** (token % 3 - 1)
    storage = torch.empty(blocks, page * (dim + 4), dtype=torch.uint8, device="npu")
    key = storage[:, : page * dim].view(torch.float8_e4m3fn).reshape(blocks, page, 1, dim)
    scales = storage[:, page * dim :].view(torch.float32).reshape(blocks, page, 1, 1)
    key.copy_(key_values[:, None].expand(-1, dim).reshape(key.shape))
    scales.copy_(key_scales.reshape(scales.shape))
    query = torch.ones(rows, heads, dim).to(key.dtype).npu()
    weights = torch.ones(rows, heads, device="npu")
    starts = torch.tensor([0, active], dtype=torch.int32, device="npu")
    local = torch.tensor([local_length], dtype=torch.int32, device="npu")
    causal_end = (4 if local_length < 10000 else 31) * world * interleave + 11
    global_len = torch.tensor([causal_end], dtype=torch.int32, device="npu")
    table = torch.arange(blocks, dtype=torch.int32, device="npu").view(1, -1)
    dcp = dict(global_seq_lens=global_len, cp_world=world, cp_rank=rank, cp_interleave=interleave)
    metadata = A5DeviceAdaptor.build_indexer_local_metadata(
        starts, local, num_heads_q=heads, head_dim=dim, topk=topk, **dcp
    )
    outputs = A5DeviceAdaptor.indexer_select_dcp_candidates(
        query, weights, key, scales, table, starts, local, weights, qli_metadata=metadata, topk=topk, **dcp
    )
    local_ids = torch.arange(local_length)
    global_ids = (local_ids // interleave * world + rank) * interleave + local_ids % interleave
    visible = [(global_ids < causal_end - active + row + 1).sum().item() for row in range(active)]
    lengths = torch.tensor(visible + [0] * (rows - active), dtype=torch.int32, device="npu").view(rows, 1)
    old_metadata = A5DeviceAdaptor.build_indexer_local_metadata(
        starts, local, num_heads_q=heads, head_dim=dim, topk=topk
    )
    old = A5DeviceAdaptor.indexer_select_local_candidates(
        query, weights, key, scales, table, starts, local, weights, lengths, qli_metadata=old_metadata, topk=topk
    )
    indices, scores = [tensor.cpu() for tensor in outputs]
    old_indices, old_scores = [tensor.cpu() for tensor in old]
    assert scores.dtype == old_scores.dtype == torch.bfloat16
    reference = (key_values * (heads * dim) * key_scales).clamp_min(0)
    for row, length in enumerate(visible + [0] * (rows - active)):
        valid = indices[row, 0] >= 0
        assert valid.sum().item() == min(length, topk)
        assert torch.isneginf(scores[row, 0, ~valid]).all()
        expected = torch.argsort(reference[:length], descending=True, stable=True)[:topk]
        torch.testing.assert_close(indices[row, 0, valid].sort().values.long(), expected.sort().values)
        torch.testing.assert_close(scores[row, 0, valid].float(), reference[indices[row, 0, valid].long()])
        torch.testing.assert_close(indices[row, 0].sort().values, old_indices[row, 0].sort().values)
        torch.testing.assert_close(scores[row, 0].float().sort().values, old_scores[row, 0].float().sort().values)


def test_bf16_candidates_pack_and_global_topk_match_reference():
    from vllm_ascend.attention.context_parallel.sharded_indexer import local_to_global, merge_topk_candidates
    from vllm_ascend.ops.triton.sharded_indexer import merge_packed_candidates, pack_local_candidates

    world, rows, topk, interleave = 8, 2, 16, 768
    packed, ids, values = [], [], []
    for rank in range(world):
        local = (torch.arange(topk, dtype=torch.int32) * 769).reshape(1, 1, topk).expand(rows, 1, topk).clone()
        scores = ((torch.arange(topk, dtype=torch.float32) * 3 + rank) % 7).to(torch.bfloat16)
        scores = scores.reshape(1, 1, topk).expand(rows, 1, topk).clone()
        local[1, :, topk // 2 :] = -1
        scores[1, :, topk // 2 :] = -torch.inf
        packed.append(pack_local_candidates(local.npu(), scores.npu(), world, rank, interleave))
        ids.append(local_to_global(local, world, rank, interleave))
        values.append(scores.float())
    result = merge_packed_candidates(torch.cat(packed, dim=0), rows, world, topk)
    expected = merge_topk_candidates(torch.cat(ids, dim=-1), torch.cat(values, dim=-1), topk)
    torch.testing.assert_close(result.cpu(), expected)
