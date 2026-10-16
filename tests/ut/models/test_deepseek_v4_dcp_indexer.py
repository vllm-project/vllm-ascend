# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.dsv4_indexer import (
    gather_dcp_indexer_cache,
    gather_dcp_topk,
    local_to_global_indices,
    merge_dcp_topk,
)


@pytest.mark.parametrize("rank", range(8))
def test_compressed_page_indices_round_trip(rank):
    local = torch.tensor([-1, 0, 31, 32, 33, 95], dtype=torch.int32)

    global_indices = local_to_global_indices(local, 32, 8, rank)

    assert global_indices.dtype == torch.int32
    assert global_indices[0] == -1
    valid = global_indices[1:]
    assert torch.equal(valid.div(32, rounding_mode="floor").remainder(8), torch.full_like(valid, rank))
    restored = valid.div(256, rounding_mode="floor") * 32 + valid.remainder(32)
    assert torch.equal(restored, local[1:])


@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 255, 256, 257, 4097])
@pytest.mark.parametrize("topk", [64, 512])
def test_merge_local_candidates_preserves_global_topk_scores(length, topk):
    dcp_size, block_size, num_queries = 8, 32, 3
    generator = torch.Generator().manual_seed(29)
    # Discrete scores deliberately exercise equal-score ordering across owners.
    full_scores = torch.randint(0, 7, (num_queries, length), generator=generator).float()
    all_global_indices = torch.arange(length, dtype=torch.int32)
    local_indices, local_scores = [], []
    for rank in range(dcp_size):
        owned = all_global_indices[(all_global_indices // block_size) % dcp_size == rank]
        score = full_scores[:, owned.long()]
        order = torch.argsort(score, dim=-1, descending=True, stable=True)[:, :topk]
        ids = owned[order]
        score = score.gather(-1, order)
        pad = topk - ids.shape[-1]
        ids = torch.nn.functional.pad(ids, (0, pad), value=-1)
        score = torch.nn.functional.pad(score, (0, pad), value=-torch.inf)
        local_indices.append(ids.unsqueeze(1))
        local_scores.append(score.unsqueeze(1))

    actual = merge_dcp_topk(torch.cat(local_indices), torch.cat(local_scores), dcp_size, topk)

    ids = actual.squeeze(1)
    valid = ids >= 0
    assert torch.all(valid.sum(dim=-1) == min(length, topk))
    assert torch.all(ids[~valid] == -1)
    for row in ids:
        selected = row[row >= 0]
        assert selected.unique().numel() == selected.numel()

    # Equal-score cutoffs may select different IDs from the unsharded reference.
    # Compare the actual selected scores, including the invalid padding slots.
    scores_with_padding = torch.nn.functional.pad(full_scores, (0, 1), value=-torch.inf)
    score_ids = torch.where(valid, ids, length).long()
    actual_scores = scores_with_padding.gather(-1, score_ids)
    expected_scores = torch.topk(full_scores, min(length, topk), dim=-1).values
    expected_scores = torch.nn.functional.pad(expected_scores, (0, topk - expected_scores.shape[-1]), value=-torch.inf)
    assert actual.dtype == torch.int32
    assert torch.equal(actual_scores, expected_scores)


def test_arbitrary_local_cutoff_ties_preserve_valid_topk_scores():
    # QLI does not define which equal-scoring rows survive its local cutoff.
    # These are valid local TopK subsets even though smaller tied IDs existed.
    indices = torch.tensor([[[9, 7]], [[39, 37]]], dtype=torch.int32)
    scores = torch.tensor([[[4.0, 4.0]], [[4.0, 4.0]]])

    with patch(
        "vllm_ascend.attention.context_parallel.dsv4_indexer.torch.argsort",
        wraps=torch.argsort,
    ) as sort:
        actual = merge_dcp_topk(indices, scores, 2, 2)

    sort.assert_called_once()
    assert sort.call_args.args[0].dtype == torch.float32
    # The shared rank-major candidate order breaks ties without sorting IDs.
    assert torch.equal(actual, torch.tensor([[[9, 7]]], dtype=torch.int32))
    assert torch.isin(actual.flatten(), indices.flatten()).all()
    full_scores = torch.full((64,), 4.0)
    assert torch.equal(full_scores[actual.long()], torch.topk(full_scores, 2).values.reshape(1, 1, 2))


def test_merge_masks_invalid_scores_from_an_empty_rank():
    indices = torch.tensor([[[-1, -1]], [[32, 33]]], dtype=torch.int32)
    scores = torch.tensor([[[100.0, 100.0]], [[2.0, 1.0]]])

    actual = merge_dcp_topk(indices, scores, 2, 2)

    assert torch.equal(actual, torch.tensor([[[32, 33]]], dtype=torch.int32))


def test_merge_preserves_fp32_score_precision():
    indices = torch.tensor([[[0, -1]], [[32, -1]]], dtype=torch.int32)
    scores = torch.tensor([[[1.0, -torch.inf]], [[1.0001, -torch.inf]]], dtype=torch.float32)

    actual = merge_dcp_topk(indices, scores, 2, 2)

    assert torch.equal(actual, torch.tensor([[[32, 0]]], dtype=torch.int32))


def test_decode_gather_converts_indices_before_collective():
    local_indices = torch.tensor([[[0, 32, -1]]], dtype=torch.int32)
    local_scores = torch.tensor([[[3.0, 2.0, -torch.inf]]])
    gathered_indices = torch.full((8, 1, 3), -1, dtype=torch.int32)
    gathered_scores = torch.full((8, 1, 3), -torch.inf)
    gathered_indices[:2] = torch.tensor([[[0, 256, -1]], [[32, 288, -1]]], dtype=torch.int32)
    gathered_scores[:2] = torch.tensor([[[3.0, 2.0, -torch.inf]], [[4.0, 1.0, -torch.inf]]])

    with patch(
        "vllm_ascend.attention.context_parallel.dsv4_indexer._all_gather_dcp_tensors",
        return_value=(gathered_indices, gathered_scores),
    ) as gather:
        actual = gather_dcp_topk(local_indices, local_scores, 32, 8, 1)

    assert torch.equal(gather.call_args.args[0], torch.tensor([[[32, 288, -1]]], dtype=torch.int32))
    assert gather.call_args.args[1] is local_scores
    assert torch.equal(actual, torch.tensor([[[32, 0, 256]]], dtype=torch.int32))


def test_prefill_gathers_matching_key_and_scale_pages():
    # Production views share a 4,160-byte page: 4,096 K bytes followed by
    # 64 scale bytes. Gather separates these into compact tensors.
    backing = torch.zeros(5 * 4160, dtype=torch.uint8)
    keys = backing.view(torch.int8).as_strided((5, 32, 1, 128), (4160, 128, 128, 1))
    scales = backing.view(torch.float16).as_strided((5, 32, 1, 1), (2080, 1, 1, 1), storage_offset=2048)
    for page in range(5):
        keys[page].fill_(page + 1)
        scales[page].fill_(page + 0.5)
    metadata = SimpleNamespace(dcp_prefill_kv_gather_block_ids=torch.tensor([3, 1], dtype=torch.int64))
    gathered_keys = torch.empty((16, 32, 1, 128), dtype=torch.int8)
    gathered_scales = torch.empty((16, 32, 1, 1), dtype=torch.float16)

    with patch(
        "vllm_ascend.attention.context_parallel.dsv4_indexer._all_gather_dcp_tensors",
        return_value=(gathered_keys, gathered_scales),
    ) as gather:
        actual_keys, actual_scales = gather_dcp_indexer_cache(keys, scales, metadata)

    assert torch.equal(gather.call_args.args[0], keys[[3, 1]])
    assert torch.equal(gather.call_args.args[1], scales[[3, 1]])
    assert gather.call_args.args[0].dtype == torch.int8
    assert gather.call_args.args[1].dtype == torch.float16
    assert gather.call_args.args[0].stride(0) == 4096
    assert gather.call_args.args[1].stride(0) == 32
    assert actual_keys is gathered_keys
    assert actual_scales is gathered_scales
