# SPDX-License-Identifier: Apache-2.0
"""Cache accounting and PD ownership for token-sharded MiniMax index keys."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendMiniMaxM3IndexerDCPSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.models.minimax_m3.msa_m3 import AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3IndexerCache
from vllm_ascend.models.minimax_m3.ops.indexer_dcp import (
    index_shard_scores,
    plan_prefill_index_gather,
    reduce_scatter_index_scores,
)


@pytest.fixture(autouse=True)
def available_cann_index_score():
    with patch("vllm_ascend.models.minimax_m3.msa_m3._CANN_INDEX_SCORE_AVAILABLE", True):
        yield


def config(tp=4, enabled=True, **overrides):
    result = SimpleNamespace(
        additional_config={"enable_minimax_m3_indexer_dcp": enabled},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        cache_config=SimpleNamespace(block_size=128),
        kv_transfer_config=None,
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


@pytest.mark.parametrize("tp", [2, 4])
def test_indexer_shard_accounts_for_actual_storage(tp):
    cfg = config(tp=tp)
    with patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config", return_value=cfg):
        cache = AscendMiniMaxM3IndexerCache(head_dim=128, prefix="indexer")
    spec = cache.get_kv_cache_spec(cfg)
    assert isinstance(spec, AscendMiniMaxM3IndexerDCPSpec)
    shape = AscendMiniMaxM3IndexerBackend.get_kv_cache_shape(10, 128, 1, spec.head_size)
    assert shape == (10, 128 // tp, 128)
    assert spec.page_size_bytes == 128 * 128 * 2 // tp
    merged = type(spec).merge([spec, spec])
    assert merged == spec


def test_disabled_cache_keeps_replicated_layout():
    cfg = config(enabled=False)
    with patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config", return_value=cfg):
        cache = AscendMiniMaxM3IndexerCache(head_dim=128, prefix="indexer")
    spec = cache.get_kv_cache_spec(cfg)
    assert type(spec) is AscendSFAIndexerCacheSpec
    assert spec.page_size_bytes == 128 * 128 * 2


@pytest.mark.parametrize("tp", [1, 8])
def test_rejects_unsupported_tp(tp):
    with (
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config", return_value=config(tp=tp)),
        pytest.raises(ValueError, match="TP2 or TP4"),
    ):
        AscendMiniMaxM3IndexerCache(head_dim=128, prefix="indexer")


def test_pd_uses_the_matching_rank_instead_of_an_index_replica():
    worker = object.__new__(MooncakePullRecvingThread)
    worker.tp_size = 4
    worker.tp_rank = 2
    worker.pcp_size = worker.dcp_size = 1
    worker.block_shapes = [[(32, 128)]]
    remote = SimpleNamespace(block_shapes=[[(32, 128)]])
    spec = AscendMiniMaxM3IndexerDCPSpec(block_size=128, num_kv_heads=1, head_size=32, dtype=torch.bfloat16)
    assert worker._get_layer_remote_tp_rank_groups(0, 0, spec, remote, 1, 4, 1) == [[2]]
    remote.block_shapes = [[(128, 128)]]
    with pytest.raises(ValueError, match="matching TP"):
        worker._get_layer_remote_tp_rank_groups(0, 0, spec, remote, 1, 4, 1)


@pytest.mark.parametrize("field", ["decode_context_parallel_size", "prefill_context_parallel_size"])
def test_rejects_generic_context_parallelism(field):
    cfg = config()
    setattr(cfg.parallel_config, field, 2)
    with (
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config", return_value=cfg),
        pytest.raises(ValueError, match="generic PCP/DCP"),
    ):
        AscendMiniMaxM3IndexerCache(head_dim=128, prefix="indexer")


def test_rejects_incompatible_pd_connector():
    cfg = config(kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnectorV1"))
    with (
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config", return_value=cfg),
        pytest.raises(ValueError, match="MooncakeConnectorV2"),
    ):
        AscendMiniMaxM3IndexerCache(head_dim=128, prefix="indexer")


def test_rejects_unsupported_cache_dtype():
    with (
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config", return_value=config()),
        pytest.raises(ValueError, match="BF16"),
    ):
        AscendMiniMaxM3IndexerCache(head_dim=128, prefix="indexer", kv_cache_torch_dtype=torch.float8_e4m3fn)


def test_rejects_missing_cann_index_score():
    with (
        patch("vllm_ascend.models.minimax_m3.msa_m3._CANN_INDEX_SCORE_AVAILABLE", False),
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config", return_value=config()),
        pytest.raises(ValueError, match="CANN MsaIndexScore"),
    ):
        AscendMiniMaxM3IndexerCache(head_dim=128, prefix="indexer")


def test_prefill_gather_excludes_stale_pages_and_reconstructs_valid_entries():
    table = torch.tensor([[7, 2, 999, -1], [2, 444, 555, 666], [888, 777, 666, 555]], dtype=torch.int32)
    lengths = torch.tensor([129, 1, 0])
    pages, mapped = plan_prefill_index_gather(table, lengths, 128)
    assert pages.tolist() == [0, 2, 7]
    assert mapped.tolist() == [[2, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    assert torch.equal(pages[mapped[0, :2].long()], table[0, :2])
    assert pages[mapped[1, 0]] == table[1, 0]


def test_prefill_gather_empty_batch_retains_only_null_page():
    pages, mapped = plan_prefill_index_gather(torch.empty(0, 4, dtype=torch.int32), torch.empty(0), 128)
    assert pages.tolist() == [0]
    assert mapped.shape == (0, 4)


@pytest.mark.parametrize("query_len", [0, 2])
def test_compact_scores_reject_nonuniform_batch(query_len):
    with pytest.raises(ValueError, match="uniform"):
        index_shard_scores(
            torch.empty(3, 4, 128, dtype=torch.bfloat16),
            torch.empty(2, 32, 128, dtype=torch.bfloat16),
            torch.zeros(3, 2, dtype=torch.int32),
            torch.ones(3, dtype=torch.int32),
            query_len,
            4,
            0,
        )


def test_compact_scores_empty_batch_does_not_launch_a_kernel():
    scores = index_shard_scores(
        torch.empty(0, 4, 128, dtype=torch.bfloat16),
        torch.empty(2, 32, 128, dtype=torch.bfloat16),
        torch.empty(0, 7, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        1,
        4,
        0,
    )
    assert scores.shape == (4, 0, 16)


def test_reduce_scatter_rejects_uneven_head_ownership():
    with pytest.raises(ValueError, match="divide evenly"):
        reduce_scatter_index_scores(torch.empty(3, 2, 16), SimpleNamespace(world_size=4))
