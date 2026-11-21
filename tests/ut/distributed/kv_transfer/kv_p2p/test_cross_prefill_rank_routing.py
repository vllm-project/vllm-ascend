# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import MooncakeConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    MooncakeConnectorWorker as HybridMooncakeConnectorWorker,
)


@pytest.mark.parametrize("worker_cls", [MooncakeConnectorWorker, HybridMooncakeConnectorWorker])
@pytest.mark.parametrize("prefill_tp,decode_tp", [(8, 1), (8, 4), (4, 8), (8, 8)])
@pytest.mark.parametrize("prefill_pp", [1, 2])
@pytest.mark.parametrize("mla,sparse", [(False, False), (True, False), (False, True)])
def test_cross_prefill_preserves_tp_shard(worker_cls, prefill_tp, decode_tp, prefill_pp, mla, sparse):
    worker = worker_cls.__new__(worker_cls)
    worker.kv_role = "kv_producer"
    worker.tp_size = worker._prefill_tp_size = prefill_tp
    worker._decode_tp_size = decode_tp
    worker._prefill_pp_size = prefill_pp
    worker.num_key_value_heads = 16
    worker.tp_num_need_pulls = 1
    worker.use_sparse = sparse
    worker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(is_deepseek_mla=mla))

    for tp_rank in range(prefill_tp):
        worker.tp_rank = tp_rank
        expected = [tp_rank + pp_rank * prefill_tp for pp_rank in range(prefill_pp)]
        # Check ranks inside the old decode grouping as well as out-of-range
        # ranks: merely clamping/modulo can avoid the crash but load wrong KV.
        assert worker._get_remote_rank("cross-prefill") == expected
        assert worker._get_remote_rank("cross-prefill", prefill_tp) == expected


@pytest.mark.parametrize("worker_cls", [MooncakeConnectorWorker, HybridMooncakeConnectorWorker])
def test_cross_prefill_rejects_different_source_tp(worker_cls):
    worker = worker_cls.__new__(worker_cls)
    worker.kv_role = "kv_producer"
    worker.tp_size = worker._prefill_tp_size = 8
    worker._prefill_pp_size = 1
    worker.tp_rank = 0

    with pytest.raises(ValueError, match="source TP=4 and destination TP=8"):
        worker._get_remote_rank("cross-prefill", 4)


@pytest.mark.parametrize("worker_cls", [MooncakeConnectorWorker, HybridMooncakeConnectorWorker])
@pytest.mark.parametrize("source_tp", [None, 4, 8])
def test_decode_keeps_existing_rank_groups(worker_cls, source_tp):
    worker = worker_cls.__new__(worker_cls)
    worker.kv_role = "kv_consumer"
    worker.tp_rank = 1
    worker._get_remote_ranks_for_req = Mock(return_value=[[0, 1], [2, 3]])

    assert worker._get_remote_rank("decode", source_tp) == [2, 3]
    worker._get_remote_ranks_for_req.assert_called_once_with("decode", source_tp)


@pytest.mark.parametrize("tp_rank", range(8))
def test_cross_prefill_split_metadata_uses_matching_port(tp_rank):
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.kv_role = "kv_producer"
    worker.tp_size = worker._prefill_tp_size = 8
    worker._decode_tp_size = worker._prefill_pp_size = 1
    worker.tp_rank = tp_rank
    worker.dcp_size = 1
    worker._is_hma_required = False
    worker.block_size = 128
    worker.block_size_scale = [[1]]
    worker.kv_group2layeridx = {0: ({"kv_cache_spec_type": "FullAttentionSpec"}, [0])}
    meta = SimpleNamespace(
        remote_ptp_size=8,
        remote_dcp_size=1,
        remote_pcp_size=1,
        remote_port=1234,
        num_external_tokens=128,
        num_computed_tokens=0,
        local_block_ids=([5],),
        remote_block_ids=([9],),
        remote_block_size=128,
    )

    ports, local_ids, remote_ids = worker._get_kv_split_metadata("cross-prefill", meta)

    assert ports == [[1234 + tp_rank]]
    assert local_ids == [([5],)]
    assert remote_ids == [([9],)]
