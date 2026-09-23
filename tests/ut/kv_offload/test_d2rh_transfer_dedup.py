# SPDX-License-Identifier: Apache-2.0
"""Shared metadata layers must not repeat component transfers or reformats."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh


@pytest.mark.parametrize("indices,names", [([0, 0], ["indexer", "attention"]), ([0], ["attention"])])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("packed_alias", [False, True])
def test_h2d_preserves_each_component_once(indices, names, legacy, packed_alias):
    original_indices = list(indices)
    spec: dict[str, object] = {"kv_cache_spec_type": "AttentionSpec"}
    if not legacy:
        spec.update(layer_names=names, layer_cache_indices={name: [i] for i, name in enumerate(names)})
    slots = len(names)
    groups = {0: (spec, indices)}
    worker = SimpleNamespace(
        remote_metadata_lock=nullcontext(),
        local_engine_id="d",
        local_handshake_port=10,
        kv_caches_base_addr={
            "d": {10: [[1000 + i * 100 for i in range(slots)]]},
            "host": {20: [[2000 + i * 100 for i in range(slots)]]},
        },
        remote_te_port={"host": {20: 30}},
        remote_block_stride_per_addr={"host": {20: [[16] * slots]}},
        remote_kv_group2layeridx={"host": {20: groups}},
        kv_group2layeridx=groups,
        pp_layer_indices=[(0, 1)],
        vllm_config=SimpleNamespace(speculative_config=None),
        block_size_scale=[[1] * slots],
        block_len_per_addr=[[8] * slots],
        block_stride_per_addr=[[16] * slots],
        engine=SimpleNamespace(batch_transfer_sync_read=Mock(return_value=0)),
        tp_rank=0,
        _stash_pending_reformat=Mock(),
    )
    request = dict(
        request_id="d-request",
        remote_request_id="p-request",
        local_block_ids=([3],),
        remote_block_ids=([5],),
        remote_engine_id="host",
        remote_host="127.0.0.1",
        remote_handshake_port=20,
        group_pulls=[
            SimpleNamespace(
                group_id=0, prefill_pp_rank=0, num_group_pulls=1, remote_tp_offset=0, is_group_transfer_end=True
            )
        ],
    )
    if packed_alias and slots > 1:
        worker.kv_caches_base_addr["d"][10] = [[1000, 1000]]
        worker.kv_caches_base_addr["host"][20] = [[2000, 2000]]
        worker.block_len_per_addr = [[8, 16]]
    with patch.object(d2rh, "transfer_groups_need_independent_block_ids", return_value=True):
        d2rh.KVCacheRecvingThread._transfer_staged_kv_cache_all_groups(worker, request)
    if packed_alias and slots > 1:
        worker.engine.batch_transfer_sync_read.assert_called_once_with("127.0.0.1:30", [1048], [2080], [16])
    else:
        worker.engine.batch_transfer_sync_read.assert_called_once_with(
            "127.0.0.1:30",
            [1048 + i * 100 for i in range(slots)],
            [2080 + i * 100 for i in range(slots)],
            [8] * slots,
        )
    worker._stash_pending_reformat.assert_called_once_with("d-request", 0, [(0, [[3]], 1, [0])])
    assert groups[0][1] == original_indices
