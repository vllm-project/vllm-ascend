# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.hisparse_consumer import (
    HiSparseLayerwiseReadThread,
    HiSparseReadState,
    LayerDestination,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.hisparse_layout import PageBuffer
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
    hisparse_target_enabled,
    hisparse_transfer_group_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import SFAPDRD2HScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import SFAPDRD2HProducerWorker


def _receiver(rank, shared):
    thread = HiSparseLayerwiseReadThread.__new__(HiSparseLayerwiseReadThread)
    thread.tp_rank = rank
    thread.engine = MagicMock()
    thread.engine.batch_transfer_sync_read.return_value = 0
    dest = LayerDestination(PageBuffer(6000, 6, 8, 6), (PageBuffer(9000, 2, 4, 6),))
    thread._state = HiSparseReadState(
        {"layer": dest}, {"req": ([1, 4, 2], [0, 3, 5])}, 2, 4, 2, rank if shared else 0, 2 if shared else 1
    )
    source = {
        "layer": {
            "main_tensor_count": 1,
            "has_indexer": True,
            "base_addrs": [2000, 3000],
            "block_len": [6, 2],
            "block_size_scale": [1, 1],
            "block_shapes": [(2, 3), (2, 1)],
            "num_logical_blocks": [3, 3],
            "block_tokens": [2, 2],
        }
    }
    return thread, source


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_receiver_targets_and_byte_boundaries(shared, rank):
    thread, source = _receiver(rank, shared)
    requests = thread._decode_read_requests([("req", [2, 1], [1, 0], 1, 1, 5)])
    thread._do_read_batch("layer", requests, "peer", source)
    peer, local, remote, sizes = thread.engine.batch_transfer_sync_read.call_args.args
    assert peer == "peer"
    main = [(6032, 2012, 6), (6016, 2006, 3)]
    expected = [m for offset, m in enumerate(main, 1) if not shared or offset % 2 == rank]
    assert list(zip(local, remote, sizes, strict=True)) == expected + [(9012, 3002, 2), (9020, 3000, 1)]


def test_receiver_rejects_incompatible_peer_and_device_errors():
    thread, source = _receiver(0, False)
    with pytest.raises(ValueError, match="computed-token boundary"):
        thread._decode_read_requests([("req", [0], [0], 0, 0)])
    request = [("req", [3], [], 0, 0, 1)]
    with pytest.raises(ValueError, match="Source page"):
        thread._do_read_batch("layer", request, "peer", source)
    with pytest.raises(ValueError, match="matching Prefill TP"):
        thread._do_read_batch("layer", request, "peer", source, ratio=2)
    thread.engine.batch_transfer_sync_read.assert_not_called()
    thread.engine.batch_transfer_sync_read.return_value = -1
    with pytest.raises(RuntimeError, match="read failed"):
        thread._do_read_batch("layer", [("req", [0], [], 0, 0, 1)], "peer", source)


def test_transfer_groups_exclude_hot_resident_and_draft():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(role=r)
            for r in ("hisparse_hot", "hisparse_indexer", "hisparse_resident", "hisparse_source", "draft")
        ],
        host_group_ids=(3, 4),
        transfer_group_ids=(1, 3),
    )
    assert hisparse_transfer_group_ids(config) == (3, 1)
    config.transfer_group_ids = (1, 2)
    with pytest.raises(ValueError, match="Host source"):
        hisparse_transfer_group_ids(config)
    with pytest.raises(ValueError, match="boolean"):
        hisparse_target_enabled({"hisparse_target": "false"})


@pytest.mark.parametrize("local,external,reported", [(4, 3, 0), (4, 3, 7), (8, 0, 0)])
def test_advertised_watermark_is_from_lookup(local, external, reported):
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler._hisparse_target = True
    scheduler._remote_prefill_matches = {}
    scheduler.main_group_idx, scheduler.indexer_group_idx = 0, 1
    scheduler.block_size = [4, 4]
    scheduler._request_trackers, scheduler._reqs_need_recv, scheduler._dspark_pending_recv = {}, set(), set()
    scheduler.side_channel_host, scheduler.side_channel_port, scheduler.engine_id = "test-host", 12345, "test-id"
    scheduler._metaserver_lock, scheduler._cancelled_metaserver_requests = threading.Lock(), set()
    scheduler._submit_metaserver_request = MagicMock()
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2, prefill_context_parallel_size=1, decode_context_parallel_size=1
        )
    )
    request = SimpleNamespace(
        request_id="req",
        num_computed_tokens=reported,
        prompt_token_ids=[1] * (local + external),
        kv_transfer_params={"do_remote_prefill": True, "metaserver": "test"},
    )
    assert scheduler.get_num_new_matched_tokens(request, local) == (external, bool(external))
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([2, 3], [4, 5])
    scheduler.update_state_after_alloc(request, blocks, external)
    message = scheduler._submit_metaserver_request.call_args.kwargs["message"]
    assert message["remote_cached_tokens"] == local
    assert message["do_remote_decode"] == bool(external)
    assert ("req" in scheduler._dspark_pending_recv) == bool(external)
    assert scheduler._remote_prefill_matches == {}


def test_final_save_fences_dispatched_layers():
    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    worker._pd_dispatched_layers = {4, 7}
    worker.wait_for_layer_send = MagicMock()
    worker.wait_for_save()
    assert sorted(c.args[0] for c in worker.wait_for_layer_send.call_args_list) == [4, 7]


def test_pp_stages_must_cover_every_destination_layer_before_final_ack():
    thread = HiSparseLayerwiseReadThread.__new__(HiSparseLayerwiseReadThread)
    thread._state = SimpleNamespace(layer_metadata={"a": 1, "b": 2, "c": 3})
    thread._lock = threading.Lock()
    thread._finished_stages = {}
    thread._check_final_layers(["ok", "incomplete"], ["b", "c"], 1, 2)
    thread._check_final_layers(["ok"], ["a"], 0, 2)
    with pytest.raises(ValueError, match="cover all"):
        thread._check_final_layers(["incomplete"], ["b"], 0, 2)
    assert not thread._finished_stages
