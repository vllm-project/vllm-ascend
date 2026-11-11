# SPDX-License-Identifier: Apache-2.0
"""Multiple P shards write real CPU buffers for every D rank, without an NPU."""

import ctypes
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import push_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata

from . import test_push_sending
from .test_push_dcp import configure, metadata

worker = test_push_sending.worker


@pytest.fixture
def producers(worker, monkeypatch):
    created = []

    def make(dcp, tp, remote, interleave, scale, cache_kind):
        for rank in range(tp):
            config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    decode_context_parallel_size=dcp,
                    tensor_parallel_size=tp,
                    prefill_context_parallel_size=1,
                )
            )
            obj = push_worker.MooncakePushConnectorWorker(config, "p", None)
            created.append(obj)
            obj.register_kv_caches({})
            obj.dcp_size, obj.tp_size, obj.tp_rank, obj.dcp_rank = dcp, tp, rank, rank % dcp
            obj.transfer_metadata = replace(
                worker.transfer_metadata,
                cp_kv_cache_interleave_size=interleave,
                layer_block_sizes=[4 * scale],
                block_size_scales=[[scale]],
                kv_caches_base_addr=[[0]],
            )
            if cache_kind == "mla":
                spec = Mock(spec=push_worker.MLAAttentionSpec, sliding_window=None, is_circular=False)
            else:
                spec = Mock(
                    spec=push_worker.AscendSFAIndexerCacheSpec,
                    sliding_window=None,
                    is_circular=False,
                    sfa_dcp_replicated_indexer_size=dcp if cache_kind in ("p_replicated", "both_replicated") else 1,
                )
            obj.kv_cache_specs = [spec]
            monkeypatch.setattr(obj, "_get_remote_metadata", lambda _: remote)
            monkeypatch.setattr(obj, "_send_done", Mock(return_value=True))
        return created

    yield make
    for obj in created:
        obj.shutdown()


@pytest.mark.parametrize(
    "p_dcp,d_dcp,p_tp,d_tp",
    [
        (1, 4, 2, 8),
        (4, 1, 8, 2),
        (2, 4, 4, 8),
        (4, 2, 8, 4),
        (4, 4, 8, 4),
        (8, 8, 8, 8),
    ],
)
@pytest.mark.parametrize(
    "p_interleave,d_interleave,p_scale,d_scale",
    [
        (4, 4, 1, 2),
        (4, 8, 2, 4),
        (8, 4, 4, 1),
        (8, 8, 2, 2),
    ],
)
@pytest.mark.parametrize("prefix", [0, 5, 16])
@pytest.mark.parametrize("cache_kind", ["mla", "p_replicated", "d_replicated", "both_replicated"])
def test_bilateral_chunks_copy_each_token_once(
    worker,
    monkeypatch,
    producers,
    p_dcp,
    d_dcp,
    p_tp,
    d_tp,
    p_interleave,
    d_interleave,
    p_scale,
    d_scale,
    prefix,
    cache_kind,
):
    remote = configure(worker, monkeypatch, d_dcp, d_tp, d_interleave, p_scale, d_scale)
    pp = remote.metadata_by_pp_rank[0]
    p_effective = 1 if cache_kind in ("p_replicated", "both_replicated") else p_dcp
    d_effective = 1 if cache_kind in ("d_replicated", "both_replicated") else d_dcp
    if d_effective == 1 and cache_kind != "mla":
        pp.indexer_replication_sizes[:] = [d_dcp]
    senders = producers(p_dcp, p_tp, remote, p_interleave, p_scale, cache_kind)
    source_blocks = tuple(1 + 2 * i for i in range(16))
    target_blocks = tuple(10 + 3 * i for i in range(16))
    sources = [(ctypes.c_ubyte * 1024)(*([254] * 1024)) for _ in senders]
    targets = [(ctypes.c_ubyte * 1024)(*([254] * 1024)) for _ in range(d_tp)]
    for rank, (sender, source) in enumerate(zip(senders, sources)):
        owned = [token for token in range(32) if (token // p_interleave) % p_effective == rank % p_effective]
        for position, token in enumerate(owned):
            page, offset = divmod(position, 4 * p_scale)
            source[source_blocks[page] * 4 * p_scale + offset] = token + 1
        sender.transfer_metadata.kv_caches_base_addr[:] = [[ctypes.addressof(source)]]
    for rank, target in enumerate(targets):
        pp.metadata_by_pcp_rank[0].metadata_by_tp_rank[rank].kv_caches_base_addr[:] = [[ctypes.addressof(target)]]
    writes = []

    def copy(session, src, dst, lengths):
        for source_addr, target_addr, size in zip(src, dst, lengths):
            assert any(
                ctypes.addressof(buf) <= source_addr < source_addr + size <= ctypes.addressof(buf) + 1024
                for buf in sources
            )
            assert any(
                ctypes.addressof(buf) <= target_addr < target_addr + size <= ctypes.addressof(buf) + 1024
                for buf in targets
            )
            writes.extend(range(target_addr, target_addr + size))
            ctypes.memmove(target_addr, source_addr, size)
        return 0

    for sender in senders:
        sender.engine.batch_transfer_sync_write.side_effect = copy
    previous = 0
    for end in (3, 9, 17, 29):
        for sender in senders:
            meta = metadata(previous, end, prefix=prefix, remote_blocks=target_blocks, tp=d_tp)
            meta.push_send_requests["p-request"] = replace(
                meta.push_send_requests["p-request"], local_block_ids=(source_blocks,)
            )
            sender.start_load_kv(meta)
            sender.save_kv_layer("model.layers.0.attn")
            sender.wait_for_save()
            assert all(call.args[1] is None for call in sender._send_done.call_args_list)
            if end < 29:
                for destination in sender._reqs_send_state["d-request"].destinations.values():
                    if destination.required_layers:
                        assert destination.completion is None
        ready = end // 4 * 4 if end < 29 else 32
        for rank, target in enumerate(targets):
            expected = [254] * 1024
            owned = [token for token in range(32) if (token // d_interleave) % d_effective == rank % d_effective]
            for position, token in enumerate(owned):
                if prefix // 4 * 4 <= token < ready:
                    page, offset = divmod(position, 4 * d_scale)
                    expected[target_blocks[page] * 4 * d_scale + offset] = token + 1
            assert list(target) == expected
        previous = end
    assert len(writes) == len(set(writes)), "Replicas/chunks must not overwrite the same D byte twice"
    assert all(sender._send_done.call_count == d_tp for sender in senders)


@pytest.mark.parametrize("p_dcp,d_dcp", [(2, 4), (4, 2), (4, 1)])
def test_invalid_producer_interleave_is_rejected(worker, monkeypatch, p_dcp, d_dcp):
    configure(worker, monkeypatch, dcp=d_dcp, tp=d_dcp)
    worker.tp_size = worker.dcp_size = p_dcp
    worker.transfer_metadata = replace(worker.transfer_metadata, cp_kv_cache_interleave_size=2)
    worker.start_load_kv(metadata(0, 8, prompt=8, tp=d_dcp))
    worker.wait_for_save()
    worker.engine.batch_transfer_sync_write.assert_not_called()
    assert worker._send_done.call_count == d_dcp
    assert all("P page size" in call.args[1] for call in worker._send_done.call_args_list)


@pytest.mark.parametrize("write_fails", [False, True])
def test_receiver_waits_for_last_source_shard(worker, monkeypatch, producers, write_fails):
    remote = configure(worker, monkeypatch, dcp=1, tp=1)
    senders = producers(2, 4, remote, 4, 1, "mla")
    worker.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    recv_meta = MooncakeConnectorMetadata()
    recv_meta.add_new_push_req("d-request", ([10, 11],), 8, {"num_computed_tokens": 0})
    worker.start_load_kv(recv_meta)
    entered, release = threading.Event(), threading.Event()

    def delayed_write(*args):
        entered.set()
        assert release.wait(timeout=5)
        return -1 if write_fails else 0

    def done(params, error):
        payload = dict(params, producer_sizes=[1, 1, 4], status="error" if error else "ok")
        worker._result_queue.put(("done", "d-request", payload))
        return True

    senders[1].engine.batch_transfer_sync_write.side_effect = delayed_write
    for sender in senders:
        monkeypatch.setattr(sender, "_send_done", Mock(side_effect=done))
    try:
        for sender in senders:
            sender.start_load_kv(metadata(0, 8, prompt=8, tp=1))
            sender._reqs_send_state["d-request"].destinations[(0, 0, 0)].metadata_future.result(timeout=3)
            sender.save_kv_layer("model.layers.0.attn")
        assert entered.wait(timeout=3)
        for rank in (0, 2, 3):
            senders[rank].wait_for_save()
        # Non-owner replicas send DONE too, but cannot substitute for shard 1.
        senders[1]._send_done.assert_not_called()
        assert not worker.get_transfer_results().finished_recving
        assert worker._reqs_recv_state["d-request"].finished_ranks == {(0, 0, rank) for rank in (0, 2, 3)}
    finally:
        release.set()
    senders[1].wait_for_save()
    result = worker.get_transfer_results()
    assert result.finished_recving == {"d-request"}
    assert result.failed_recving == ({"d-request"} if write_fails else set())
