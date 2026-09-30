# SPDX-License-Identifier: Apache-2.0
"""Main adapter tests with the repository's CPU/NPU fixture boundary."""

import copy
import os
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.distributed.ascend_store.test_backend import _make_mooncake_store_config
from tests.ut.kv_offload import test_mooncake_layerwise_connector as layer_fixtures
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    KVCacheRecvingLayerThread,
    SendTask,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_backend as backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos import KvQosPolicy
from vllm_ascend.distributed.kv_transfer.qos_lifecycle import wait_remote_writes

POLICY = dict(priority_to_qos={0: 0, 3: 3, 7: 7}, level_names=True)


@pytest.mark.parametrize("contribute", [False, True])
def test_setup_preserves_tenant_device_capacity_and_environment(contribute):
    obj = backend.MooncakeBackend.__new__(backend.MooncakeBackend)
    obj.qos_policy = KvQosPolicy.from_config(POLICY)
    obj.config = _make_mooncake_store_config(tenant_id="team-a")
    obj.device_id = 3
    obj._contribute_memory = contribute
    factory = MagicMock()
    factory.return_value.default_segment = "host:5000"
    module = ModuleType("mooncake.qos_lane")
    module.QosStorePool = factory
    events = []
    factory.side_effect = lambda **kw: (
        events.append("factory") or SimpleNamespace(default_segment="host:5000", default_store=object())
    )
    before = dict(os.environ)
    with (
        patch.dict(sys.modules, {"mooncake.qos_lane": module}),
        patch.object(backend, "get_ip", return_value="host"),
        patch.object(backend.torch.npu, "set_device", side_effect=lambda n: events.append(("device", n))),
    ):
        obj.store = obj._setup_store()
    assert events == [("device", 3), "factory"]
    kw = factory.call_args.kwargs["setup_kwargs"]
    assert kw["tenant_id"] == "team-a"
    assert kw["global_segment_size"] == (obj.config.global_segment_size if contribute else 0)
    assert kw["local_buffer_size"] == (obj.config.local_buffer_size if contribute else 0)
    assert kw["rdma_devices"] == obj.config.device_name
    assert dict(os.environ) == before


@pytest.mark.parametrize("value", [0, 3, None, False])
def test_static_conflict_happens_before_backend_setup(value):
    before = dict(os.environ)
    with patch.object(backend.MooncakeStoreConfig, "load_from_env") as load:
        with pytest.raises(ValueError, match="conflict"):
            backend.MooncakeBackend(MagicMock(), extra_config={"kv_qos": POLICY, "qos_priority": value})
        load.assert_not_called()
    assert dict(os.environ) == before


def test_request_put_keeps_main_replication_config():
    obj = backend.MooncakeBackend.__new__(backend.MooncakeBackend)
    obj.qos_policy = KvQosPolicy.from_config(POLICY)
    obj.qos_pool = MagicMock()
    obj.config = _make_mooncake_store_config()
    obj.config.preferred_segment = True
    obj.config.prefer_alloc_in_same_node = True
    obj.local_seg = "host:123"
    obj.put_request("r", 7, ["unchanged-key"], [[100]], [[16]])
    args = obj.qos_pool.transfer.call_args.args
    assert args[:5] == (7, "put", ["unchanged-key"], [[100]], [[16]])
    assert args[5].preferred_segment == "host:123"
    assert args[5].prefer_alloc_in_same_node is True


def test_actual_layerwise_multicomponent_descriptors_and_event_order():
    # Reuse upstream geometry, execute both real branches and get_transfer_meta.
    fixture = layer_fixtures.TestKVCacheSendingLayerThread()
    fixture.setUp()
    try:
        sender = fixture.thread
        meta = copy.deepcopy(fixture.req_meta_base)
        meta.chunk_finish = True
        meta.kv_priority = 7
        meta.remote_qos_te_rpc_ports = {0: 6100, 3: 6103, 7: 6107}
        meta.remote_layer_metadata["layer1"] = copy.deepcopy(meta.remote_layer_metadata["layer0"])
        task = SendTask(
            send_request={"r": meta},
            wait_event=MagicMock(),
            layer_idx=2,
            layer_name="layer0",
            layer_names=["layer0", "layer1"],
            group_rearrange_block_ids=[[5, 8]],
        )
        order = []
        task.wait_event.synchronize.side_effect = lambda: order.append("sync")
        sender.engine.batch_transfer_sync_write.side_effect = lambda *a: order.append("write") or 0
        sender.callback_func.side_effect = lambda *a, **k: order.append("done")
        sender.reuse_completion_callback = lambda *a: order.append("reuse")
        sender.send_queue.put(task)
        sender._handle_request(sender.send_queue.get())
        reference = sender.engine.batch_transfer_sync_write.call_args.args[1:]
        assert order == ["sync", "write", "done", "reuse"]
        assert len(reference[0]) > 0
        order.clear()
        sender.qos_policy = KvQosPolicy.from_config(POLICY)
        sender.qos_pool = MagicMock()
        sender.qos_pool.write.side_effect = lambda *a: order.append("write") or 0
        sender.send_queue.put(task)
        sender._handle_request(sender.send_queue.get())
        actual = sender.qos_pool.write.call_args.args
        assert actual[:2] == (7, "127.0.0.1:6107")
        assert actual[2:] == reference
        assert order == ["sync", "write", "done", "reuse"]
        assert sender.send_queue.unfinished_tasks == 0
    finally:
        fixture.doCleanups()


def receiver():
    obj = KVCacheRecvingLayerThread.__new__(KVCacheRecvingLayerThread)
    obj.lock = threading.Lock()
    obj.task_tracker = {}
    obj.done_requests = set()
    obj.failed_requests = set()
    obj.qos_pending_requests = {"r"}
    obj.qos_receive_error = False
    obj.qos_stop_event = threading.Event()
    obj.is_alive = lambda: True
    return obj


def test_shutdown_waits_for_all_remote_write_contributions():
    obj = receiver()
    obj.update_done_task("r", 2, "p-rank0")
    with pytest.raises(TimeoutError):
        wait_remote_writes(obj, timeout=0)
    obj.update_done_task("r", 2, "p-rank0")
    assert obj.qos_pending_requests == {"r"}
    obj.update_done_task("r", 2, "p-rank1")
    wait_remote_writes(obj, timeout=0)
    assert obj.done_requests == {"r"}


def test_failed_write_prevents_unregister_even_after_error_consumed():
    obj = receiver()
    obj.update_failed_task("r")
    assert obj.get_and_clear_failed_requests() == {"r"}
    with pytest.raises(RuntimeError, match="completion unknown"):
        wait_remote_writes(obj, timeout=0)


def test_qos_tp_mismatch_guard_does_not_reject_disabled_main_path():
    from tests.ut.distributed.ascend_store import test_pool_worker as fixtures

    fixture = fixtures.TestKVPoolWorkerTpMismatch()
    base = {"backend": "mooncake", "prefill_tp_size": 4}
    assert fixture._make_worker(extra_config=base).tp_mismatch
    with pytest.raises(ValueError, match="QoS.*TP mismatch"):
        fixture._make_worker(extra_config=dict(base, kv_qos=POLICY))


@pytest.mark.parametrize("operation", ["get", "put"])
def test_missing_qos_metadata_never_reaches_default_store(operation):
    obj = backend.MooncakeBackend.__new__(backend.MooncakeBackend)
    obj.qos_policy = KvQosPolicy.from_config(POLICY)
    obj.store = MagicMock()
    with pytest.raises(ValueError, match=operation + "_request"):
        getattr(obj, operation)(["key"], [[100]], [[16]])
    assert not obj.store.mock_calls


@pytest.mark.parametrize("mode", ["new", "running", "preempted", "async_load"])
def test_scheduler_modes_keep_mixed_request_priorities(mode):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

    scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
    scheduler.qos_policy = KvQosPolicy.from_config(POLICY)
    scheduler.kv_role, scheduler.consumer_is_to_put = "kv_both", False
    scheduler.tp_mismatch = scheduler.layerwise_offload = scheduler.save_decode_cache = False
    ids = ["r0", "r3", "r7"]
    scheduler._unfinished_requests = {
        rid: (SimpleNamespace(kv_transfer_params={"kv_priority": q}), [[i]])
        for i, (rid, q) in enumerate(zip(ids, [0, 3, 7]))
    }
    scheduler._preempted_req_ids = set(ids) if mode == "preempted" else set()
    scheduler._request_trackers = {}
    scheduler._loading_req_ids = set()

    def metadata(rid):
        index = ids.index(rid)
        return ReqMeta(
            req_id=rid,
            token_len_chunk=17,
            block_ids_by_group=[[index]],
            block_hashes=[b"fixture"],
            can_save=True,
            load_spec=None,
        )

    # The allocator is controlled; the scheduling branches and policy binding
    # are the production build_connector_meta implementation.
    scheduler._process_new_request = lambda r, *a: metadata(r.req_id)
    scheduler._process_running_cached_request = lambda blocks, rid, *a: metadata(rid)
    scheduler._process_preempted_cached_request = lambda blocks, rid, *a: metadata(rid)
    scheduler._process_async_load_request = lambda rid, *a: metadata(rid)
    scheduler.touch_sending_mamba_blocks = lambda r: None
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(req_id=rid) for rid in ids] if mode == "new" else [],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=ids if mode in ["running", "preempted"] else [], new_block_ids=[[[i]] for i in range(3)]
        ),
    )
    result = scheduler.build_connector_meta(output)
    assert [(r.req_id, r.kv_priority, r.token_len_chunk, r.block_ids_by_group) for r in result.requests] == [
        (rid, q, 17, [[i]]) for i, (rid, q) in enumerate(zip(ids, [0, 3, 7]))
    ]
