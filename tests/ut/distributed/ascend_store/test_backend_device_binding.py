# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Startup regression tests: real connector/worker/backend/thread construction.

NPU contexts, distributed rank queries, store SDKs and lookup IPC are mocked.
These tests cover device binding, not model execution or KV payload correctness.
"""

import json
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

# Install dependency stubs before importing the production modules.
# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
# isort: on

import pytest
import torch
from vllm.distributed import parallel_state
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.platforms import interface
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
    ascend_store_connector,
    kv_transfer,
    pool_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import base, memcache_backend, mooncake_backend
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE
from vllm_ascend.platform import NPUPlatform


@pytest.fixture
def npu(monkeypatch):
    state = threading.local()
    events = []
    queries = []

    def current_device():
        queries.append(threading.current_thread().name)
        return getattr(state, "device", 0)

    def set_device(device):
        state.device = device if isinstance(device, int) else device.index
        events.append((threading.current_thread().name, state.device))

    fake = MagicMock()
    fake.current_device.side_effect = current_device
    fake.set_device.side_effect = set_device
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    # Also support the old code's torch.device("npu:N") on CPU-only PyTorch.
    real_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda value: SimpleNamespace(index=int(value.split(":")[1]))
        if isinstance(value, str) and value.startswith("npu:")
        else real_device(value),
    )
    monkeypatch.setattr(parallel_state, "_WORLD", SimpleNamespace(local_rank=0))
    monkeypatch.setattr(interface, "_assigned_physical_gpu_ids", None)
    # The old base module has no platform import; allow testing it unchanged.
    monkeypatch.setattr(base, "current_platform", NPUPlatform, raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    return SimpleNamespace(
        api=fake,
        events=events,
        queries=queries,
        bind=lambda device: setattr(state, "device", device),
        peek=lambda: getattr(state, "device", 0),
    )


@pytest.fixture(params=["memcache", "mooncake", "mooncake_independent", "mooncake_fabric"])
def backend(request, monkeypatch, tmp_path, npu):
    name = request.param
    for env in ("ASCEND_GLOBAL_RESOURCE_CONFIG", "ASCEND_ENABLE_USE_FABRIC_MEM", "MF_DEVICE_UB_QOS"):
        monkeypatch.delenv(env, raising=False)
    if name == "mooncake_independent":
        monkeypatch.setenv("ASCEND_GLOBAL_RESOURCE_CONFIG", '{"store": {}}')
    elif name == "mooncake_fabric":
        monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "1")
    mooncake_config = tmp_path / "mooncake.json"
    mooncake_config.write_text(json.dumps({"metadata_server": "P2PHANDSHAKE"}))
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(mooncake_config))
    memcache_config = tmp_path / "memcache.conf"
    memcache_config.write_text("ock.mmc.local_service.protocol=device_sdma\n")
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(memcache_config))
    monkeypatch.setattr(memcache_backend, "MEMCACHE_THREAD_START_WAIT_S", 0)
    monkeypatch.setattr(mooncake_backend, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mooncake_backend, "global_te", GlobalTE())
    stores = []
    native_devices = []

    def record_setup(*args, **kwargs):
        native_devices.append(npu.peek())
        return 0

    def create_store():
        native_devices.append(npu.peek())
        store = MagicMock()
        store.init.side_effect = record_setup
        store.setup.side_effect = record_setup
        store.register_buffer.return_value = 0
        stores.append(store)
        return store

    engine = MagicMock()
    engine.initialize.side_effect = record_setup
    engine.get_rpc_port.return_value = 50052
    engine.register_memory.return_value = 0
    monkeypatch.setattr(sys.modules["mooncake.engine"], "TransferEngine", lambda: engine)
    monkeypatch.setattr(sys.modules["mooncake.store"], "MooncakeDistributedStore", create_store, raising=False)
    monkeypatch.setattr(sys.modules["memcache_hybrid"], "DistributedObjectStore", create_store, raising=False)
    return SimpleNamespace(
        name="memcache" if name == "memcache" else "mooncake",
        cls=memcache_backend.MemcacheBackend if name == "memcache" else mooncake_backend.MooncakeBackend,
        lazy_supported=name in ("memcache", "mooncake_fabric"),
        stores=stores,
        native_devices=native_devices,
    )


def make_config(backend, device_id=0, tp_size=1, use_layerwise=False):
    config = MagicMock()
    config.parallel_config = SimpleNamespace(
        data_parallel_rank=device_id,
        data_parallel_index=device_id,
        data_parallel_rank_local=device_id,
        data_parallel_size=4,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        rank=0,
        world_size=tp_size,
        assigned_physical_gpu_ids=list(range(device_id, device_id + tp_size)),
        distributed_executor_backend="mp" if tp_size > 1 else "uni",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
    )
    config.model_config.model = "test/model"
    config.model_config.use_mla = False
    config.model_config.hf_config = config.model_config.hf_text_config = SimpleNamespace(num_hidden_layers=1)
    config.model_config.get_num_layers.return_value = 1
    config.model_config.get_total_num_kv_heads.return_value = tp_size
    config.cache_config.block_size = 16
    config.cache_config.prefix_match_unit = 16
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    config.speculative_config = None
    config.kv_events_config = None
    config.kv_transfer_config.kv_connector = "AscendStoreConnector"
    config.kv_transfer_config.kv_connector_module_path = ascend_store_connector.__name__
    config.kv_transfer_config.kv_role = "kv_both"
    config.kv_transfer_config.kv_connector_extra_config = {
        "backend": backend.name,
        "load_async": True,
        "use_layerwise": use_layerwise,
    }
    return config


def create_connector(config, role):
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    kv_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["model.layers.0.self_attn"], kv_cache_spec=spec)],
    )
    return KVConnectorFactory.create_connector(config, role, kv_config)


@pytest.fixture
def transfer_threads(monkeypatch):
    """Stop real transfer threads at their queue boundary after assertions."""
    queues = []
    threads: list[threading.Thread] = []
    stop = object()

    class StoppableQueue(queue.Queue):
        def __init__(self):
            super().__init__()
            queues.append(self)

        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            if item is stop:
                raise RuntimeError("test queue closed")
            return item

    monkeypatch.setattr(kv_transfer, "queue", SimpleNamespace(Queue=StoppableQueue))
    monkeypatch.setattr(ascend_store_connector, "LookupKeyServer", MagicMock())
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    yield threads
    for request_queue in queues:
        request_queue.put(stop)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("device_id", [0, 1, 2, 3])
@pytest.mark.parametrize("use_layerwise", [False, True])
def test_worker_and_colocated_scheduler_keep_device(backend, npu, transfer_threads, device_id, use_layerwise):
    # Simulate a bound worker in a DP shard whose group-local rank is still 0.
    npu.bind(device_id)
    config = make_config(backend, device_id, use_layerwise=use_layerwise)
    connector = create_connector(config, KVConnectorRole.WORKER)
    worker = connector.connector_worker
    cache = torch.empty((4, 16, 1, 8), dtype=torch.float16)
    worker.register_kv_caches({"model.layers.0.self_attn": (cache, cache.clone())})
    threads = [worker.kv_send_thread, worker.kv_recv_thread]
    transfer_threads.extend(threads)
    assert all(isinstance(thread, kv_transfer.KVTransferThread) for thread in threads)
    assert all((thread.name, device_id) in npu.events for thread in threads)

    create_connector(config, KVConnectorRole.SCHEDULER)
    assert npu.peek() == device_id
    assert npu.events and all(device == device_id for _, device in npu.events)
    assert backend.native_devices and set(backend.native_devices) == {device_id}
    if backend.name == "memcache":
        backend.stores[0].init.assert_called_once_with(device_id, init_bm=True)
        backend.stores[1].init.assert_called_once_with(device_id, init_bm=False)
    else:
        scheduler_setup = backend.stores[1].setup.call_args.kwargs
        assert scheduler_setup["global_segment_size"] == scheduler_setup["local_buffer_size"] == 0


@pytest.mark.parametrize(
    "visible,assigned,expected",
    [(None, [3], 3), ("7,2,5,1", [5], 2), ("7,2,5,1", [1, 5], 3), ("5", [5], 0)],
)
def test_standalone_scheduler_resolves_visible_device(backend, npu, monkeypatch, visible, assigned, expected):
    monkeypatch.setattr(parallel_state, "_WORLD", None)
    if visible is not None:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    config = make_config(backend, device_id=2, tp_size=len(assigned))
    config.parallel_config.assigned_physical_gpu_ids = assigned
    create_connector(config, KVConnectorRole.SCHEDULER)
    # No read of the default current device before the explicit binding.
    assert npu.queries == []
    assert npu.peek() == expected
    assert backend.native_devices and set(backend.native_devices) == {expected}
    if backend.name == "memcache":
        backend.stores[0].init.assert_called_once_with(expected, init_bm=False)
    else:
        assert backend.stores[0].setup.call_args.kwargs["global_segment_size"] == 0


@pytest.mark.parametrize("rank_local", [None, 2])
def test_standalone_scheduler_without_assigned_ids(backend, npu, monkeypatch, rank_local):
    monkeypatch.setattr(parallel_state, "_WORLD", None)
    config = make_config(backend, device_id=2, tp_size=2)
    config.parallel_config.assigned_physical_gpu_ids = None
    config.parallel_config.data_parallel_rank_local = rank_local
    create_connector(config, KVConnectorRole.SCHEDULER)
    assert npu.queries == []
    assert npu.peek() == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"distributed_executor_backend": "ray"},
        {"distributed_executor_backend": "external_launcher"},
        {"data_parallel_backend": "ray"},
        {"nnodes_within_dp": 2},
    ],
)
def test_scheduler_respects_executor_device_isolation(backend, npu, monkeypatch, overrides):
    monkeypatch.setattr(parallel_state, "_WORLD", None)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "5")
    config = make_config(backend, device_id=2)
    config.parallel_config.assigned_physical_gpu_ids = None
    vars(config.parallel_config).update(overrides)
    create_connector(config, KVConnectorRole.SCHEDULER)
    assert npu.queries == []
    assert npu.peek() == 0


def test_scheduler_rejects_invisible_assignment(backend, npu, monkeypatch):
    monkeypatch.setattr(parallel_state, "_WORLD", None)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="not visible"):
        create_connector(make_config(backend, device_id=3), KVConnectorRole.SCHEDULER)
    assert npu.events == npu.queries == backend.native_devices == []


def test_colocated_scheduler_keeps_nonzero_tp_worker_device(backend, npu):
    npu.bind(3)
    config = make_config(backend, device_id=2, tp_size=2)
    create_connector(config, KVConnectorRole.SCHEDULER)
    assert npu.peek() == 3
    assert backend.native_devices and set(backend.native_devices) == {3}


def test_lazy_setup_keeps_captured_device(backend, npu):
    npu.bind(3)
    store = backend.cls(make_config(backend, device_id=3).parallel_config, lazy_init=True)
    if backend.lazy_supported:
        assert backend.stores == []
    # Deferred setup may run on a fresh thread, whose default device is 0,
    # after the original caller has also changed its device.
    npu.bind(1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(store.ensure_initialized).result(timeout=5)
    assert backend.native_devices and set(backend.native_devices) == {3}
    store.set_device()
    assert npu.peek() == 3
    store.ensure_initialized()
    assert len(backend.stores) == 1
