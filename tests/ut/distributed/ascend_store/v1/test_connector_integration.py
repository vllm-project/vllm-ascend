"""Real Connector, Scheduler and Worker composition over CPU native-copy adapters."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.native_copy_fixtures import ByteCopySDK, native_backend
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlocks,
    make_output,
    make_request,
    new_request_data,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import BackendSpec, LayerwiseAccessKind
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import AscendStoreV1Connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import Resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import LookupCodec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LookupRequest,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import Worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import worker as worker_module


@pytest.mark.parametrize("mode", ("bulk_sync", "bulk_async", "key_range", "gva"))
@pytest.mark.parametrize("load_failure", (False, True))
def test_connector_round_trip_keeps_local_prefix_and_releases_sources(
    monkeypatch, configuration_case, mode, load_failure
):
    layerwise = mode in ("key_range", "gva")
    async_load = mode == "bulk_async"
    backend_name = "memcache" if mode == "gva" else "mooncake"
    topology = make_topology()
    config = replace(topology.config, backend_name=backend_name, use_layerwise=layerwise, load_async=async_load)
    topology = replace(topology, config=config)
    sdk = ByteCopySDK({})
    backend = native_backend(backend_name, sdk)
    access = LayerwiseAccessKind.GVA if mode == "gva" else LayerwiseAccessKind.KEY_RANGE
    spec = BackendSpec(backend_name, lambda *_args, **_kwargs: backend, access, True)
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_: config)
    monkeypatch.setattr(connector_module, "resolve_worker_topology", lambda *_: topology)
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _: spec)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _: None, Event=FakeEvent), raising=False)
    vllm_config, cache_config = configuration_case
    vllm_config.parallel_config.rank = 1
    vllm_config.parallel_config.data_parallel_rank = 0
    worker_connector = AscendStoreV1Connector(vllm_config, KVConnectorRole.WORKER, cache_config)
    worker = worker_connector.worker
    assert worker is not None and not worker._bound
    if layerwise:
        worker._executor._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    caches = {
        layer.layer_names[0]: torch.arange(64, dtype=torch.float32).reshape(8, 4, 2, 1) + layer.physical_layer_id * 100
        for layer in topology.groups[0].layers
    }
    originals = {name: tensor[2].clone() for name, tensor in caches.items()}
    codec = LookupCodec()

    class LocalLookup:
        def lookup(self, request):
            # Exercise the same message encoding as RPC without requiring an IPC socket.
            decoded = codec.decode_request(codec.encode_request(request))
            return codec.decode_result(codec.encode_result(worker.lookup(decoded)))

        def close(self):
            pass

    monkeypatch.setattr(connector_module, "LookupClient", lambda *_args, **_kwargs: LocalLookup())
    scheduler_connector = AscendStoreV1Connector(vllm_config, KVConnectorRole.SCHEDULER, cache_config)
    try:
        with pytest.raises(RuntimeError, match="bound"):
            worker.lookup(LookupRequest(0, 8, (0,), (b"a", b"b")))
        assert not sdk.regions
        worker_connector.register_kv_caches(caches)
        with pytest.raises(RuntimeError, match="rebound"):
            worker_connector.register_kv_caches(caches)
        store = RangeStoreCommand("producer", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17)
        worker_connector.bind_connector_metadata(KVTransferStep(store=StoreCommandBatch((store,))))
        worker_connector.start_load_kv(None)
        assert not sdk.readable
        for name in caches:
            worker_connector.save_kv_layer(name, caches[name], None)
        worker_connector.wait_for_save()
        worker_connector.handle_preemptions(KVTransferStep(released_request_ids=frozenset({"producer"})))
        assert worker_connector.build_connector_worker_meta().released_store_jobs == {17: 1}
        assert worker_connector.build_connector_worker_meta() is None
        worker_connector.get_finished(set())
        assert not worker_connector.get_block_ids_with_load_errors()
        worker_connector.clear_connector_metadata()
        assert len(sdk.readable) == 2

        if load_failure:
            if mode == "gva":
                monkeypatch.setattr(sdk, "batch_copy", lambda *_: -7)
            else:
                method = "batch_get_into_multi_buffer_ranges" if layerwise else "batch_get_into_multi_buffers"
                monkeypatch.setattr(sdk, method, lambda keys, *_: [-7] * len(keys))
        request = make_request("consumer", prompt_tokens=8, tokens=9)
        request.block_hashes = [b"a", b"b"]
        assert scheduler_connector.get_num_new_matched_tokens(request, 4) == (4, async_load)
        scheduler_connector.update_state_after_alloc(request, FakeBlocks([5, 6]), 4)
        output = (
            make_output()
            if async_load
            else make_output(new=(new_request_data("consumer", ([5, 6],), 8),), scheduled_tokens={"consumer": 1})
        )
        step = scheduler_connector.build_connector_meta(output)
        assert [(c.start_token, c.end_token) for c in step.load_commands] == [(4, 8)]
        for tensor in caches.values():
            tensor[5:7].fill_(-1)
        worker_connector.handle_preemptions(step)
        worker_connector.bind_connector_metadata(step)
        worker_connector.start_load_kv(None)
        for name in caches:
            worker_connector.wait_for_layer_load(name)
        worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
        worker_connector.wait_for_save()
        assert worker_connector.get_finished(set()) == (set(), {"consumer"})
        assert worker_connector.get_block_ids_with_load_errors() == ({6} if load_failure else set())
        worker_connector.clear_connector_metadata()
        for name, tensor in caches.items():
            if load_failure:
                assert torch.all(tensor[6] == -1)
            else:
                torch.testing.assert_close(tensor[6], originals[name])
            assert torch.all(tensor[5] == -1)
        # A no-forward, empty step still follows the full hook lifecycle.
        worker_connector.handle_preemptions(KVTransferStep())
        worker_connector.bind_connector_metadata(KVTransferStep())
        worker_connector.start_load_kv(None)
        assert worker_connector.get_finished(set()) == (set(), set())
        worker_connector.get_block_ids_with_load_errors()
        worker_connector.clear_connector_metadata()
        worker_connector.clear_connector_metadata()
    finally:
        scheduler_connector.shutdown()
        worker_connector.shutdown()
    assert sdk.closed and not sdk.regions


@pytest.mark.parametrize("failure", ("prepare", "context", "register", "start"))
def test_cache_binding_failure_closes_owned_resources_and_rejects_retry(monkeypatch, failure):
    topology = make_topology()
    backend = FakeBackend()
    resources = Resources(backend, make_backend_spec(), topology.transfer_groups)
    worker = Worker(topology, resources, coordinator=make_coordinator(topology))
    if failure == "context":
        monkeypatch.setattr(worker_module, "bind_context", Mock(side_effect=RuntimeError("binding failed")))
    else:
        method = {"prepare": "prepare_kv_caches", "register": "register_kv_buffers", "start": "start_executors"}[
            failure
        ]
        monkeypatch.setattr(resources, method, Mock(side_effect=RuntimeError("binding failed")))
    caches = {layer.layer_names[0]: torch.zeros(8, 4, 2, 1) for layer in topology.groups[0].layers}
    with pytest.raises(RuntimeError, match="binding failed"):
        worker.bind_kv_caches(caches)
    assert backend.closed and resources.executors_stopped and resources.kv_caches is None
    with pytest.raises(RuntimeError, match="rebound"):
        worker.bind_kv_caches(caches)
    worker.close()


def test_close_before_registration_does_not_access_uncreated_services():
    topology = make_topology()
    backend = FakeBackend()
    resources = Resources(backend, make_backend_spec(), topology.transfer_groups)
    worker = Worker(topology, resources, coordinator=make_coordinator(topology))
    worker.close()
    worker.close()
    assert backend.closed and resources.kv_caches is None
    assert not any(call[0] == "register_buffer" for call in backend.calls)
