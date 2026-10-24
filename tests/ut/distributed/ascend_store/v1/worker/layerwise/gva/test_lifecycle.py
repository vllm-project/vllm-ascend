"""Factory-acquired GVA communication follows the Worker's cleanup boundary."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeBackend, make_pool_config, make_topology
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import factory
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import BackendSpec, LayerwiseAccessKind
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import worker as layerwise_worker


@pytest.mark.parametrize("failure", ("channel", "sessions", "worker", None))
def test_factory_retains_cleanup_before_worker_initialization(configuration_case, monkeypatch, failure):
    config, cache = configuration_case
    topology = make_topology(tp_size=2, dcp_size=2, put_step=2)
    topology = replace(topology, cp_partition=replace(topology.cp_partition, object_shard_count=2))
    backend = FakeBackend()
    sessions = Mock()
    create_sessions = Mock(return_value=sessions)
    monkeypatch.setattr(backend, "create_sessions", create_sessions)
    spec = BackendSpec("fake_gva", lambda *_args, **_kwargs: backend, LayerwiseAccessKind.GVA, False)
    pool = make_pool_config(topology, "fake_gva", use_layerwise=True)
    monkeypatch.setattr(factory, "resolve_kv_pool_config", lambda *_: pool)
    monkeypatch.setattr(factory, "resolve_worker_topology", lambda *_: topology)
    monkeypatch.setattr(factory, "resolve_backend_spec", lambda _: spec)
    events = []
    channel = Mock()
    channel.close.side_effect = lambda: events.append("channel")
    sessions.close.side_effect = lambda: events.append("sessions")
    monkeypatch.setattr(backend, "close", lambda: events.append("backend"))
    acquire = Mock(return_value=channel)
    monkeypatch.setattr(factory, "StoreCollective", acquire)
    if failure == "channel":
        acquire.side_effect = RuntimeError("channel initialization failed")
    elif failure == "sessions":
        create_sessions.side_effect = RuntimeError("sessions initialization failed")
    elif failure == "worker":
        # Fail after session ownership was bound, before executor creation.
        fail = Mock(side_effect=RuntimeError("worker initialization failed"))
        monkeypatch.setattr(layerwise_worker, "compile_layer_ids_by_name", fail)

    if failure is not None:
        with pytest.raises(RuntimeError, match=f"{failure} initialization failed"):
            factory.create_kv_pool_worker(config, cache)
    else:
        worker = factory.create_kv_pool_worker(config, cache)
        assert not events
        worker.close()
        worker.close()
    acquire.assert_called_once_with(topology.context_global_ranks)
    expected = ["backend"] if failure == "channel" else ["channel", "backend"]
    if failure in (None, "worker"):
        expected.insert(0, "sessions")
    assert events == expected
