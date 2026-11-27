# SPDX-License-Identifier: Apache-2.0
"""Compatibility surface tests for the mooncake V1 module-boundary refactor.

The two entry modules must keep re-exporting the exact objects that live in the
mooncake_v1 package: entry attributes are aliases, never wrappers, so
isinstance/type checks and test patches that still target the entry name keep
hitting the real classes and functions.
"""

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_connector as pull_entry,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_layerwise_connector as layerwise_entry,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.base import (
    connector as base_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.base import (
    scheduler as base_scheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common import (
    constants,
    transport,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common import (
    protocol as common_protocol,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import (
    connector as layerwise_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import (
    protocol as layerwise_protocol,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import (
    scheduler as layerwise_scheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import (
    threads as layerwise_threads,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise import (
    worker as layerwise_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import (
    connector as pull_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import (
    reformat as pull_reformat,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import (
    scheduler as pull_scheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import (
    threads as pull_threads,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull import (
    worker as pull_worker,
)


def test_pull_entry_reexports_package_objects():
    assert pull_entry.KVCacheSendingThread is pull_threads.KVCacheSendingThread
    assert pull_entry.KVCacheRecvingThread is pull_threads.KVCacheRecvingThread
    assert pull_entry.MooncakeConnectorWorker is pull_worker.MooncakeConnectorWorker
    assert pull_entry.MooncakeConnectorScheduler is pull_scheduler.MooncakeConnectorScheduler
    assert issubclass(pull_scheduler.MooncakeConnectorScheduler, base_scheduler.MooncakeConnectorSchedulerBase)
    assert pull_entry.SHUTDOWN_JOIN_TIMEOUT_SECONDS is constants.SHUTDOWN_JOIN_TIMEOUT_SECONDS
    assert pull_entry.KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS is constants.KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS
    assert pull_entry.THREAD_STOP_POLL_SECONDS is constants.THREAD_STOP_POLL_SECONDS


def test_layerwise_entry_reexports_package_objects():
    assert layerwise_entry.KVCacheSendingLayerThread is layerwise_threads.KVCacheSendingLayerThread
    assert layerwise_entry.KVCacheRecvingLayerThread is layerwise_threads.KVCacheRecvingLayerThread
    assert layerwise_entry.group_contiguous_ids is layerwise_threads.group_contiguous_ids
    assert layerwise_entry.group_concurrent_contiguous is layerwise_threads.group_concurrent_contiguous
    assert layerwise_entry.MooncakeLayerwiseConnectorWorker is layerwise_worker.MooncakeLayerwiseConnectorWorker
    assert (
        layerwise_entry.MooncakeLayerwiseConnectorScheduler is layerwise_scheduler.MooncakeLayerwiseConnectorScheduler
    )
    assert issubclass(
        layerwise_scheduler.MooncakeLayerwiseConnectorScheduler,
        base_scheduler.MooncakeConnectorSchedulerBase,
    )
    for name in ("ReqMeta", "SendTask", "TransferMeta", "SendReqInfo", "MooncakeLayerwiseConnectorMetadata"):
        assert getattr(layerwise_entry, name) is getattr(layerwise_protocol, name)
    assert layerwise_entry.SHUTDOWN_JOIN_TIMEOUT_SECONDS is constants.SHUTDOWN_JOIN_TIMEOUT_SECONDS
    assert layerwise_entry.KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS is constants.KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS
    assert layerwise_entry.ensure_zmq_recv_with_poller is transport.ensure_zmq_recv_with_poller
    # Back-compat alias of the poller-based variant, not the common ensure_zmq_recv.
    assert layerwise_entry.ensure_zmq_recv is transport.ensure_zmq_recv_with_poller
    assert layerwise_entry.get_external_request_id is common_protocol.get_external_request_id


def test_entry_keeps_historical_protocol_constants_and_msgspec():
    """Historical module attributes stay importable from the entry modules."""
    for name in ("GET_META_MSG", "DONE_RECVING_MSG"):
        assert getattr(pull_entry, name) is getattr(common_protocol, name)
    assert pull_entry.KV_CACHE_BUFFER_ALIGNMENT is pull_worker.KV_CACHE_BUFFER_ALIGNMENT
    for name in ("GET_META_MSG", "DONE_SENDING_MSG", "FAILED_SENDING_MSG", "get_external_request_id"):
        assert getattr(layerwise_entry, name) is getattr(common_protocol, name)
    assert layerwise_entry.msgspec.__name__ == "msgspec"


def test_layerwise_metadata_uses_package_protocol_objects():
    """The entry-defined metadata class reads the same ReqMeta/SendTask objects."""
    globals_ = layerwise_entry.MooncakeLayerwiseConnectorMetadata.__init__.__globals__
    assert globals_["ReqMeta"] is layerwise_protocol.ReqMeta
    assert globals_["SendTask"] is layerwise_protocol.SendTask


def test_entry_facades_are_package_classes_with_shared_base():
    assert pull_entry.MooncakeConnector is pull_connector.MooncakeConnector
    assert layerwise_entry.MooncakeLayerwiseConnector is layerwise_connector.MooncakeLayerwiseConnector
    assert issubclass(pull_connector.MooncakeConnector, base_connector.MooncakeConnectorBase)
    assert issubclass(layerwise_connector.MooncakeLayerwiseConnector, base_connector.MooncakeConnectorBase)
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, SupportsHMA

    for connector in (pull_connector.MooncakeConnector, layerwise_connector.MooncakeLayerwiseConnector):
        mro = connector.__mro__
        assert KVConnectorBase_V1 in mro
        assert SupportsHMA in mro


def test_is_conv_state_dim_first_kept_importable_and_same_implementation():
    assert pull_entry.is_conv_state_dim_first is pull_reformat.is_conv_state_dim_first
    assert callable(pull_entry.is_conv_state_dim_first)


def test_connector_factory_lazy_load_resolves_entry_classes(monkeypatch):
    from vllm_ascend.distributed.kv_transfer import register_connector

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()
    assert KVConnectorFactory.get_connector_class_by_name("MooncakeConnectorV1") is pull_entry.MooncakeConnector
    assert (
        KVConnectorFactory.get_connector_class_by_name("MooncakeLayerwiseConnector")
        is layerwise_entry.MooncakeLayerwiseConnector
    )


def test_profiling_symbol_paths_stay_resolvable():
    chains = [
        (pull_entry, "MooncakeConnectorScheduler", "build_connector_meta"),
        (pull_entry, "MooncakeConnector", "start_load_kv"),
        (pull_entry, "MooncakeConnectorWorker", "start_load_kv"),
        (pull_entry, "KVCacheRecvingThread", "add_request"),
        (pull_entry, "KVCacheRecvingThread", "_transfer_kv_cache_all_groups"),
        # profiling_config also names a pull KVCacheRecvingThread._transfer_kv_cache
        # path; that symbol never existed on upstream main (only the layerwise
        # thread has it), so it is a pre-existing stale entry, not touched here.
        (layerwise_entry, "KVCacheSendingLayerThread", "_transfer_kv_cache"),
        (pull_entry, "KVCacheTaskTracker", "update_done_task_count"),
        (pull_entry, "MooncakeConnectorWorker", "get_finished"),
        (pull_entry, "MooncakeConnector", "get_finished"),
    ]
    for module, attr, method in chains:
        assert hasattr(getattr(module, attr), method), f"{attr}.{method}"
