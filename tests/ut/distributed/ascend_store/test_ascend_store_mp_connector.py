from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.kv_events import BlockStored

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector import (
    AscendStoreMPConnector,
    AscendStoreMPConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.npu_ipc import (
    KVCacheStorageSpec,
    export_worker_kv_caches,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import MPServerUnavailableError

# isort: on

CONNECTOR_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector"
SERVER_URL = "ipc:///tmp/ascend_store_mp_test"
_UNSET_SERVER_URL = object()


class _CPUMemoryAdapter:
    @staticmethod
    def export_storage(storage: torch.Tensor) -> KVCacheStorageSpec:
        return KVCacheStorageSpec(
            size_bytes=storage.untyped_storage().nbytes(),
            device_type="cpu",
            device_uuid="cpu",
            handle_type="test_cpu",
            handle_version=1,
            handle=b"handle",
        )


def _make_vllm_config(server_url: object = SERVER_URL, rank: int = 0) -> MagicMock:
    config = MagicMock()
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.rank = rank
    config.kv_transfer_config.kv_connector = "AscendStoreConnector"
    config.kv_transfer_config.engine_id = "engine-0"
    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.kv_connector_extra_config = {}
    if server_url is not _UNSET_SERVER_URL:
        config.kv_transfer_config.kv_connector_extra_config["kv_cache_server_url"] = server_url
    return config


def _make_kv_cache_config() -> MagicMock:
    config = MagicMock()
    config.kv_cache_groups[0].kv_cache_spec.page_size_bytes = 1024
    return config


@pytest.mark.parametrize(("use_layerwise", "expected"), [(False, False), (True, True)])
def test_connector_requires_piecewise_for_layerwise_mode(use_layerwise: bool, expected: bool) -> None:
    assert AscendStoreMPConnector.requires_piecewise_for_cudagraph({"use_layerwise": use_layerwise}) is expected


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_connector_registers_its_role(role: KVConnectorRole) -> None:
    config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(config, role, kv_cache_config)

        client_class.assert_called_once_with(SERVER_URL)
        if role == KVConnectorRole.SCHEDULER:
            client_class.return_value.register_scheduler.assert_called_once_with(config, kv_cache_config, 1024)
            client_class.return_value.register_worker.assert_not_called()
        else:
            client_class.return_value.register_worker.assert_called_once_with(config, kv_cache_config)
            client_class.return_value.register_scheduler.assert_not_called()

        connector.shutdown()
        client_class.return_value.close.assert_called_once_with()


def test_connector_uses_default_server_url(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_STORE_SERVER_URL", raising=False)
    config = _make_vllm_config(server_url=_UNSET_SERVER_URL)

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())

    client_class.assert_called_once_with("tcp://127.0.0.1:5555")


def test_connector_rejects_invalid_default_server_url(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_STORE_SERVER_URL", "")
    config = _make_vllm_config(server_url=_UNSET_SERVER_URL)

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        pytest.raises(ValueError, match="VLLM_ASCEND_STORE_SERVER_URL"),
    ):
        AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())

    client_class.assert_not_called()


@pytest.mark.parametrize("use_layerwise", [False, True])
def test_scheduler_lookup_delegates_to_kv_cache_client(use_layerwise: bool) -> None:
    config = _make_vllm_config()
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = use_layerwise
    kv_cache_config = _make_kv_cache_config()
    request = MagicMock()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.lookup.return_value = (16, False)
        connector = AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)

        result = connector.get_num_new_matched_tokens(request, 32)

        assert result == (16, False)
        client_class.return_value.lookup.assert_called_once_with(request, 32)


def test_worker_cannot_call_scheduler_lookup() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        with pytest.raises(RuntimeError, match="only available on the scheduler connector"):
            connector.get_num_new_matched_tokens(MagicMock(), 32)

        client_class.return_value.lookup.assert_not_called()


def test_worker_registers_process_neutral_kv_cache_layouts() -> None:
    config = _make_vllm_config()
    storage = torch.empty((4, 8), dtype=torch.float16)

    def export(caches):
        return export_worker_kv_caches(caches, _CPUMemoryAdapter())

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", side_effect=export),
    ):
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.register_kv_caches({"layer.0": storage, "layer.1": storage[1:]})

    spec = client_class.return_value.register_kv_caches.call_args.args[0]
    first = spec.caches["layer.0"][0]
    second = spec.caches["layer.1"][0]
    assert first.storage_index == second.storage_index == 0
    assert first.storage_offset_bytes == 0
    assert second.storage_offset_bytes == storage.stride(0) * storage.element_size()
    assert first.shape == (4, 8)
    assert first.stride == storage.stride()
    assert first.dtype == "torch.float16"
    assert len(spec.storages) == 1
    assert spec.storages[0].size_bytes == storage.untyped_storage().nbytes()
    assert spec.storages[0].device_type == "cpu"


def test_worker_rejects_empty_kv_caches_before_rpc() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        with pytest.raises(ValueError, match="must not be empty"):
            connector.register_kv_caches({})

        client_class.return_value.register_kv_caches.assert_not_called()


def test_worker_wait_for_save_releases_source_event_when_rpc_fails() -> None:
    config = _make_vllm_config()
    metadata = AscendConnectorMetadata(set(), set())
    metadata.add_request(ReqMeta("request-0", can_save=True))
    exported_event = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.record_npu_event", return_value=exported_event),
    ):
        client_class.return_value.wait_for_save.side_effect = RuntimeError("RPC failed")
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.bind_connector_metadata(metadata)
        with pytest.raises(RuntimeError, match="RPC failed"):
            connector.wait_for_save()

    client_class.return_value.wait_for_save.assert_called_once_with(metadata, exported_event.spec)
    exported_event.close.assert_called_once_with()


def test_worker_layerwise_delegates_load_and_save_per_layer() -> None:
    config = _make_vllm_config()
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
    metadata = AscendConnectorMetadata(set(), set())
    exported_event = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.record_npu_event", return_value=exported_event),
    ):
        client_class.return_value.start_load_kv.return_value = True
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.bind_connector_metadata(metadata)

        connector.start_load_kv(MagicMock())
        connector.wait_for_layer_load("model.layers.0.self_attn")
        connector.save_kv_layer("model.layers.0.self_attn", MagicMock(), MagicMock())

    client_class.return_value.start_load_kv.assert_called_once_with(metadata)
    client_class.return_value.wait_for_layer_load.assert_called_once_with()
    client_class.return_value.save_kv_layer.assert_called_once_with(exported_event.spec)
    exported_event.close.assert_called_once_with()


def test_worker_layerwise_load_preserves_rpc_error() -> None:
    config = _make_vllm_config()
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
    error = MPServerUnavailableError("transport disconnected")

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.wait_for_layer_load.side_effect = error
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        with pytest.raises(MPServerUnavailableError, match="transport disconnected") as exc_info:
            connector.wait_for_layer_load("model.layers.0.self_attn")

    assert exc_info.value is error


def test_worker_layerwise_save_preserves_rpc_error_and_releases_event() -> None:
    config = _make_vllm_config()
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
    error = MPServerUnavailableError("transport disconnected")
    exported_event = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.record_npu_event", return_value=exported_event),
    ):
        client_class.return_value.save_kv_layer.side_effect = error
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        with pytest.raises(MPServerUnavailableError, match="transport disconnected") as exc_info:
            connector.save_kv_layer("model.layers.0.self_attn", MagicMock(), MagicMock())

    assert exc_info.value is error
    exported_event.close.assert_called_once_with()


def test_worker_rejected_store_releases_delayed_request() -> None:
    config = _make_vllm_config()
    metadata = AscendConnectorMetadata(set(), set(), delayed_free_req_ids={"request-0"})
    metadata.add_request(ReqMeta("request-0", can_save=True))
    exported_event = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.record_npu_event", return_value=exported_event),
    ):
        client_class.return_value.wait_for_save.return_value = False
        client_class.return_value.get_finished.return_value = (set(), set())
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.bind_connector_metadata(metadata)

        connector.wait_for_save()

        assert connector.get_finished({"request-0"}) == ({"request-0"}, set())
        assert connector.get_finished({"request-0"}) == (set(), set())

    exported_event.close.assert_called_once_with()


def test_worker_get_finished_delegates_with_bound_metadata() -> None:
    config = _make_vllm_config()
    metadata = AscendConnectorMetadata(set(), set())

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.get_finished.return_value = ({"saving-0"}, {"loading-0"})
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.bind_connector_metadata(metadata)

        result = connector.get_finished({"saving-0", "loading-0"})

    assert result == ({"saving-0"}, {"loading-0"})
    client_class.return_value.get_finished.assert_called_once_with({"saving-0", "loading-0"}, metadata)


def test_worker_get_finished_returns_empty_sets_for_degraded_metadata() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.bind_connector_metadata(AscendStoreMPConnectorMetadata())

        assert connector.get_finished({"request-0"}) == (set(), set())

    client_class.return_value.get_finished.assert_not_called()


def test_worker_build_connector_meta_delegates_to_client() -> None:
    config = _make_vllm_config()
    metadata = AscendStoreKVConnectorWorkerMetadata({7: 1})

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.build_connector_worker_meta.return_value = metadata
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        assert connector.build_connector_worker_meta() is metadata

    client_class.return_value.build_connector_worker_meta.assert_called_once_with()


def test_worker_kv_events_are_wrapped_for_vllm_aggregation() -> None:
    config = _make_vllm_config()
    event = BlockStored(
        block_hashes=[b"hash-0"],
        parent_block_hash=None,
        token_ids=[1],
        block_size=1,
        lora_id=None,
        medium="CPU",
        lora_name=None,
    )

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.get_kv_events.return_value = [event]
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        events = connector.get_kv_connector_kv_cache_events()

    assert events is not None
    assert events.get_all_events() == [event]
    assert events.get_number_of_workers() == 1


def test_worker_kv_events_return_none_when_no_events_are_available() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.get_kv_events.return_value = []
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        assert connector.get_kv_connector_kv_cache_events() is None


def _make_load_metadata() -> AscendConnectorMetadata:
    metadata = AscendConnectorMetadata(set(), set())
    metadata.add_request(
        ReqMeta(
            "request-0",
            token_len_chunk=16,
            block_ids=[7, 8],
            block_hashes=[b"hash-0"],
            load_spec=LoadSpec(0, 16, True),
        )
    )
    return metadata


def test_worker_start_load_delegates_and_returns_server_load_errors() -> None:
    config = _make_vllm_config()
    metadata = _make_load_metadata()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.start_load_kv.return_value = True
        client_class.return_value.get_block_ids_with_load_errors.return_value = {8}
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.bind_connector_metadata(metadata)

        connector.start_load_kv(MagicMock())

        assert connector.get_block_ids_with_load_errors() == {8}

    client_class.return_value.start_load_kv.assert_called_once_with(metadata)


def test_worker_start_load_marks_candidate_blocks_invalid_when_rpc_fails() -> None:
    config = _make_vllm_config()
    metadata = _make_load_metadata()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.start_load_kv.return_value = False
        client_class.return_value.get_block_ids_with_load_errors.return_value = set()
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.bind_connector_metadata(metadata)

        connector.start_load_kv(MagicMock())

        assert connector.get_block_ids_with_load_errors() == {7, 8}


def test_worker_load_error_query_failure_invalidates_pending_load_blocks() -> None:
    config = _make_vllm_config()
    metadata = _make_load_metadata()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.start_load_kv.return_value = True
        client_class.return_value.get_block_ids_with_load_errors.return_value = None
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.bind_connector_metadata(metadata)

        connector.start_load_kv(MagicMock())

        assert connector.get_block_ids_with_load_errors() == {7, 8}


def test_worker_keeps_exported_cache_alive_until_shutdown() -> None:
    config = _make_vllm_config()
    exported = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", return_value=exported),
    ):
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.register_kv_caches({"layer.0": MagicMock()})

        exported.close.assert_not_called()
        connector.shutdown()

    client_class.return_value.register_kv_caches.assert_called_once_with(exported.spec)
    exported.close.assert_called_once_with()


def test_worker_releases_exported_cache_when_registration_fails() -> None:
    config = _make_vllm_config()
    exported = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", return_value=exported),
    ):
        client_class.return_value.register_kv_caches.side_effect = RuntimeError("registration failed")
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        with pytest.raises(RuntimeError, match="registration failed"):
            connector.register_kv_caches({"layer.0": MagicMock()})

    exported.close.assert_called_once_with()


def test_worker_keeps_exported_cache_while_registration_recovers() -> None:
    config = _make_vllm_config()
    exported = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", return_value=exported),
    ):
        client_class.return_value.register_kv_caches.return_value = False
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.register_kv_caches({"layer.0": MagicMock()})

        exported.close.assert_not_called()
        connector.shutdown()

    exported.close.assert_called_once_with()


def test_worker_rejects_a_second_cache_registration() -> None:
    config = _make_vllm_config()
    exported = MagicMock()

    with (
        patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class,
        patch(f"{CONNECTOR_MODULE}.export_worker_kv_caches", return_value=exported) as export,
    ):
        connector = AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())
        connector.register_kv_caches({"layer.0": MagicMock()})
        with pytest.raises(RuntimeError, match="already registered"):
            connector.register_kv_caches({"layer.0": MagicMock()})
        connector.shutdown()

    export.assert_called_once()
    client_class.return_value.register_kv_caches.assert_called_once_with(exported.spec)
    exported.close.assert_called_once_with()


def test_connector_rejects_sleep_mode() -> None:
    config = _make_vllm_config()
    config.model_config.enable_sleep_mode = True

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        with pytest.raises(ValueError, match="does not support sleep mode"):
            AscendStoreMPConnector(config, KVConnectorRole.WORKER, _make_kv_cache_config())

        client_class.assert_not_called()


def test_build_connector_meta_returns_empty_metadata_when_degraded() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.build_connector_meta.return_value = None
        connector = AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())
        metadata = connector.build_connector_meta(MagicMock())

    assert isinstance(metadata, AscendStoreMPConnectorMetadata)


def test_connector_closes_client_when_registration_fails() -> None:
    config = _make_vllm_config()

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        client_class.return_value.register_scheduler.side_effect = RuntimeError("registration failed")

        with pytest.raises(RuntimeError, match="registration failed"):
            AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())

        client_class.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("server_url", [None, "", 123])
def test_connector_rejects_invalid_server_url(server_url: object) -> None:
    config = _make_vllm_config(server_url)

    with patch(f"{CONNECTOR_MODULE}.KVCacheClient") as client_class:
        with pytest.raises(ValueError, match="kv_cache_server_url"):
            AscendStoreMPConnector(config, KVConnectorRole.SCHEDULER, _make_kv_cache_config())

        client_class.assert_not_called()
