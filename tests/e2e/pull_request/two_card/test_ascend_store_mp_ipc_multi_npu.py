"""Two-NPU end-to-end: one KVCacheServer serves TP=2 Workers on separate devices.

Each Worker exports its own KV cache through an NPU IPC handle; the server
resolves both physical device UUIDs, binds one real Mooncake backend per
Worker with the resolved device index, and both Store/Retrieve round trips
run against real Mooncake. No Worker local_rank ever crosses the RPC.
"""

import contextlib
import json
import multiprocessing
import threading
import traceback
from functools import partial
from multiprocessing.connection import Connection
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests.e2e.pull_request.one_card.test_ascend_store_mp_ipc import (
    _MOONCAKE_MEMORY_ALIGNMENT_BYTES,
    _MOONCAKE_TEST_NUM_BLOCKS,
    _SERVER_URL,
    _receive,
    _request_server_stop,
    _stop_process,
    _wait_for_active_export,
    _wait_for_mooncake_master,
    _wait_until_connected,
    _wait_until_registered,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.npu_ipc import WorkerKVCacheSpec
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import WorkerRegistration


class _ModelConfig:
    def __init__(self):
        self.model = "org/llama-multi-npu-test"
        self.max_model_len = 1024
        self.use_mla = False
        self.hf_text_config = SimpleNamespace()
        self.hf_config = self.hf_text_config

    @staticmethod
    def get_num_layers(_parallel_config) -> int:
        return 2

    @staticmethod
    def get_total_num_kv_heads() -> int:
        return 2


class _ObservingMooncakeBackend:
    """Real MP Mooncake backend plus the light instrumentation the test reads."""

    def __init__(self, parallel_config, device_index: int, lazy_init: bool):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.backend.mooncake import (
            MPMooncakeBackend,
        )

        self._backend = MPMooncakeBackend(parallel_config, device_index, lazy_init=lazy_init)
        self.device_index = device_index
        self.stored_keys: list[str] = []

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def put(self, keys: list[str], addrs, sizes) -> None:
        self.stored_keys.extend(keys)
        self._backend.put(keys, addrs, sizes)


class _MultiNPUObservedWorker:
    """Expose per-Worker assertions without shipping multi-MiB tensors over the pipe."""

    def __init__(self, registration: "WorkerRegistration", connection: Connection):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.worker import MPKVPoolWorker

        self._connection = connection
        self._backend: _ObservingMooncakeBackend | None = None
        self._backend_creation_count = 0
        self._worker = MPKVPoolWorker(
            registration.config,
            kv_cache_config=registration.config.build_kv_cache_config(),
            rank=registration.identity.rank,
            backend_factory=self._create_backend,
        )

    def _create_backend(self, _parallel_config, device_index: int | None, lazy_init: bool):
        assert device_index is not None, "the MP worker resolves the device index before creating the backend"
        self._backend_creation_count += 1
        self._backend = _ObservingMooncakeBackend(_parallel_config, device_index, lazy_init)
        return self._backend

    def configure_kv_caches(self, spec: "WorkerKVCacheSpec") -> None:
        self._worker.configure_kv_caches(spec)
        assert self._backend is not None
        self._connection.send(
            (
                "configured",
                {
                    "backend_device_index": self._backend.device_index,
                    "backend_creation_count": self._backend_creation_count,
                },
            )
        )

    def start_load_kv(self, metadata) -> None:
        self._worker.start_load_kv(metadata)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._worker.get_block_ids_with_load_errors()

    def wait_for_save(self, metadata, event_spec) -> None:
        self._worker.wait_for_save(metadata, event_spec)
        assert self._backend is not None
        self._connection.send(("stored", {"stored_key_count": len(self._backend.stored_keys)}))

    def get_finished(self, finished_req_ids, metadata):
        return self._worker.get_finished(finished_req_ids, metadata)

    def close(self) -> None:
        current_caches = getattr(self._worker, "kv_caches", None)
        self._worker.close()
        self._connection.send(
            (
                "closed",
                {
                    "mapping_released": current_caches == {},
                    "worker_caches_empty": self._worker.kv_caches == {},
                },
            )
        )


def _make_tp2_worker_config(rank: int, server_url: str):
    return SimpleNamespace(
        model_config=_ModelConfig(),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            rank=rank,
            world_size=2,
            data_parallel_index=0,
            data_parallel_size=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="ascend-store-mp-multi-npu-test",
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"backend": "mooncake", "kv_cache_server_url": server_url},
            is_kv_producer=True,
        ),
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_events_config=None,
    )


def _create_rank_routed_worker(registration: "WorkerRegistration", connections: dict[int, Connection]):
    return _MultiNPUObservedWorker(registration, connections[registration.identity.rank])


def _run_multi_npu_server(
    endpoint_connection: Connection,
    observation_connections: dict[int, Connection],
    control_connection: Connection,
) -> None:
    server = None
    control_thread = None
    registered_locations: set[str] = set()
    try:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheServer
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.backend.mooncake import (
            global_te,
        )

        # Record the location string every real register_memory receives, so
        # the test can prove Mooncake was handed npu:0 and npu:1.
        real_get_transfer_engine = global_te.get_transfer_engine

        def _recording_get_transfer_engine(*args, **kwargs):
            engine = real_get_transfer_engine(*args, **kwargs)

            class _RecordingEngine:
                def register_memory(self, ptr, length, location):
                    registered_locations.add(location)
                    return engine.register_memory(ptr, length, location)

                def __getattr__(self, name):
                    return getattr(engine, name)

            return _RecordingEngine()

        global_te.get_transfer_engine = _recording_get_transfer_engine

        worker_factory = partial(_create_rank_routed_worker, connections=observation_connections)
        server = KVCacheServer(_SERVER_URL, scheduler_threads=2, worker_threads=2, worker_factory=worker_factory)
        control_thread = threading.Thread(
            target=_request_server_stop,
            args=(server, control_connection),
            daemon=True,
            name="ascend-store-mp-multi-npu-stop",
        )
        control_thread.start()
        endpoint_connection.send(("ready", server.endpoint))
        endpoint_connection.close()
        server.run()
    except BaseException:
        error = traceback.format_exc()
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            endpoint_connection.send(("error", error))
        raise
    finally:
        if server is not None and not server.close():
            server.abort()
        if control_thread is not None:
            control_thread.join(10.0)
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            observation_connections[0].send(("locations", sorted(registered_locations)))
        endpoint_connection.close()
        for connection in observation_connections.values():
            connection.close()
        control_connection.close()


def _allocate_worker_caches(device_index: int, first_fill: float, second_fill: float):
    import torch

    element_size = torch.empty((), dtype=torch.float16).element_size()
    elements_per_layer = _MOONCAKE_MEMORY_ALIGNMENT_BYTES // element_size
    allocation = torch.empty(elements_per_layer * 3, dtype=torch.float16, device=f"npu:{device_index}")
    alignment_offset = (-allocation.data_ptr()) % _MOONCAKE_MEMORY_ALIGNMENT_BYTES
    aligned_offset = alignment_offset // element_size
    cache_storage = allocation[aligned_offset : aligned_offset + 2 * elements_per_layer]

    # Ascend Transport rejects synthetic buffers below its 2 MiB alignment.
    assert cache_storage.data_ptr() % _MOONCAKE_MEMORY_ALIGNMENT_BYTES == 0
    assert cache_storage.numel() * element_size % _MOONCAKE_MEMORY_ALIGNMENT_BYTES == 0

    block_elements = elements_per_layer // _MOONCAKE_TEST_NUM_BLOCKS
    first_layer = cache_storage[:elements_per_layer].view(_MOONCAKE_TEST_NUM_BLOCKS, block_elements)
    second_layer = cache_storage[elements_per_layer:].view(_MOONCAKE_TEST_NUM_BLOCKS, block_elements)
    first_layer.fill_(first_fill)
    second_layer.fill_(second_fill)
    return first_layer, second_layer


def _store_connector(connector, request_id: str, block_hash: str) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, ReqMeta

    store_metadata = AscendConnectorMetadata(set(), set(), delayed_free_req_ids={request_id})
    store_metadata.add_request(
        ReqMeta(
            request_id,
            token_len_chunk=16,
            block_ids=[1],
            block_hashes=[block_hash],
            can_save=True,
        )
    )
    connector.bind_connector_metadata(store_metadata)
    connector.wait_for_save()
    assert connector.get_finished({request_id}) == ({request_id}, set())
    connector.clear_connector_metadata()


def _retrieve_connector(connector, block_hash: str) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
        AscendConnectorMetadata,
        LoadSpec,
        ReqMeta,
    )

    load_metadata = AscendConnectorMetadata(set(), set())
    load_metadata.add_request(
        ReqMeta(
            "load-request",
            token_len_chunk=16,
            block_ids=[1],
            block_hashes=[block_hash],
            load_spec=LoadSpec(0, 16, True),
        )
    )
    connector.bind_connector_metadata(load_metadata)
    connector.start_load_kv(None)
    assert connector.get_block_ids_with_load_errors() == set()
    connector.clear_connector_metadata()


def test_mooncake_two_npu_workers_store_and_retrieve(tmp_path, monkeypatch) -> None:
    import torch
    import torch_npu  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.utils.network_utils import get_open_port

    from tests.e2e.conftest import MooncakeLauncher
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_mp_connector import (
        AscendStoreMPConnector,
    )

    if not torch.npu.is_available() or torch.npu.device_count() < 2:
        pytest.skip("Two NPUs are required for the multi-NPU IPC round trip")

    context = multiprocessing.get_context("spawn")
    endpoint_connection, endpoint_child_connection = context.Pipe()
    observation_connections = {}
    observation_child_connections = {}
    for rank in (0, 1):
        parent, child = context.Pipe()
        observation_connections[rank] = parent
        observation_child_connections[rank] = child
    control_connection, control_child_connection = context.Pipe()
    server = None
    connectors = []
    failure: BaseException | None = None
    server_exitcode = None
    server_forced = False

    master_port = get_open_port()
    metrics_port = get_open_port()
    try:
        with MooncakeLauncher(master_port, metrics_port) as launcher:
            _wait_for_mooncake_master(launcher.process, master_port)
            config_path = tmp_path / "mooncake.json"
            config_path.write_text(
                json.dumps(
                    {
                        "metadata_server": "P2PHANDSHAKE",
                        "protocol": "ascend",
                        "device_name": "",
                        "master_server_address": f"127.0.0.1:{master_port}",
                        "global_segment_size": "1GB",
                        "local_buffer_size": "64MB",
                        "preferred_segment": False,
                        "prefer_alloc_in_same_node": True,
                    }
                )
            )
            monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
            monkeypatch.delenv("MOONCAKE_MASTER", raising=False)
            monkeypatch.delenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", raising=False)

            server = context.Process(
                target=_run_multi_npu_server,
                args=(endpoint_child_connection, observation_child_connections, control_child_connection),
                name="kv-cache-multi-npu-server",
            )
            server.start()
            endpoint_child_connection.close()
            for child in observation_child_connections.values():
                child.close()
            control_child_connection.close()

            server_status, server_result = _receive(endpoint_connection, "KV cache server")
            if server_status != "ready":
                raise RuntimeError(f"KV cache server failed to start:\n{server_result}")

            caches = {}
            for rank, (first_fill, second_fill) in ((0, (13.0, 17.0)), (1, (21.0, 25.0))):
                torch.npu.set_device(rank)
                connector = AscendStoreMPConnector(
                    _make_tp2_worker_config(rank, server_result),
                    KVConnectorRole.WORKER,
                    kv_cache_config=None,
                )
                connectors.append(connector)
                _wait_until_connected(connector._kv_cache_client)
                _wait_until_registered(connector._kv_cache_client)

                first_layer, second_layer = _allocate_worker_caches(rank, first_fill, second_fill)
                torch.npu.synchronize()
                caches[rank] = (first_layer, second_layer, first_fill, second_fill)
                connector.register_kv_caches({"model.layers.0.attn": first_layer, "model.layers.1.attn": second_layer})
                _wait_for_active_export(connector)

            for rank in (0, 1):
                status, result = _receive(observation_connections[rank], f"Worker {rank} configuration")
                assert status == "configured"
                assert result == {"backend_device_index": rank, "backend_creation_count": 1}

            # Each Worker stores its own blocks, then retrieves them after its
            # local copies are zeroed.
            for rank in (0, 1):
                torch.npu.set_device(rank)
                _store_connector(connectors[rank], f"store-worker-{rank}", f"multi-npu-worker-{rank}")

            for rank in (0, 1):
                # synchronize() only covers the current device, so each
                # Worker's zeroing must be synced on its own NPU.
                torch.npu.set_device(rank)
                first_layer, second_layer, first_fill, second_fill = caches[rank]
                first_layer.zero_()
                second_layer.zero_()
                torch.npu.synchronize()

            for rank in (0, 1):
                torch.npu.set_device(rank)
                _retrieve_connector(connectors[rank], f"multi-npu-worker-{rank}")

            for rank in (0, 1):
                torch.npu.set_device(rank)
                torch.npu.synchronize()
                first_layer, second_layer, first_fill, second_fill = caches[rank]
                expected_first = torch.zeros_like(first_layer, device="cpu")
                expected_first[1].fill_(first_fill)
                expected_second = torch.zeros_like(second_layer, device="cpu")
                expected_second[1].fill_(second_fill)
                assert torch.equal(first_layer.cpu(), expected_first)
                assert torch.equal(second_layer.cpu(), expected_second)

            for rank in (0, 1):
                stored_status, stored_result = _receive(observation_connections[rank], f"Worker {rank} store")
                assert stored_status == "stored"
                # The exact key count depends on TP sharding; at least one
                # real Mooncake put must have happened per Worker.
                assert stored_result["stored_key_count"] >= 1

            for rank in (0, 1):
                connectors[rank].shutdown()
                closed_status, closed_result = _receive(observation_connections[rank], f"Worker {rank} close")
                assert closed_status == "closed"
                assert closed_result == {"mapping_released": True, "worker_caches_empty": True}
            connectors = []
    except BaseException as exc:
        failure = exc
    finally:
        endpoint_connection.close()
        for connector in connectors:
            connector.shutdown()
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            control_connection.send("stop")
        control_connection.close()
        server_exitcode, server_forced = _stop_process(server)

    if failure is not None:
        for connection in observation_connections.values():
            connection.close()
        raise failure

    locations_status, locations = _receive(observation_connections[0], "Mooncake registration locations")
    for connection in observation_connections.values():
        connection.close()
    assert locations_status == "locations"
    assert locations == ["npu:0", "npu:1"]

    if server_forced:
        pytest.fail("KV cache server did not stop after releasing both Worker mappings")
    assert server_exitcode == 0
