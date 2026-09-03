import multiprocessing as mp
import socket
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_events import BlockStored

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheClient, KVCacheMethod, KVCacheServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.npu_ipc import (
    NPUEventSpec,
    WorkerKVCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.protocol import (
    encode_lookup_response,
    encode_registration_request,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerLookupHandler,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import (
    MPClient,
    MPRemoteError,
    MPRequestTimeoutError,
    MPServerBusyError,
    MPServerUnavailableError,
)

KV_CACHE_CLIENT_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.client"
_DEFAULT_URL = "tcp://127.0.0.1:*"
_BLOCK_HASHES = [bytes.fromhex("01" * 32), bytes.fromhex("02" * 32)]


class _FakeWorker:
    def __init__(self, matched_tokens: int):
        self._matched_tokens = matched_tokens
        self._invalid_block_ids: set[int] = set()
        self.kv_cache_spec = None

    def configure_kv_caches(self, spec: WorkerKVCacheSpec) -> None:
        self.kv_cache_spec = spec

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
    ) -> int:
        return min(token_len, self._matched_tokens)

    def wait_for_save(self, metadata: AscendConnectorMetadata, event: NPUEventSpec) -> None:
        assert isinstance(metadata, AscendConnectorMetadata)
        assert isinstance(event, NPUEventSpec)

    @staticmethod
    def get_finished(finished_req_ids: set[str], metadata: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        return finished_req_ids & metadata.delayed_free_req_ids, finished_req_ids & metadata.loading_req_ids

    @staticmethod
    def build_connector_worker_meta() -> AscendStoreKVConnectorWorkerMetadata:
        return AscendStoreKVConnectorWorkerMetadata({7: 1})

    @staticmethod
    def get_kv_events() -> list[BlockStored]:
        return [
            BlockStored(
                block_hashes=[b"hash-0"],
                parent_block_hash=None,
                token_ids=[1],
                block_size=1,
                lora_id=None,
                medium="CPU",
                lora_name=None,
            )
        ]

    def start_load_kv(self, metadata: AscendConnectorMetadata) -> None:
        for request in metadata.requests:
            if request.load_spec is not None and request.load_spec.can_load:
                self._invalid_block_ids.update(request.block_ids)

    @staticmethod
    def wait_for_layer_load() -> None:
        return None

    @staticmethod
    def save_kv_layer_from_event(event: NPUEventSpec) -> None:
        assert isinstance(event, NPUEventSpec)

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids = self._invalid_block_ids
        self._invalid_block_ids = set()
        return block_ids

    def close(self) -> None:
        return None


class _FakeScheduler:
    def __init__(self, identity: SchedulerIdentity, lookup_handler: WorkerLookupHandler):
        self._identity = identity
        self._lookup_handler = lookup_handler

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        matched_tokens = self._lookup_handler(
            self._identity,
            len(request.prompt_token_ids),
            request.block_hashes,
            [0],
            False,
            num_computed_tokens,
        )
        return max(matched_tokens - num_computed_tokens, 0), False

    def close(self) -> None:
        return None


class _BlockingScheduler:
    def __init__(self, started_events, release_events):
        self._started_events = started_events
        self._release_events = release_events

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        request_index = int(request.request_id.rsplit("-", maxsplit=1)[1])
        self._started_events[request_index].set()
        if not self._release_events[request_index].wait(5):
            raise TimeoutError(f"Timed out waiting to release request {request.request_id}")
        return 0, False

    def close(self) -> None:
        return None


def _make_vllm_config(engine_id: str = "engine-0", rank: int = 0, data_parallel_rank: int = 0, block_size: int = 16):
    hf_config = SimpleNamespace(num_hidden_layers=2, model_type="llama")
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/model",
            max_model_len=1024,
            hf_text_config=hf_config,
            hf_config=hf_config,
            use_mla=False,
            get_num_layers=lambda _parallel_config: 2,
            get_total_num_kv_heads=lambda: 1,
        ),
        parallel_config=SimpleNamespace(
            rank=rank,
            world_size=1,
            data_parallel_rank=data_parallel_rank,
            data_parallel_index=data_parallel_rank,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id=engine_id,
            kv_role="kv_both",
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={},
        ),
        cache_config=SimpleNamespace(block_size=block_size, prefix_match_unit=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_events_config=None,
    )


def _make_request(request_id: str = "request-0"):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(range(32)),
        block_hashes=_BLOCK_HASHES,
        num_tokens=32,
    )


def _create_scheduler(registration: SchedulerRegistration, lookup_handler: WorkerLookupHandler) -> _FakeScheduler:
    return _FakeScheduler(registration.identity, lookup_handler)


def _create_worker(registration: WorkerRegistration, worker_hits: dict[tuple[int, int], int]) -> _FakeWorker:
    identity = registration.identity
    return _FakeWorker(worker_hits[(identity.data_parallel_rank, identity.rank)])


def _create_blocking_scheduler(
    _registration: SchedulerRegistration,
    _lookup_handler: WorkerLookupHandler,
    started_events,
    release_events,
) -> _BlockingScheduler:
    return _BlockingScheduler(started_events, release_events)


def _run_server(bind_url: str, conn, worker_hits: dict[tuple[int, int], int]) -> None:
    server = KVCacheServer(
        bind_url,
        scheduler_threads=4,
        worker_threads=4,
        scheduler_factory=_create_scheduler,
        worker_factory=partial(_create_worker, worker_hits=worker_hits),
    )
    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _run_affinity_server(bind_url: str, conn, started_events, release_events) -> None:
    server = KVCacheServer(
        bind_url,
        scheduler_threads=4,
        worker_threads=4,
        scheduler_factory=partial(
            _create_blocking_scheduler,
            started_events=started_events,
            release_events=release_events,
        ),
    )
    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _start_server(
    bind_url: str = _DEFAULT_URL,
    worker_hits: dict[tuple[int, int], int] | None = None,
) -> tuple[mp.process.BaseProcess, str]:
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(target=_run_server, args=(bind_url, child_conn, worker_hits or {(0, 0): 16}))
    process.start()
    child_conn.close()

    try:
        assert parent_conn.poll(5), "KV cache server did not start in time"
        endpoint = parent_conn.recv()
    except Exception:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        raise
    finally:
        parent_conn.close()

    return process, endpoint


def _start_affinity_server():
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe()
    started_events = [context.Event(), context.Event()]
    release_events = [context.Event(), context.Event()]
    process = context.Process(
        target=_run_affinity_server,
        args=(_DEFAULT_URL, child_conn, started_events, release_events),
    )
    process.start()
    child_conn.close()

    try:
        assert parent_conn.poll(5), "KV cache affinity server did not start in time"
        endpoint = parent_conn.recv()
    except Exception:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        raise
    finally:
        parent_conn.close()

    return process, endpoint, started_events, release_events


def _stop_server(process: mp.process.BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)


def _get_unused_tcp_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _wait_until_connected(client: KVCacheClient, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not client.is_connected:
        if time.monotonic() >= deadline:
            pytest.fail("KV cache client did not connect in time")
        time.sleep(0.01)


def _wait_until_registered(client: KVCacheClient, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not client.is_registered:
        if time.monotonic() >= deadline:
            pytest.fail("KV cache client did not register in time")
        time.sleep(0.01)


@pytest.fixture
def kv_cache_server_url() -> Iterator[str]:
    process, endpoint = _start_server()
    try:
        yield endpoint
    finally:
        _stop_server(process)


@pytest.mark.parametrize("thread_option", ["scheduler_threads", "worker_threads"])
def test_server_rejects_nonpositive_execution_pool_size(thread_option: str) -> None:
    with pytest.raises(ValueError, match=thread_option):
        KVCacheServer(_DEFAULT_URL, **{thread_option: 0})


def test_server_request_stop_completes_run_loop() -> None:
    server = KVCacheServer(_DEFAULT_URL, scheduler_factory=_create_scheduler)
    server_thread = threading.Thread(target=server.run)

    try:
        server_thread.start()
        server.request_stop()
        server_thread.join(timeout=5)

        assert not server_thread.is_alive()
    finally:
        server.close()


def test_server_close_drains_rpc_before_closing_services() -> None:
    calls = []

    def record(event: str, result: bool = True):
        def hook(*args: object, **kwargs: object) -> bool:
            calls.append(event)
            return result

        return hook

    server = KVCacheServer.__new__(KVCacheServer)
    server._close_lock = threading.Lock()
    server._abort_requested = threading.Event()
    server._closed = False
    server._rpc_server = MagicMock()
    server._service = MagicMock()
    server._rpc_server.request_stop.side_effect = record("request_stop")
    server._service.stop_lease_maintenance.side_effect = lambda wait=True: calls.append(
        "stop_maintenance_wait" if wait else "stop_maintenance_signal"
    )
    server._rpc_server.wait_for_drain.side_effect = record("drain_rpc")
    server._service.close.side_effect = lambda: calls.append("close_service")
    server._rpc_server.close.side_effect = record("close_rpc")

    assert server.close()

    assert calls == [
        "request_stop",
        "stop_maintenance_signal",
        "drain_rpc",
        "stop_maintenance_wait",
        "close_service",
        "close_rpc",
    ]


def test_server_close_does_not_touch_services_when_rpc_cannot_drain() -> None:
    server = KVCacheServer.__new__(KVCacheServer)
    server._close_lock = threading.Lock()
    server._abort_requested = threading.Event()
    server._closed = False
    server._rpc_server = MagicMock()
    server._service = MagicMock()
    server._rpc_server.request_stop.return_value = False

    assert not server.close()

    server._service.stop_lease_maintenance.assert_not_called()
    server._service.close.assert_not_called()
    server._rpc_server.wait_for_drain.assert_not_called()
    server._rpc_server.close.assert_not_called()


def test_server_abort_skips_graceful_service_close() -> None:
    server = KVCacheServer.__new__(KVCacheServer)
    server._close_lock = threading.Lock()
    server._abort_requested = threading.Event()
    server._closed = False
    server._rpc_server = MagicMock()
    server._service = MagicMock()

    server.abort()

    server._rpc_server.abort.assert_called_once_with()
    server._service.stop_lease_maintenance.assert_called_once_with(wait=False)
    server._service.close.assert_not_called()


def test_client_creation_does_not_wait_for_server() -> None:
    with patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class:
        with KVCacheClient("tcp://127.0.0.1:12345"):
            pass

        rpc_client_class.assert_called_once_with("tcp://127.0.0.1:12345")
        rpc_client_class.return_value.wait_until_connected.assert_not_called()
        rpc_client_class.return_value.close.assert_called_once_with()


def test_registration_checks_application_readiness() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = MPRequestTimeoutError("registration timeout")

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            registered = client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)

        assert not registered
        rpc_client.ping.assert_not_called()
        # A timed-out registration may have reached the Server. Closing sends
        # a best-effort unregister so an uncertain request cannot retain resources.
        assert [call.args[0] for call in rpc_client.request.call_args_list] == [
            KVCacheMethod.REGISTER_SCHEDULER,
            KVCacheMethod.UNREGISTER_SCHEDULER,
        ]


def test_lookup_retries_registration_after_server_busy() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [MPServerBusyError("Server busy"), [b"OK"], encode_lookup_response(16, False)]

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            assert not client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
            assert client.lookup(_make_request(), 0) == (16, False)
            assert client.is_registered

        rpc_client.ping.assert_not_called()


def test_lookup_returns_cache_miss_on_transport_failure() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [[b"OK"], MPServerUnavailableError("Server unavailable")]

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            assert client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
            assert client.lookup(_make_request(), 0) == (0, False)
            assert not client.is_registered


def test_lookup_returns_cache_miss_without_unregistering_when_server_is_busy() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client = rpc_client_class.return_value
        rpc_client.is_transport_connected = True
        rpc_client.request.side_effect = [[b"OK"], MPServerBusyError("Server busy")]

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            assert client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
            assert client.lookup(_make_request(), 0) == (0, False)
            assert client.is_registered


def test_lookup_validates_request_before_degrading() -> None:
    with (
        patch(f"{KV_CACHE_CLIENT_MODULE}.MPClient") as rpc_client_class,
        patch(f"{KV_CACHE_CLIENT_MODULE}.KVCacheClient._start_lease_loop"),
    ):
        rpc_client_class.return_value.is_transport_connected = False

        with KVCacheClient("tcp://127.0.0.1:12345") as client:
            assert not client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
            with pytest.raises(ValueError, match="num_computed_tokens must not be negative"):
                client.lookup(_make_request(), -1)


def test_multiple_workers_are_registered_and_rank_zero_serves_lookup() -> None:
    process, endpoint = _start_server(worker_hits={(0, 0): 16, (0, 1): 32})
    clients: list[KVCacheClient] = []

    try:
        for rank in (1, 0):
            client = KVCacheClient(endpoint)
            clients.append(client)
            _wait_until_connected(client)
            assert client.register_worker(_make_vllm_config(rank=rank), kv_cache_config=None)
            spec = WorkerKVCacheSpec(caches={f"layer.{rank}": ()}, storages=())
            assert client.register_kv_caches(spec)

        scheduler_client = KVCacheClient(endpoint)
        clients.append(scheduler_client)
        _wait_until_connected(scheduler_client)
        assert scheduler_client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
        assert scheduler_client.lookup(_make_request(), 0) == (16, False)
    finally:
        for client in clients:
            client.close()
        _stop_server(process)


def test_worker_wait_for_save_round_trip() -> None:
    process, endpoint = _start_server()
    client = KVCacheClient(endpoint)

    try:
        _wait_until_connected(client)
        assert client.register_worker(_make_vllm_config(), kv_cache_config=None)
        metadata = AscendConnectorMetadata(set(), set())
        event = NPUEventSpec("host-0", b"event-handle")

        assert client.wait_for_save(metadata, event)
    finally:
        client.close()
        _stop_server(process)


def test_worker_get_finished_round_trip() -> None:
    process, endpoint = _start_server()
    client = KVCacheClient(endpoint)

    try:
        _wait_until_connected(client)
        assert client.register_worker(_make_vllm_config(), kv_cache_config=None)
        metadata = AscendConnectorMetadata(
            preempted_req_ids=set(),
            loading_req_ids={"loading-0", "loading-stale"},
            delayed_free_req_ids={"saving-0", "saving-stale"},
        )

        assert client.get_finished({"saving-0", "loading-0"}, metadata) == ({"saving-0"}, {"loading-0"})
    finally:
        client.close()
        _stop_server(process)


def test_worker_build_connector_meta_round_trip() -> None:
    process, endpoint = _start_server()
    client = KVCacheClient(endpoint)

    try:
        _wait_until_connected(client)
        assert client.register_worker(_make_vllm_config(), kv_cache_config=None)

        metadata = client.build_connector_worker_meta()

        assert isinstance(metadata, AscendStoreKVConnectorWorkerMetadata)
        assert metadata.completed_events == {7: 1}
    finally:
        client.close()
        _stop_server(process)


def test_worker_get_kv_events_round_trip() -> None:
    process, endpoint = _start_server()
    client = KVCacheClient(endpoint)

    try:
        _wait_until_connected(client)
        assert client.register_worker(_make_vllm_config(), kv_cache_config=None)

        events = client.get_kv_events()

        assert len(events) == 1
        assert isinstance(events[0], BlockStored)
        assert events[0].block_hashes == [b"hash-0"]
    finally:
        client.close()
        _stop_server(process)


def test_worker_start_load_and_load_errors_round_trip() -> None:
    process, endpoint = _start_server()
    client = KVCacheClient(endpoint)

    try:
        _wait_until_connected(client)
        assert client.register_worker(_make_vllm_config(), kv_cache_config=None)
        metadata = AscendConnectorMetadata(set(), set())
        metadata.add_request(
            ReqMeta(
                "request-0",
                token_len_chunk=16,
                block_ids=[7],
                block_hashes=[b"hash-0"],
                load_spec=LoadSpec(0, 16, True),
            )
        )

        assert client.start_load_kv(metadata)
        assert client.get_block_ids_with_load_errors() == {7}
        assert client.get_block_ids_with_load_errors() == set()
    finally:
        client.close()
        _stop_server(process)


def test_worker_layerwise_calls_round_trip() -> None:
    process, endpoint = _start_server()
    client = KVCacheClient(endpoint)

    try:
        _wait_until_connected(client)
        assert client.register_worker(_make_vllm_config(), kv_cache_config=None)
        assert client.start_load_kv(AscendConnectorMetadata(set(), set()))
        client.wait_for_layer_load()
        client.save_kv_layer(NPUEventSpec("host-0", b"event-handle"))
    finally:
        client.close()
        _stop_server(process)


def test_dp_schedulers_use_their_own_rank_zero_worker() -> None:
    process, endpoint = _start_server(worker_hits={(0, 0): 16, (1, 0): 32})
    clients: list[KVCacheClient] = []

    try:
        for data_parallel_rank in (0, 1):
            worker_client = KVCacheClient(endpoint)
            clients.append(worker_client)
            _wait_until_connected(worker_client)
            assert worker_client.register_worker(
                _make_vllm_config(data_parallel_rank=data_parallel_rank),
                kv_cache_config=None,
            )

        for data_parallel_rank, expected_tokens in ((0, 16), (1, 32)):
            scheduler_client = KVCacheClient(endpoint)
            clients.append(scheduler_client)
            _wait_until_connected(scheduler_client)
            assert scheduler_client.register_scheduler(
                _make_vllm_config(data_parallel_rank=data_parallel_rank),
                kv_cache_config=None,
                page_size_bytes=0,
            )
            assert scheduler_client.lookup(_make_request(), 0) == (expected_tokens, False)
    finally:
        for client in clients:
            client.close()
        _stop_server(process)


def test_registration_identity_uses_engine_id_dp_rank_and_worker_rank() -> None:
    config = _make_vllm_config(engine_id="engine-1", rank=3, data_parallel_rank=2)

    scheduler_registration = SchedulerRegistration.create(config, kv_cache_config=None, page_size_bytes=0)
    worker_registration = WorkerRegistration.create(config, kv_cache_config=None)

    assert scheduler_registration.identity == SchedulerIdentity("engine-1", data_parallel_rank=2)
    assert worker_registration.identity == WorkerIdentity("engine-1", rank=3, data_parallel_rank=2)


def test_registration_is_idempotent_and_rejects_conflicts(kv_cache_server_url: str) -> None:
    registration = SchedulerRegistration.create(
        _make_vllm_config(block_size=16),
        kv_cache_config=None,
        page_size_bytes=0,
    )
    payloads = encode_registration_request(registration)
    conflicting_registration = SchedulerRegistration.create(
        _make_vllm_config(block_size=32),
        kv_cache_config=None,
        page_size_bytes=0,
    )
    conflicting_payloads = encode_registration_request(conflicting_registration)

    with MPClient(kv_cache_server_url) as client:
        client.wait_until_connected()
        assert client.request(KVCacheMethod.REGISTER_SCHEDULER, payloads) == [b"OK"]
        assert client.request(KVCacheMethod.REGISTER_SCHEDULER, payloads) == [b"OK"]

        with pytest.raises(MPRemoteError, match="different configuration"):
            client.request(KVCacheMethod.REGISTER_SCHEDULER, conflicting_payloads)


def test_lookup_serializes_requests_for_the_same_scheduler() -> None:
    process, endpoint, started_events, release_events = _start_affinity_server()
    client = KVCacheClient(endpoint)
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        _wait_until_connected(client)
        assert client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)

        first_future = executor.submit(client.lookup, _make_request("request-0"), 0)
        assert started_events[0].wait(5), "First lookup did not start in time"

        second_future = executor.submit(client.lookup, _make_request("request-1"), 0)
        assert not started_events[1].wait(0.2), "Lookups for the same Scheduler ran concurrently"

        release_events[0].set()
        assert started_events[1].wait(5), "Second lookup did not start after the first completed"
        release_events[1].set()

        assert first_future.result(timeout=5) == (0, False)
        assert second_future.result(timeout=5) == (0, False)
    finally:
        for event in release_events:
            event.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.close()
        _stop_server(process)


def test_lease_renewal_is_not_blocked_by_scheduler_lookup() -> None:
    process, endpoint, started_events, release_events = _start_affinity_server()
    client = KVCacheClient(endpoint)
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        _wait_until_connected(client)
        with patch.object(client, "_start_lease_loop"):
            assert client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)

        lookup_future = executor.submit(client.lookup, _make_request("request-0"), 0)
        assert started_events[0].wait(5), "Lookup did not start in time"

        client._maintain_lease()

        assert client.is_registered
        release_events[0].set()
        assert lookup_future.result(timeout=5) == (0, False)
    finally:
        release_events[0].set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.close()
        _stop_server(process)


def test_lookup_runs_different_dp_schedulers_in_parallel() -> None:
    process, endpoint, started_events, release_events = _start_affinity_server()
    clients = [KVCacheClient(endpoint), KVCacheClient(endpoint)]
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        for data_parallel_rank, client in enumerate(clients):
            _wait_until_connected(client)
            assert client.register_scheduler(
                _make_vllm_config(data_parallel_rank=data_parallel_rank),
                kv_cache_config=None,
                page_size_bytes=0,
            )

        futures = [
            executor.submit(client.lookup, _make_request(f"request-{data_parallel_rank}"), 0)
            for data_parallel_rank, client in enumerate(clients)
        ]
        assert started_events[0].wait(5), "DP Scheduler 0 lookup did not start in time"
        assert started_events[1].wait(5), "DP Scheduler 1 lookup did not run in parallel"

        for event in release_events:
            event.set()
        assert [future.result(timeout=5) for future in futures] == [(0, False), (0, False)]
    finally:
        for event in release_events:
            event.set()
        executor.shutdown(wait=True, cancel_futures=True)
        for client in clients:
            client.close()
        _stop_server(process)


def test_lookup_recovers_when_server_starts_later() -> None:
    server_url = _get_unused_tcp_url()
    process = None

    with KVCacheClient(server_url) as worker_client, KVCacheClient(server_url) as scheduler_client:
        assert not worker_client.register_worker(_make_vllm_config(), kv_cache_config=None)
        assert not scheduler_client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
        assert scheduler_client.lookup(_make_request(), 0) == (0, False)

        try:
            process, endpoint = _start_server(server_url, worker_hits={(0, 0): 16})
            assert endpoint == server_url
            _wait_until_connected(worker_client)
            _wait_until_connected(scheduler_client)
            _wait_until_registered(worker_client)
            _wait_until_registered(scheduler_client)
            assert scheduler_client.lookup(_make_request(), 0) == (16, False)
        finally:
            if process is not None:
                _stop_server(process)


def test_server_rejects_malformed_lookup(kv_cache_server_url: str) -> None:
    with MPClient(kv_cache_server_url) as client:
        client.wait_until_connected()
        with pytest.raises(MPRemoteError, match="Scheduler identity expects at least 2 payloads"):
            client.request(KVCacheMethod.LOOKUP)
        assert client.ping() == "OK"
