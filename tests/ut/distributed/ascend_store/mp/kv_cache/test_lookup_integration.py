import multiprocessing as mp
import time
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# isort: off
# Import real pyzmq before _mock_deps to prevent it from being mocked.
import zmq.asyncio  # noqa: F401

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheClient, KVCacheServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.manager import KVCacheServiceManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.scheduler import MPKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.pool.worker import MPKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.protocol import encode_registration
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import (
    SchedulerRegistration,
    WorkerLookupHandler,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

# isort: on

POOL_SCHEDULER_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"
_DEFAULT_URL = "tcp://127.0.0.1:*"
_BLOCK_HASHES = [bytes.fromhex("01" * 32), bytes.fromhex("02" * 32)]


class _FakeStore:
    def __init__(self, exists_result: list[int]):
        self._exists_result = exists_result

    def exists(self, keys: list[str]) -> list[int]:
        return self._exists_result[: len(keys)]


def _make_vllm_config(
    tp_size: int = 1, rank: int = 0, engine_id: str = "engine-0", data_parallel_rank: int = 0
) -> MagicMock:
    config = MagicMock()

    hf_config = MagicMock(spec=[])
    config.model_config.model = "org/llama-7b"
    config.model_config.hf_text_config = hf_config
    config.model_config.hf_config = hf_config
    config.model_config.use_mla = False
    config.model_config.max_model_len = 1024
    config.model_config.get_num_layers.return_value = 2
    config.model_config.get_total_num_kv_heads.return_value = tp_size

    config.parallel_config.rank = rank
    config.parallel_config.world_size = tp_size
    config.parallel_config.data_parallel_rank = data_parallel_rank
    config.parallel_config.data_parallel_index = data_parallel_rank
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1

    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.engine_id = engine_id
    config.kv_transfer_config.kv_connector = "AscendStoreConnector"
    config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
    config.kv_transfer_config.get_from_extra_config.return_value = True

    config.cache_config.block_size = 16
    config.cache_config.prefix_match_unit = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    config.kv_events_config = None
    return config


def _create_scheduler(registration: SchedulerRegistration, lookup_handler: WorkerLookupHandler) -> MPKVPoolScheduler:
    with patch(f"{POOL_SCHEDULER_MODULE}.importlib") as importlib_mock:
        importlib_mock.import_module.return_value = MagicMock()
        return MPKVPoolScheduler(registration, lookup_handler)


def _create_worker(
    registration: WorkerRegistration,
    worker_results: dict[tuple[str, int, int], list[int]],
) -> MPKVPoolWorker:
    identity = registration.identity
    worker_key = (identity.engine_id, identity.data_parallel_rank, identity.rank)
    return MPKVPoolWorker(
        registration.config,
        store=_FakeStore(worker_results[worker_key]),
        kv_cache_config=registration.config.build_kv_cache_config(),
        rank=identity.rank,
    )


def _run_lookup_server(bind_url: str, conn, worker_results: dict[tuple[str, int, int], list[int]]) -> None:
    server = KVCacheServer(
        bind_url,
        scheduler_threads=4,
        worker_threads=4,
        scheduler_factory=_create_scheduler,
        worker_factory=partial(_create_worker, worker_results=worker_results),
    )
    try:
        conn.send(server.endpoint)
        conn.close()
        server.run()
    finally:
        server.close()


def _start_lookup_server(worker_results: dict[tuple[str, int, int], list[int]]) -> tuple[mp.process.BaseProcess, str]:
    context = mp.get_context("fork")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(target=_run_lookup_server, args=(_DEFAULT_URL, child_conn, worker_results))
    process.start()
    child_conn.close()

    try:
        assert parent_conn.poll(5), "Lookup server did not start in time"
        endpoint = parent_conn.recv()
    except Exception:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        raise
    finally:
        parent_conn.close()

    return process, endpoint


def _stop_lookup_server(process: mp.process.BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)


def _wait_until_connected(client: KVCacheClient, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not client.is_connected:
        if time.monotonic() >= deadline:
            raise AssertionError("KV cache client did not connect in time")
        time.sleep(0.01)


def _create_client(endpoint: str) -> KVCacheClient:
    client = KVCacheClient(endpoint)
    _wait_until_connected(client)
    return client


def _make_request(request_id: str = "request-0") -> MagicMock:
    request = MagicMock()
    request.request_id = request_id
    request.prompt_token_ids = list(range(32))
    request.block_hashes = _BLOCK_HASHES
    request.num_tokens = 32
    return request


def test_scheduler_lookup_round_trip_uses_original_logic() -> None:
    worker_results = {("engine-0", 0, 0): [1, 1, 1, 0], ("engine-0", 0, 1): [0, 0, 0, 0]}
    process, endpoint = _start_lookup_server(worker_results)
    clients: list[KVCacheClient] = []

    try:
        for rank in range(2):
            worker_client = _create_client(endpoint)
            clients.append(worker_client)
            assert worker_client.register_worker(_make_vllm_config(tp_size=2, rank=rank), kv_cache_config=None)

        scheduler_client = _create_client(endpoint)
        clients.append(scheduler_client)
        assert scheduler_client.register_scheduler(
            _make_vllm_config(tp_size=2),
            kv_cache_config=None,
            page_size_bytes=0,
        )

        assert scheduler_client.lookup(_make_request(), num_computed_tokens=0) == (16, False)
    finally:
        for client in clients:
            client.close()
        _stop_lookup_server(process)


def test_lookup_isolated_by_engine_and_data_parallel_rank() -> None:
    worker_results = {("engine-0", 0, 0): [1, 0], ("engine-0", 1, 0): [0, 0], ("engine-1", 0, 0): [1, 1]}
    expected_results = {("engine-0", 0): (16, False), ("engine-0", 1): (0, False), ("engine-1", 0): (31, False)}
    process, endpoint = _start_lookup_server(worker_results)
    clients: list[KVCacheClient] = []

    try:
        for engine_id, data_parallel_rank, rank in worker_results:
            worker_client = _create_client(endpoint)
            clients.append(worker_client)
            config = _make_vllm_config(rank=rank, engine_id=engine_id, data_parallel_rank=data_parallel_rank)
            assert worker_client.register_worker(config, kv_cache_config=None)

        for (engine_id, data_parallel_rank), expected in expected_results.items():
            scheduler_client = _create_client(endpoint)
            clients.append(scheduler_client)
            config = _make_vllm_config(engine_id=engine_id, data_parallel_rank=data_parallel_rank)
            assert scheduler_client.register_scheduler(config, kv_cache_config=None, page_size_bytes=0)
            assert scheduler_client.lookup(_make_request(), num_computed_tokens=0) == expected
    finally:
        for client in clients:
            client.close()
        _stop_lookup_server(process)


def test_lookup_returns_miss_after_coordinator_unregisters() -> None:
    worker_results = {("engine-0", 0, 0): [1, 0]}
    process, endpoint = _start_lookup_server(worker_results)
    worker_client = _create_client(endpoint)
    scheduler_client = _create_client(endpoint)

    try:
        assert worker_client.register_worker(_make_vllm_config(), kv_cache_config=None)
        assert scheduler_client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
        assert scheduler_client.lookup(_make_request(), num_computed_tokens=0) == (16, False)

        worker_client.close()

        assert scheduler_client.lookup(_make_request("request-1"), num_computed_tokens=0) == (0, False)
    finally:
        worker_client.close()
        scheduler_client.close()
        _stop_lookup_server(process)


def test_unregister_closes_the_real_scheduler_service() -> None:
    # Fake scheduler factories all implement close(), so only the real
    # MPKVPoolScheduler can expose a service missing the manager's close
    # contract. Client-side unregister hides that failure behind best-effort
    # cleanup, so the manager is called directly here.
    registration = SchedulerRegistration.create(
        _make_vllm_config(),
        kv_cache_config=None,
        page_size_bytes=0,
        session_id="scheduler-session",
    )
    service_manager = KVCacheServiceManager(scheduler_factory=_create_scheduler)

    scheduler = service_manager.register_scheduler(registration, encode_registration(registration))
    assert isinstance(scheduler, MPKVPoolScheduler)

    assert service_manager.unregister_scheduler(registration.identity, registration.session_id) is True
    assert service_manager.scheduler_count == 0


def test_update_state_after_alloc_round_trip_after_lookup() -> None:
    worker_results = {("engine-0", 0, 0): [1, 1]}
    process, endpoint = _start_lookup_server(worker_results)
    worker_client = _create_client(endpoint)
    scheduler_client = _create_client(endpoint)

    try:
        assert worker_client.register_worker(_make_vllm_config(), kv_cache_config=None)
        assert scheduler_client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
        assert scheduler_client.lookup(_make_request(), num_computed_tokens=0) == (31, False)

        blocks = SimpleNamespace(get_block_ids=lambda: ([7],))
        scheduler_client.update_state_after_alloc(_make_request(), blocks, num_external_tokens=31)

        # The server keeps serving the same scheduler session afterwards.
        assert scheduler_client.lookup(_make_request("request-1"), num_computed_tokens=0) == (31, False)
    finally:
        worker_client.close()
        scheduler_client.close()
        _stop_lookup_server(process)


def test_build_connector_meta_round_trip_after_lookup_and_alloc() -> None:
    worker_results = {("engine-0", 0, 0): [1, 1]}
    process, endpoint = _start_lookup_server(worker_results)
    worker_client = _create_client(endpoint)
    scheduler_client = _create_client(endpoint)

    try:
        assert worker_client.register_worker(_make_vllm_config(), kv_cache_config=None)
        assert scheduler_client.register_scheduler(_make_vllm_config(), kv_cache_config=None, page_size_bytes=0)
        request = _make_request()
        assert scheduler_client.lookup(request, num_computed_tokens=0) == (31, False)
        blocks = SimpleNamespace(get_block_ids=lambda: ([7],))
        scheduler_client.update_state_after_alloc(request, blocks, num_external_tokens=31)

        scheduler_output = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            num_scheduled_tokens={"request-0": 32},
            scheduled_new_reqs=[SimpleNamespace(req_id="request-0", num_computed_tokens=0, block_ids=([7], [8]))],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                new_block_ids=[],
                num_computed_tokens=[],
            ),
        )
        metadata, touch_block_ids = scheduler_client.build_connector_meta(scheduler_output, {})

        assert touch_block_ids == []
        assert len(metadata.requests) == 1
        assert metadata.requests[0].req_id == "request-0"
        assert metadata.requests[0].load_spec is not None

        # The build step saved tokens, so closing the request delays the block
        # free; an empty worker output round-trips without disturbing the session.
        assert scheduler_client.request_finished("request-0", [7, 8]) == (True, None)
        assert scheduler_client.update_connector_output({}) == []
    finally:
        worker_client.close()
        scheduler_client.close()
        _stop_lookup_server(process)


def test_mp_classes_reuse_original_business_methods() -> None:
    assert MPKVPoolScheduler.get_num_new_matched_tokens is KVPoolScheduler.get_num_new_matched_tokens
