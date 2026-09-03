import threading
from functools import partial
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.manager import KVCacheServiceManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.npu_ipc import WorkerKVCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.protocol import encode_registration
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import (
    SchedulerRegistration,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.rpc import AffinityExecutor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import StaleSessionError

_BLOCK_HASHES = [bytes.fromhex("01" * 32), bytes.fromhex("02" * 32)]


class _FakeScheduler:
    def __init__(self, identity, lookup_handler):
        self._identity = identity
        self._lookup_handler = lookup_handler
        self.store_scheduler = object()
        self.touch_block_ids = []
        self.free_block_ids = []
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        matched_tokens = self._lookup_handler(
            self._identity,
            len(request.prompt_token_ids),
            request.block_hashes,
            [0],
            False,
            num_computed_tokens,
        )
        return matched_tokens, False

    @staticmethod
    def build_connector_meta(output):
        return output

    def take_block_pool_commands(self) -> list[int]:
        block_ids = self.touch_block_ids
        self.touch_block_ids = []
        return block_ids

    @staticmethod
    def update_connector_output(output) -> None:
        return None

    def take_free_block_commands(self) -> list[int]:
        block_ids = self.free_block_ids
        self.free_block_ids = []
        return block_ids


class _FakeWorker:
    def __init__(self, matched_tokens: int = 0, closed: threading.Event | None = None):
        self._matched_tokens = matched_tokens
        self._closed = closed
        self.lookup_hashes = None
        self.lookup_threads = []
        self.close_threads = []
        self.configure_threads = []
        self.kv_cache_spec = None
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.close_threads.append(threading.get_ident())
        if self._closed is not None:
            self._closed.set()

    def configure_kv_caches(self, spec: WorkerKVCacheSpec) -> None:
        self.kv_cache_spec = spec
        self.configure_threads.append(threading.get_ident())

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
    ) -> int:
        self.lookup_hashes = block_hashes
        self.lookup_threads.append(threading.get_ident())
        return min(token_len, self._matched_tokens)


def _make_vllm_config(rank: int = 0, data_parallel_rank: int = 0):
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
            engine_id="engine-0",
            kv_role="kv_both",
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={},
        ),
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_events_config=None,
    )


def _scheduler_registration(session_id: str, data_parallel_rank: int = 0) -> SchedulerRegistration:
    return SchedulerRegistration.create(
        _make_vllm_config(data_parallel_rank=data_parallel_rank),
        None,
        0,
        session_id=session_id,
    )


def _worker_registration(session_id: str, rank: int = 0, data_parallel_rank: int = 0) -> WorkerRegistration:
    return WorkerRegistration.create(_make_vllm_config(rank, data_parallel_rank), None, session_id=session_id)


def _create_scheduler(registration, lookup_handler) -> _FakeScheduler:
    return _FakeScheduler(registration.identity, lookup_handler)


def test_lookup_routes_to_rank_zero_worker_in_the_same_dp_group() -> None:
    workers = {(0, 0): _FakeWorker(16), (0, 1): _FakeWorker(32), (1, 0): _FakeWorker(48)}

    def worker_factory(registration):
        identity = registration.identity
        return workers[(identity.data_parallel_rank, identity.rank)]

    service_manager = KVCacheServiceManager(_create_scheduler, worker_factory)
    for data_parallel_rank, rank in workers:
        registration = _worker_registration(f"worker-{data_parallel_rank}-{rank}", rank, data_parallel_rank)
        service_manager.register_worker(registration, encode_registration(registration))

    scheduler_registration = _scheduler_registration("scheduler-session")
    service_manager.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(64)),
        block_hashes=_BLOCK_HASHES,
        num_tokens=64,
    )

    result = service_manager.lookup(scheduler_registration.identity, scheduler_registration.session_id, request, 0)
    assert result == (16, False)
    assert workers[(0, 0)].lookup_hashes == [block_hash.hex() for block_hash in _BLOCK_HASHES]
    assert workers[(0, 1)].lookup_hashes is None
    assert workers[(1, 0)].lookup_hashes is None


def test_lookup_and_close_run_on_the_worker_lane() -> None:
    worker = _FakeWorker(16)
    worker_executor = AffinityExecutor(1, 4, "test-worker-lane")
    service_manager = KVCacheServiceManager(
        _create_scheduler,
        lambda registration: worker,
        worker_executor=worker_executor,
    )
    scheduler_registration = _scheduler_registration("scheduler-session")
    worker_registration = _worker_registration("worker-session")
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(16)),
        block_hashes=_BLOCK_HASHES,
        num_tokens=16,
    )

    try:
        service_manager.register_worker(worker_registration, encode_registration(worker_registration))
        service_manager.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
        result = service_manager.lookup(scheduler_registration.identity, scheduler_registration.session_id, request, 0)

        assert result == (16, False)
        assert len(worker.lookup_threads) == 1
        assert worker.lookup_threads != [threading.get_ident()]
        service_manager.close()
        assert worker.close_threads == worker.lookup_threads
    finally:
        service_manager.close()
        worker_executor.shutdown(wait=True, cancel_futures=True)


def test_worker_cache_spec_is_configured_on_its_owner_lane() -> None:
    workers = {0: _FakeWorker(), 1: _FakeWorker()}
    worker_executor = AffinityExecutor(2, 4, "test-worker-cache-lane")

    def worker_factory(registration):
        return workers[registration.identity.rank]

    service_manager = KVCacheServiceManager(worker_factory=worker_factory, worker_executor=worker_executor)
    registrations = [_worker_registration(f"worker-{rank}", rank=rank) for rank in workers]
    specs = {rank: WorkerKVCacheSpec(caches={f"layer.{rank}": ()}, storages=()) for rank in workers}

    try:
        for registration in registrations:
            service_manager.register_worker(registration, encode_registration(registration))
            worker_executor.submit(
                partial(
                    service_manager.register_worker_kv_caches,
                    registration.identity,
                    registration.session_id,
                    specs[registration.identity.rank],
                ),
                registration.identity,
            ).result()

        assert workers[0].kv_cache_spec == specs[0]
        assert workers[1].kv_cache_spec == specs[1]
        assert len(workers[0].configure_threads) == 1
        assert len(workers[1].configure_threads) == 1
        assert workers[0].configure_threads != [threading.get_ident()]
        assert workers[1].configure_threads != [threading.get_ident()]
    finally:
        service_manager.close()
        worker_executor.shutdown(wait=True, cancel_futures=True)


def test_new_worker_session_accepts_a_new_cache_mapping() -> None:
    workers = []

    def worker_factory(_registration):
        worker = _FakeWorker()
        workers.append(worker)
        return worker

    service_manager = KVCacheServiceManager(worker_factory=worker_factory)
    old_registration = _worker_registration("old-session")
    new_registration = _worker_registration("new-session")
    old_spec = WorkerKVCacheSpec(caches={"old.layer": ()}, storages=())
    new_spec = WorkerKVCacheSpec(caches={"new.layer": ()}, storages=())

    try:
        old_worker = service_manager.register_worker(old_registration, encode_registration(old_registration))
        service_manager.register_worker_kv_caches(old_registration.identity, old_registration.session_id, old_spec)

        new_worker = service_manager.register_worker(new_registration, encode_registration(new_registration))
        service_manager.register_worker_kv_caches(new_registration.identity, new_registration.session_id, new_spec)

        assert old_worker.close_count == 1
        assert new_worker is workers[1]
        assert new_worker.kv_cache_spec == new_spec
        assert service_manager.worker_count == 1
        with pytest.raises(StaleSessionError, match="retired"):
            service_manager.register_worker_kv_caches(
                old_registration.identity,
                old_registration.session_id,
                old_spec,
            )
    finally:
        service_manager.close()


def test_scheduler_block_pool_commands_cross_the_manager_boundary() -> None:
    service_manager = KVCacheServiceManager(_create_scheduler)
    registration = _scheduler_registration("scheduler-session")

    try:
        scheduler = service_manager.register_scheduler(registration, encode_registration(registration))
        scheduler.touch_block_ids = [5, 8]
        scheduler.free_block_ids = [13, 21]
        scheduler_output = SimpleNamespace()
        connector_output = SimpleNamespace()

        metadata, touch_block_ids = service_manager.build_connector_meta(
            registration.identity,
            registration.session_id,
            scheduler_output,
        )
        free_block_ids = service_manager.update_connector_output(
            registration.identity,
            registration.session_id,
            connector_output,
        )

        assert metadata is scheduler_output
        assert touch_block_ids == [5, 8]
        assert free_block_ids == [13, 21]
        assert scheduler.touch_block_ids == []
        assert scheduler.free_block_ids == []
    finally:
        service_manager.close()


def test_lookup_does_not_fall_back_to_a_non_coordinator_worker() -> None:
    worker = _FakeWorker(32)
    service_manager = KVCacheServiceManager(_create_scheduler, lambda registration: worker)
    worker_registration = _worker_registration("worker-session", rank=1)
    scheduler_registration = _scheduler_registration("scheduler-session")

    service_manager.register_worker(worker_registration, encode_registration(worker_registration))
    service_manager.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(32)),
        block_hashes=_BLOCK_HASHES,
        num_tokens=32,
    )

    result = service_manager.lookup(scheduler_registration.identity, scheduler_registration.session_id, request, 0)
    assert result == (0, False)
    assert worker.lookup_hashes is None


def test_manager_expires_idle_worker_while_lookup_renews_scheduler() -> None:
    now = [0.0]
    worker_closed = threading.Event()
    worker = _FakeWorker(closed=worker_closed)
    service_manager = KVCacheServiceManager(
        _create_scheduler,
        lambda registration: worker,
        lease_timeout_s=10.0,
        lease_check_interval_s=0.01,
        clock=lambda: now[0],
    )
    scheduler_registration = _scheduler_registration("scheduler-session")
    worker_registration = _worker_registration("worker-session")
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=list(range(16)),
        block_hashes=_BLOCK_HASHES,
        num_tokens=16,
    )

    service_manager.register_scheduler(scheduler_registration, encode_registration(scheduler_registration))
    service_manager.register_worker(worker_registration, encode_registration(worker_registration))
    now[0] = 9.0
    service_manager.lookup(scheduler_registration.identity, scheduler_registration.session_id, request, 0)
    now[0] = 11.0

    service_manager.start_lease_maintenance()
    try:
        assert worker_closed.wait(1), "Idle Worker lease did not expire"
        assert service_manager.scheduler_count == 1
        assert service_manager.worker_count == 0
        assert service_manager.lookup(
            scheduler_registration.identity,
            scheduler_registration.session_id,
            request,
            0,
        ) == (0, False)
    finally:
        service_manager.close()
