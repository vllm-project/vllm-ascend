import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.manager import KVCacheServiceManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.protocol import (
    decode_registration,
    encode_registration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.kv_cache.registration import (
    SchedulerRegistration,
    WorkerRegistration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import (
    RegistrationConflictError,
    StaleSessionError,
)


class _FakeScheduler:
    def __init__(self):
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False


class _FakeWorker:
    def __init__(self):
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[str],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
    ) -> int:
        return 0


class _UnserializableRuntimeState:
    def __reduce__(self):
        raise AssertionError("runtime state must not enter a registration payload")


@dataclass(frozen=True)
class _UnsupportedKVCacheSpec(KVCacheSpec):
    @property
    def page_size_bytes(self) -> int:
        return self.block_size


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


def _scheduler_registration(session_id: str, *, data_parallel_rank: int = 0, block_size: int = 16):
    return SchedulerRegistration.create(
        _make_vllm_config(data_parallel_rank=data_parallel_rank, block_size=block_size),
        None,
        0,
        session_id=session_id,
    )


def test_registration_projects_only_kv_pool_configuration() -> None:
    config = _make_vllm_config()
    config.model_config.runtime_state = _UnserializableRuntimeState()
    config.model_config.runtime_tensor = torch.ones(1024)
    config.compilation_config = _UnserializableRuntimeState()
    config.kv_transfer_config.kv_role = "kv_both"
    config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
    config.kv_events_config = SimpleNamespace(enable_kv_cache_events=True)

    registration = SchedulerRegistration.create(config, None, 4096)
    payload = encode_registration(registration)
    runtime_config = registration.config
    kv_cache_config = registration.config.build_kv_cache_config()

    assert len(payload) < 64 * 1024
    assert runtime_config is registration.config
    assert runtime_config.model_config.model == "org/model"
    assert runtime_config.model_config.max_model_len == 1024
    assert runtime_config.kv_transfer_config.kv_connector_extra_config == {"backend": "mooncake"}
    assert runtime_config.kv_events_config is not None
    assert runtime_config.kv_events_config.enable_kv_cache_events
    assert kv_cache_config is None


@pytest.mark.parametrize(
    "kv_cache_spec",
    [
        FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16),
        UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={
                "model.layers.0.attn": FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                )
            },
        ),
        AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=576, dtype=torch.bfloat16),
    ],
    ids=["vllm", "vllm-uniform", "ascend"],
)
def test_registration_round_trips_supported_kv_cache_specs(kv_cache_spec) -> None:
    kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[_UnserializableRuntimeState()],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.attn"],
                kv_cache_spec=kv_cache_spec,
            )
        ],
    )

    registration = SchedulerRegistration.create(_make_vllm_config(), kv_cache_config, 4096)
    restored = decode_registration((encode_registration(registration),), SchedulerRegistration)
    restored_kv_cache_config = restored.config.build_kv_cache_config()

    assert restored_kv_cache_config is not None
    assert restored_kv_cache_config.num_blocks == 64
    assert restored_kv_cache_config.kv_cache_tensors == []
    assert restored_kv_cache_config.kv_cache_groups[0].layer_names == ["model.layers.0.attn"]
    restored_spec = restored_kv_cache_config.kv_cache_groups[0].kv_cache_spec
    assert type(restored_spec) is type(kv_cache_spec)
    assert restored_spec == kv_cache_spec


def test_registration_rejects_unsupported_kv_cache_spec() -> None:
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.attn"],
                kv_cache_spec=_UnsupportedKVCacheSpec(block_size=16),
            )
        ],
    )

    with pytest.raises(TypeError, match="Unsupported KV cache spec type"):
        SchedulerRegistration.create(_make_vllm_config(), kv_cache_config, 4096)


def test_registration_rejects_runtime_objects_in_extra_config() -> None:
    config = _make_vllm_config()
    config.kv_transfer_config.kv_connector_extra_config = {"tensor": torch.ones(1)}

    with pytest.raises(TypeError, match="Unsupported registration configuration value Tensor"):
        SchedulerRegistration.create(config, None, 0)


def test_registration_rejects_missing_required_configuration() -> None:
    config = _make_vllm_config()
    del config.parallel_config.world_size

    with pytest.raises(ValueError, match="parallel_config.world_size must be set"):
        SchedulerRegistration.create(config, None, 0)


@pytest.mark.parametrize(
    ("section", "field", "value", "error", "match"),
    [
        ("parallel_config", "tensor_parallel_size", 0, ValueError, "tensor_parallel_size must be positive"),
        ("cache_config", "block_size", "16", TypeError, "block_size must be an integer"),
        ("kv_transfer_config", "kv_role", "invalid", ValueError, "kv_role must be one of"),
        (
            "kv_transfer_config",
            "kv_connector_extra_config",
            [],
            TypeError,
            "kv_connector_extra_config must be a mapping",
        ),
    ],
)
def test_registration_rejects_invalid_required_configuration(section, field, value, error, match) -> None:
    config = _make_vllm_config()
    setattr(getattr(config, section), field, value)

    with pytest.raises(error, match=match):
        SchedulerRegistration.create(config, None, 0)


@pytest.mark.parametrize("page_size_bytes", ["4096", True])
def test_registration_rejects_non_integer_page_size_bytes(page_size_bytes) -> None:
    config = _make_vllm_config()

    with pytest.raises(TypeError, match="page_size_bytes must be an integer"):
        SchedulerRegistration.create(config, None, page_size_bytes)


def test_registration_rejects_negative_page_size_bytes() -> None:
    config = _make_vllm_config()

    with pytest.raises(ValueError, match="page_size_bytes must not be negative"):
        SchedulerRegistration.create(config, None, -1)


def test_registration_preserves_optional_configuration() -> None:
    config = _make_vllm_config()
    hf_config = SimpleNamespace(num_hidden_layers=2)
    config.model_config.hf_text_config = hf_config
    config.model_config.hf_config = hf_config
    config.scheduler_config.disable_hybrid_kv_cache_manager = None

    registration = SchedulerRegistration.create(config, None, 0)

    assert registration.config.model_config.model_type is None
    assert registration.config.model_config.compress_ratios is None
    assert registration.config.scheduler_config.disable_hybrid_kv_cache_manager is False
    assert registration.config.speculative_config is None
    assert registration.config.kv_events_config is None


def _worker_registration(session_id: str, *, data_parallel_rank: int = 0, rank: int = 0):
    return WorkerRegistration.create(
        _make_vllm_config(data_parallel_rank=data_parallel_rank, rank=rank),
        None,
        session_id=session_id,
    )


def _create_service_manager(scheduler_factory=None, worker_factory=None) -> KVCacheServiceManager:
    scheduler_factory = scheduler_factory or (lambda registration: _FakeScheduler())
    return KVCacheServiceManager(
        lambda registration, _lookup_handler: scheduler_factory(registration),
        worker_factory or (lambda registration: _FakeWorker()),
    )


def test_scheduler_factories_for_different_identities_run_in_parallel() -> None:
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    def scheduler_factory(registration):
        if registration.identity.data_parallel_rank == 0:
            first_started.set()
            assert release_first.wait(5), "First Scheduler factory was not released"
        else:
            second_started.set()
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    first_registration = _scheduler_registration("session-0", data_parallel_rank=0)
    second_registration = _scheduler_registration("session-1", data_parallel_rank=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            service_manager.register_scheduler, first_registration, encode_registration(first_registration)
        )
        assert first_started.wait(5), "First Scheduler factory did not start"

        second_future = executor.submit(
            service_manager.register_scheduler, second_registration, encode_registration(second_registration)
        )
        try:
            assert second_started.wait(1), "Second Scheduler factory was blocked by the first registration"
        finally:
            release_first.set()

        assert isinstance(first_future.result(timeout=5), _FakeScheduler)
        assert isinstance(second_future.result(timeout=5), _FakeScheduler)

    assert service_manager.scheduler_count == 2


def test_concurrent_identical_scheduler_registration_shares_one_factory_result() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    created_schedulers = []

    def scheduler_factory(registration):
        scheduler = _FakeScheduler()
        created_schedulers.append(scheduler)
        factory_started.set()
        assert release_factory.wait(5), "Scheduler factory was not released"
        return scheduler

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    registration = _scheduler_registration("session-0")
    payload = encode_registration(registration)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(service_manager.register_scheduler, registration, payload)
        assert factory_started.wait(5), "Scheduler factory did not start"
        second_future = executor.submit(service_manager.register_scheduler, registration, payload)
        release_factory.set()

        first_service = first_future.result(timeout=5)
        second_service = second_future.result(timeout=5)

    assert first_service is second_service
    assert created_schedulers == [first_service]
    assert service_manager.scheduler_count == 1


def test_conflicting_scheduler_registration_fails_while_original_is_registering() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()

    def scheduler_factory(registration):
        factory_started.set()
        assert release_factory.wait(5), "Scheduler factory was not released"
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    registration = _scheduler_registration("session-0", block_size=16)
    conflicting = _scheduler_registration("session-0", block_size=32)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service_manager.register_scheduler, registration, encode_registration(registration))
        assert factory_started.wait(5), "Scheduler factory did not start"

        try:
            with pytest.raises(RegistrationConflictError, match="different configuration"):
                service_manager.register_scheduler(conflicting, encode_registration(conflicting))
        finally:
            release_factory.set()

        assert isinstance(future.result(timeout=5), _FakeScheduler)


def test_concurrent_scheduler_registration_shares_factory_failure_and_next_request_retries() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    attempts = 0

    def scheduler_factory(registration):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            factory_started.set()
            assert release_factory.wait(5), "Scheduler factory was not released"
            raise RuntimeError("factory failed")
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    registration = _scheduler_registration("session-0")
    payload = encode_registration(registration)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(service_manager.register_scheduler, registration, payload)
        assert factory_started.wait(5), "Scheduler factory did not start"
        second_future = executor.submit(service_manager.register_scheduler, registration, payload)
        release_factory.set()

        with pytest.raises(RuntimeError, match="factory failed"):
            first_future.result(timeout=5)
        with pytest.raises(RuntimeError, match="factory failed"):
            second_future.result(timeout=5)

    assert attempts == 1
    assert service_manager.scheduler_count == 0

    assert isinstance(service_manager.register_scheduler(registration, payload), _FakeScheduler)
    assert attempts == 2
    assert service_manager.scheduler_count == 1


def test_new_scheduler_session_replaces_and_retires_old_session() -> None:
    created = []

    def scheduler_factory(registration):
        scheduler = _FakeScheduler()
        created.append(scheduler)
        return scheduler

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    old_registration = _scheduler_registration("old-session")
    new_registration = _scheduler_registration("new-session")

    old_service = service_manager.register_scheduler(old_registration, encode_registration(old_registration))
    new_service = service_manager.register_scheduler(new_registration, encode_registration(new_registration))

    assert old_service.close_count == 1
    assert new_service is created[1]
    assert service_manager.lookup(new_registration.identity, "new-session", SimpleNamespace(), 0) == (0, False)

    with pytest.raises(StaleSessionError, match="retired"):
        service_manager.register_scheduler(old_registration, encode_registration(old_registration))
    with pytest.raises(StaleSessionError):
        service_manager.lookup(old_registration.identity, "old-session", SimpleNamespace(), 0)


def test_old_session_is_fenced_while_new_session_is_registering() -> None:
    new_factory_started = threading.Event()
    release_new_factory = threading.Event()

    def scheduler_factory(registration):
        if registration.session_id == "new-session":
            new_factory_started.set()
            assert release_new_factory.wait(5), "New Scheduler factory was not released"
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    old_registration = _scheduler_registration("old-session")
    new_registration = _scheduler_registration("new-session")
    service_manager.register_scheduler(old_registration, encode_registration(old_registration))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            service_manager.register_scheduler, new_registration, encode_registration(new_registration)
        )
        assert new_factory_started.wait(5), "New Scheduler factory did not start"
        try:
            with pytest.raises(StaleSessionError, match="retired"):
                service_manager.register_scheduler(old_registration, encode_registration(old_registration))
        finally:
            release_new_factory.set()

        assert isinstance(future.result(timeout=5), _FakeScheduler)


def test_different_new_session_is_rejected_while_session_transition_is_running() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()

    def scheduler_factory(registration):
        if registration.session_id == "session-1":
            factory_started.set()
            assert release_factory.wait(5), "Scheduler factory was not released"
        return _FakeScheduler()

    service_manager = _create_service_manager(scheduler_factory=scheduler_factory)
    first = _scheduler_registration("session-0")
    second = _scheduler_registration("session-1")
    third = _scheduler_registration("session-2")
    service_manager.register_scheduler(first, encode_registration(first))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service_manager.register_scheduler, second, encode_registration(second))
        assert factory_started.wait(5), "Scheduler factory did not start"
        try:
            with pytest.raises(RegistrationConflictError, match="already registering session"):
                service_manager.register_scheduler(third, encode_registration(third))
        finally:
            release_factory.set()

        assert isinstance(future.result(timeout=5), _FakeScheduler)


def test_unregister_retires_current_session_and_closes_service() -> None:
    service_manager = _create_service_manager()
    registration = _scheduler_registration("session-0")
    service = service_manager.register_scheduler(registration, encode_registration(registration))

    assert service_manager.unregister_scheduler(registration.identity, registration.session_id)
    assert service.close_count == 1
    assert service_manager.scheduler_count == 0

    with pytest.raises(StaleSessionError, match="retired"):
        service_manager.register_scheduler(registration, encode_registration(registration))


def test_concurrent_identical_worker_registration_shares_one_factory_result() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    created_workers = []

    def worker_factory(registration):
        worker = _FakeWorker()
        created_workers.append(worker)
        factory_started.set()
        assert release_factory.wait(5), "Worker factory was not released"
        return worker

    service_manager = _create_service_manager(worker_factory=worker_factory)
    registration = _worker_registration("worker-session")
    payload = encode_registration(registration)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(service_manager.register_worker, registration, payload)
        assert factory_started.wait(5), "Worker factory did not start"
        second_future = executor.submit(service_manager.register_worker, registration, payload)
        release_factory.set()

        first_service = first_future.result(timeout=5)
        second_service = second_future.result(timeout=5)

    assert first_service is second_service
    assert created_workers == [first_service]
    assert service_manager.worker_count == 1
