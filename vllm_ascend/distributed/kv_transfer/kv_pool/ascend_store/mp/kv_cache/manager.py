"""KV cache service orchestration independent of the RPC transport."""

import hashlib
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING

from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

from ...metadata import AscendConnectorMetadata, AscendStoreKVConnectorWorkerMetadata
from ..rpc import TaskExecutor
from ..service import ServiceLifecycleManager
from .error import ServiceNotRegisteredError
from .npu_ipc import NPUEventSpec, WorkerKVCacheSpec
from .registration import (
    SchedulerIdentity,
    SchedulerRegistration,
    WorkerIdentity,
    WorkerLookupHandler,
    WorkerRegistration,
)
from .scheduler_view import BlocksView, ConnectorOutputView, RequestIdView, RequestView, SchedulerOutputView

if TYPE_CHECKING:
    from .pool.scheduler import MPKVPoolScheduler
    from .pool.worker import MPKVPoolWorker

SchedulerFactory = Callable[[SchedulerRegistration, WorkerLookupHandler], "MPKVPoolScheduler"]
WorkerFactory = Callable[[WorkerRegistration], "MPKVPoolWorker"]

_LOOKUP_COORDINATOR_RANK = 0
_SERVICE_LEASE_TIMEOUT_S = 60.0
_LEASE_CHECK_INTERVAL_S = 5.0


class KVCacheServiceManager:
    """Own scheduler and worker lifecycles behind a transport-neutral API.

    Requests are accepted only for current sessions. Cross-service lookup and
    automatic cleanup use the executor thread assigned to each service when
    executors are provided.

    Optional factories let tests replace or observe service construction while
    exercising the real lifecycle paths. Production uses the built-in MP
    Scheduler and Worker implementations; the factories are not business
    extension points.
    """

    def __init__(
        self,
        scheduler_factory: SchedulerFactory | None = None,
        worker_factory: WorkerFactory | None = None,
        lease_timeout_s: float = _SERVICE_LEASE_TIMEOUT_S,
        lease_check_interval_s: float = _LEASE_CHECK_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        scheduler_executor: TaskExecutor | None = None,
        worker_executor: TaskExecutor | None = None,
    ):
        self._scheduler_factory = scheduler_factory or self._create_scheduler
        self._worker_factory = worker_factory or self._create_worker
        self._worker_executor = worker_executor
        self._schedulers = ServiceLifecycleManager[SchedulerIdentity, "MPKVPoolScheduler"](
            "Scheduler",
            self._close_service,
            lease_timeout_s=lease_timeout_s,
            check_interval_s=lease_check_interval_s,
            clock=clock,
            thread_name="ascend-store-scheduler-lifecycle",
            owner_close_handler=partial(self._close_service_on_owner, scheduler_executor),
        )
        self._workers = ServiceLifecycleManager[WorkerIdentity, "MPKVPoolWorker"](
            "Worker",
            self._close_service,
            lease_timeout_s=lease_timeout_s,
            check_interval_s=lease_check_interval_s,
            clock=clock,
            thread_name="ascend-store-worker-lifecycle",
            owner_close_handler=partial(self._close_service_on_owner, worker_executor),
        )

    @property
    def scheduler_count(self) -> int:
        return self._schedulers.count

    @property
    def worker_count(self) -> int:
        return self._workers.count

    @staticmethod
    def _create_scheduler(
        registration: SchedulerRegistration, lookup_handler: WorkerLookupHandler
    ) -> "MPKVPoolScheduler":
        from .pool.scheduler import MPKVPoolScheduler

        return MPKVPoolScheduler(registration, lookup_handler)

    @staticmethod
    def _create_worker(registration: WorkerRegistration) -> "MPKVPoolWorker":
        from .pool.worker import MPKVPoolWorker

        return MPKVPoolWorker(
            registration.config,
            kv_cache_config=registration.config.build_kv_cache_config(),
            rank=registration.identity.rank,
        )

    def _build_scheduler(self, registration: SchedulerRegistration) -> "MPKVPoolScheduler":
        return self._scheduler_factory(registration, self._lookup_worker)

    # ==============================
    # Service registration and sessions
    # ==============================

    # Registration ties each config-derived identity to a session and a hash of
    # its registration payload. Retrying the same session is safe, while an older
    # session cannot renew or unregister its replacement.

    def register_scheduler(self, registration: SchedulerRegistration, payload: bytes) -> "MPKVPoolScheduler":
        self._validate_scheduler_registration(registration)
        scheduler = self._schedulers.register(
            registration.identity,
            registration.session_id,
            hashlib.sha256(payload).digest(),
            lambda: self._build_scheduler(registration),
        )
        return scheduler

    def register_worker(self, registration: WorkerRegistration, payload: bytes) -> "MPKVPoolWorker":
        self._validate_worker_registration(registration)
        worker = self._workers.register(
            registration.identity,
            registration.session_id,
            hashlib.sha256(payload).digest(),
            lambda: self._worker_factory(registration),
        )
        return worker

    def renew_scheduler(self, identity: SchedulerIdentity, session_id: str) -> None:
        if not self._schedulers.renew(identity, session_id):
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")

    def renew_worker(self, identity: WorkerIdentity, session_id: str) -> None:
        if not self._workers.renew(identity, session_id):
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")

    def unregister_scheduler(self, identity: SchedulerIdentity, session_id: str) -> bool:
        return self._schedulers.unregister(identity, session_id)

    def unregister_worker(self, identity: WorkerIdentity, session_id: str) -> bool:
        return self._workers.unregister(identity, session_id)

    @staticmethod
    def _validate_scheduler_registration(registration: SchedulerRegistration) -> None:
        expected_identity = SchedulerIdentity.from_config_spec(registration.config)
        if registration.identity != expected_identity:
            raise ValueError(
                f"Scheduler identity does not match registration config: "
                f"{registration.identity!r} != {expected_identity!r}"
            )

    @staticmethod
    def _validate_worker_registration(registration: WorkerRegistration) -> None:
        expected_identity = WorkerIdentity.from_config_spec(registration.config)
        if registration.identity != expected_identity:
            raise ValueError(
                f"Worker identity does not match registration config: "
                f"{registration.identity!r} != {expected_identity!r}"
            )

    # ==============================
    # Scheduler service operations
    # ==============================

    # Scheduler calls reuse the original KVPoolScheduler request state rather than
    # rebuilding it in the MP layer. Resolving the current session also renews its
    # lease, so normal request traffic keeps the Scheduler registered.

    def lookup(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        scheduler = self._require_scheduler(identity, session_id)
        return scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        request: RequestView,
        blocks: BlocksView,
        num_external_tokens: int,
    ) -> None:
        scheduler = self._require_scheduler(identity, session_id)
        # The inherited method stores the view in _unfinished_requests, which
        # doubles as the request registry for later business methods.
        scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, identity: SchedulerIdentity, session_id: str, output: SchedulerOutputView) -> tuple:
        scheduler = self._require_scheduler(identity, session_id)
        metadata = scheduler.build_connector_meta(output)
        return metadata, scheduler.take_block_pool_commands()

    def request_finished(
        self, identity: SchedulerIdentity, session_id: str, req_id: str, block_ids, all_groups: bool
    ) -> tuple:
        scheduler = self._require_scheduler(identity, session_id)
        request = RequestIdView(request_id=req_id)
        if all_groups:
            return scheduler.request_finished_all_groups(request, block_ids)
        return scheduler.request_finished(request, block_ids)

    def update_connector_output(
        self,
        identity: SchedulerIdentity,
        session_id: str,
        output: ConnectorOutputView,
    ) -> list[int]:
        scheduler = self._require_scheduler(identity, session_id)
        scheduler.update_connector_output(output)
        return scheduler.take_free_block_commands()

    def _require_scheduler(self, identity: SchedulerIdentity, session_id: str) -> "MPKVPoolScheduler":
        scheduler = self._schedulers.get_for_session(identity, session_id)
        if scheduler is None:
            raise ServiceNotRegisteredError(f"Scheduler {identity!r} is not registered")
        return scheduler

    # ==============================
    # Worker service operations
    # ==============================

    # Worker requests resolve the current session before touching backend state.
    # This rejects replaced sessions, while successful transfer requests also
    # renew the Worker's lease.

    def register_worker_kv_caches(
        self,
        identity: WorkerIdentity,
        session_id: str,
        spec: WorkerKVCacheSpec,
    ) -> None:
        worker = self._require_worker(identity, session_id)
        worker.configure_kv_caches(spec)

    def wait_for_save(
        self,
        identity: WorkerIdentity,
        session_id: str,
        metadata: AscendConnectorMetadata,
        event_spec: NPUEventSpec,
    ) -> None:
        worker = self._require_worker(identity, session_id)
        worker.wait_for_save(metadata, event_spec)

    def get_finished(
        self,
        identity: WorkerIdentity,
        session_id: str,
        finished_req_ids: set[str],
        metadata: AscendConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        worker = self._require_worker(identity, session_id)
        return worker.get_finished(finished_req_ids, metadata)

    def build_connector_worker_meta(
        self,
        identity: WorkerIdentity,
        session_id: str,
    ) -> AscendStoreKVConnectorWorkerMetadata | None:
        worker = self._require_worker(identity, session_id)
        return worker.build_connector_worker_meta()

    def get_kv_events(self, identity: WorkerIdentity, session_id: str) -> list[BlockStored]:
        worker = self._require_worker(identity, session_id)
        return worker.get_kv_events()

    def start_load_kv(self, identity: WorkerIdentity, session_id: str, metadata: AscendConnectorMetadata) -> None:
        worker = self._require_worker(identity, session_id)
        worker.start_load_kv(metadata)

    def wait_for_layer_load(self, identity: WorkerIdentity, session_id: str) -> None:
        worker = self._require_worker(identity, session_id)
        worker.wait_for_layer_load()

    def save_kv_layer(self, identity: WorkerIdentity, session_id: str, event_spec: NPUEventSpec) -> None:
        worker = self._require_worker(identity, session_id)
        worker.save_kv_layer_from_event(event_spec)

    def get_block_ids_with_load_errors(self, identity: WorkerIdentity, session_id: str) -> set[int]:
        worker = self._require_worker(identity, session_id)
        return worker.get_block_ids_with_load_errors()

    def _require_worker(self, identity: WorkerIdentity, session_id: str) -> "MPKVPoolWorker":
        worker = self._workers.get_for_session(identity, session_id)
        if worker is None:
            raise ServiceNotRegisteredError(f"Worker {identity!r} is not registered")
        return worker

    # ==============================
    # Scheduler-to-Worker lookup coordination
    # ==============================

    # Scheduler lookup targets rank zero in the same engine and data-parallel
    # group. With an executor, the lookup runs on the thread assigned to that
    # Worker. It deliberately uses find() without renewing the lease, so Scheduler
    # traffic cannot keep an otherwise idle Worker alive.

    def _lookup_worker(
        self,
        scheduler_identity: SchedulerIdentity,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None,
        use_layerwise: bool,
        hbm_hit_tokens: int,
    ) -> int:
        worker_identity = self._get_lookup_worker_identity(scheduler_identity)
        callback = partial(
            self._execute_worker_lookup,
            worker_identity,
            token_len,
            block_hashes,
            kv_cache_group_ids,
            use_layerwise,
            hbm_hit_tokens,
        )
        if self._worker_executor is None:
            return callback()
        return self._worker_executor.submit(callback, worker_identity).result()

    def _execute_worker_lookup(
        self,
        worker_identity: WorkerIdentity,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None,
        use_layerwise: bool,
        hbm_hit_tokens: int,
    ) -> int:
        worker = self._workers.find(worker_identity)
        if worker is None:
            return 0

        hash_strings = [block_hash.hex() for block_hash in block_hashes]
        return worker.lookup_scheduler(token_len, hash_strings, kv_cache_group_ids, use_layerwise, hbm_hit_tokens)

    @staticmethod
    def _get_lookup_worker_identity(scheduler_identity: SchedulerIdentity) -> WorkerIdentity:
        return WorkerIdentity(
            scheduler_identity.engine_id,
            rank=_LOOKUP_COORDINATOR_RANK,
            data_parallel_rank=scheduler_identity.data_parallel_rank,
        )

    # ==============================
    # Lease maintenance and service closure
    # ==============================

    # Scheduler and Worker lifecycles start and stop together. When a service has
    # an executor, expiry and shutdown send close() to the thread selected by that
    # service identity and wait for it, so the executors must remain alive until
    # this manager has closed its services.

    def start_lease_maintenance(self) -> None:
        self._schedulers.start_maintenance()
        self._workers.start_maintenance()

    def stop_lease_maintenance(self, wait: bool = True) -> None:
        self._workers.stop_maintenance(wait=wait)
        self._schedulers.stop_maintenance(wait=wait)

    def close(self) -> None:
        self._workers.close()
        self._schedulers.close()

    def _close_service_on_owner(
        self,
        executor: TaskExecutor | None,
        identity: SchedulerIdentity | WorkerIdentity,
        service: object,
    ) -> None:
        callback = partial(self._close_service, service)
        if executor is None:
            callback()
            return
        executor.submit(callback, identity, block=True).result()

    @staticmethod
    def _close_service(service: "MPKVPoolScheduler | MPKVPoolWorker") -> None:
        service.close()
