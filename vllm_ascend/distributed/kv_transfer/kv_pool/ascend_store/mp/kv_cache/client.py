"""Client-side access to KV cache services."""

import contextlib
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from vllm.config import VllmConfig
from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from ...metadata import AscendConnectorMetadata, AscendStoreKVConnectorWorkerMetadata
from ..rpc import MPClient, MPRemoteError, MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError
from .error import (
    SERVICE_NOT_REGISTERED_PREFIX,
    STALE_SESSION_PREFIX,
    ServiceNotRegisteredError,
    ServiceSessionExpiredError,
)
from .npu_ipc import NPUEventSpec, WorkerKVCacheSpec
from .protocol import (
    KVCacheMethod,
    decode_ack_response,
    decode_build_connector_meta_response,
    decode_build_connector_worker_meta_response,
    decode_get_block_ids_with_load_errors_response,
    decode_get_finished_response,
    decode_get_kv_events_response,
    decode_lookup_response,
    decode_request_finished_response,
    decode_update_connector_output_response,
    encode_build_connector_meta_request,
    encode_build_connector_worker_meta_request,
    encode_get_block_ids_with_load_errors_request,
    encode_get_finished_request,
    encode_get_kv_events_request,
    encode_lookup_request,
    encode_register_kv_caches_request,
    encode_registration_request,
    encode_request_finished,
    encode_save_kv_layer_request,
    encode_scheduler_session,
    encode_start_load_kv_request,
    encode_update_connector_output,
    encode_update_state_after_alloc,
    encode_wait_for_layer_load_request,
    encode_wait_for_save_request,
    encode_worker_session,
)
from .registration import SchedulerRegistration, WorkerRegistration

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 5000
_REGISTRATION_TIMEOUT_MS = 500
# First-time cache registration imports NPU IPC mappings, creates the backend,
# and registers its buffers, which exceeds the identity-registration budget on
# real hardware. A timeout here only triggers an idempotent retry.
_CACHE_REGISTRATION_TIMEOUT_MS = 5000
# Unregistering closes the service on the server: the Worker stops transfer
# threads and unregisters backend buffers, which can also exceed the fast-fail
# identity budget. Cleanup stays best-effort, with lease expiry as backstop.
_UNREGISTER_TIMEOUT_MS = 5000
_LEASE_RENEW_INTERVAL_MS = 1000
_LEASE_REQUEST_TIMEOUT_MS = 1000
_ConfiguredRegistration = tuple[SchedulerRegistration | WorkerRegistration, tuple[bytes, ...]]


@dataclass(frozen=True)
class _WorkerKVCacheRegistration:
    """Worker cache registration retained for service recovery."""

    spec: WorkerKVCacheSpec
    payloads: tuple[bytes, ...]


class _RegistrationState(Enum):
    """Client-local knowledge of the configured service registration."""

    UNCONFIGURED = auto()
    UNREGISTERED = auto()
    REGISTERING = auto()
    REGISTERED = auto()
    SUPERSEDED = auto()


class KVCacheClient:
    """Expose typed KV cache RPCs through one recoverable service session.

    The client owns service registration and lease recovery. It also decides
    which remote failures are returned and which become cache misses or no-ops.
    """

    def __init__(self, server_url: str):
        self._rpc_client = MPClient(server_url)
        self._client_lifecycle_lock = threading.Lock()
        self._registration_attempt_lock = threading.Lock()
        self._lease_lock = threading.Lock()
        self._lease_stop = threading.Event()
        self._lease_thread: threading.Thread | None = None
        self._registration: _ConfiguredRegistration | None = None
        self._worker_kv_cache_registration: _WorkerKVCacheRegistration | None = None
        self._session_id = uuid.uuid4().hex
        self._registration_state = _RegistrationState.UNCONFIGURED
        self._last_reported_degradation: tuple[type[BaseException], str] | None = None
        self._closed = False

    def __enter__(self) -> "KVCacheClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._rpc_client.is_transport_connected

    @property
    def is_registered(self) -> bool:
        with self._client_lifecycle_lock:
            return not self._closed and self._registration_state is _RegistrationState.REGISTERED

    # ==============================
    # Recoverable service registration
    # ==============================

    # A client owns one configured Scheduler or Worker session and reuses it
    # across recoverable failures. Only one registration attempt runs at a time;
    # if Worker cache information already exists, it must be registered again
    # before the recovered session is reported as registered.

    def register_scheduler(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None, page_size_bytes: int
    ) -> bool:
        registration = SchedulerRegistration.create(
            vllm_config, kv_cache_config, page_size_bytes, session_id=self._session_id
        )
        return self._configure_registration(registration)

    def register_worker(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> bool:
        registration = WorkerRegistration.create(vllm_config, kv_cache_config, session_id=self._session_id)
        return self._configure_registration(registration)

    def _configure_registration(self, registration: SchedulerRegistration | WorkerRegistration) -> bool:
        with self._client_lifecycle_lock:
            if self._closed:
                raise RuntimeError("KVCacheClient is closed")
            if self._registration_state is _RegistrationState.SUPERSEDED:
                raise ServiceSessionExpiredError("KV cache service session has been superseded")
            if self._registration is not None and type(self._registration[0]) is not type(registration):
                raise RuntimeError("A KVCacheClient cannot register both Scheduler and Worker services")

            self._registration = (registration, encode_registration_request(registration))
            self._registration_state = _RegistrationState.UNREGISTERED

        registered = self._try_register()
        self._start_lease_loop()
        return registered

    def register_kv_caches(self, spec: WorkerKVCacheSpec, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> bool:
        """Register the Worker's fixed cache mapping."""
        self._raise_if_superseded()
        registration = self._get_worker_registration()
        payloads = encode_register_kv_caches_request(registration, spec)
        cache_registration = _WorkerKVCacheRegistration(spec, payloads)
        with self._client_lifecycle_lock:
            if self._worker_kv_cache_registration is not None:
                raise RuntimeError("Worker KV caches are already registered")
            self._worker_kv_cache_registration = cache_registration

        try:
            if not self.is_registered:
                return self._try_register()

            responses = self._try_worker_rpc(
                KVCacheMethod.REGISTER_KV_CACHES,
                lambda _registration: payloads,
                timeout_ms,
            )
            if responses is None:
                # Service recovery registers the same cache mapping again.
                self._mark_unregistered()
                return False
            decode_ack_response(responses, KVCacheMethod.REGISTER_KV_CACHES)
            return True
        except BaseException:
            with self._client_lifecycle_lock:
                if self._worker_kv_cache_registration is cache_registration:
                    self._worker_kv_cache_registration = None
            raise

    def _try_register(self) -> bool:
        try:
            return self._register()
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError, ServiceNotRegisteredError) as exc:
            self._report_degradation("REGISTER_SERVICE", exc)
            return False

    def _register(self) -> bool:
        with self._registration_attempt_lock:
            with self._client_lifecycle_lock:
                if self._closed:
                    return False
                if self._registration_state is _RegistrationState.SUPERSEDED:
                    raise ServiceSessionExpiredError("KV cache service session has been superseded")

                configured_registration = self._registration
                worker_kv_cache_registration = self._worker_kv_cache_registration
                if configured_registration is None:
                    return False
                if self._registration_state is _RegistrationState.REGISTERED:
                    return True
                self._registration_state = _RegistrationState.REGISTERING

            registration, payloads = configured_registration
            method = (
                KVCacheMethod.REGISTER_SCHEDULER
                if isinstance(registration, SchedulerRegistration)
                else KVCacheMethod.REGISTER_WORKER
            )

            if not self._rpc_client.is_transport_connected:
                self._mark_unregistered()
                raise MPServerUnavailableError("MP client transport is unavailable")

            try:
                responses = self._send_service_request(method, payloads, _REGISTRATION_TIMEOUT_MS)
                decode_ack_response(responses, method)
                if isinstance(registration, WorkerRegistration) and worker_kv_cache_registration is not None:
                    responses = self._send_service_request(
                        KVCacheMethod.REGISTER_KV_CACHES,
                        worker_kv_cache_registration.payloads,
                        _CACHE_REGISTRATION_TIMEOUT_MS,
                    )
                    decode_ack_response(responses, KVCacheMethod.REGISTER_KV_CACHES)
            except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError, ServiceNotRegisteredError):
                self._mark_unregistered()
                raise
            except ServiceSessionExpiredError:
                self._mark_superseded()
                raise
            except BaseException:
                self._mark_unregistered()
                raise

            self._clear_reported_degradation()
            with self._client_lifecycle_lock:
                if self._registration is not configured_registration:
                    return False
                if self._registration_state is _RegistrationState.SUPERSEDED:
                    return False
                if self._worker_kv_cache_registration is not worker_kv_cache_registration:
                    self._registration_state = _RegistrationState.UNREGISTERED
                    return False
                self._registration_state = _RegistrationState.REGISTERED
                return not self._closed

    # ==============================
    # Lease-driven registration recovery
    # ==============================

    # The lease thread both keeps a registered session alive and restores it
    # after temporary loss. It retries the same configuration, but stops
    # permanently when the server says another session has replaced this one.

    def _start_lease_loop(self) -> None:
        with self._client_lifecycle_lock:
            if self._closed:
                return

            with self._lease_lock:
                if self._lease_thread is not None and self._lease_thread.is_alive():
                    return

                self._lease_stop.clear()
                self._lease_thread = threading.Thread(
                    target=self._lease_loop, daemon=True, name="ascend-store-kv-lease"
                )
                self._lease_thread.start()

    def _lease_loop(self) -> None:
        interval_s = _LEASE_RENEW_INTERVAL_MS / 1000
        while not self._lease_stop.wait(interval_s):
            try:
                self._maintain_lease()
            except Exception:
                logger.exception("KV cache service lease maintenance failed")

    def _maintain_lease(self) -> None:
        with self._client_lifecycle_lock:
            if self._closed or self._registration is None or self._registration_state is _RegistrationState.SUPERSEDED:
                return
            registration = self._registration[0]
            registered = self._registration_state is _RegistrationState.REGISTERED

        if not registered:
            with contextlib.suppress(ServiceSessionExpiredError):
                self._try_register()
            return

        if isinstance(registration, SchedulerRegistration):
            method = KVCacheMethod.RENEW_SCHEDULER
            payloads = encode_scheduler_session(registration.identity, registration.session_id)
        else:
            method = KVCacheMethod.RENEW_WORKER
            payloads = encode_worker_session(registration.identity, registration.session_id)

        try:
            responses = self._send_service_request(method, payloads, _LEASE_REQUEST_TIMEOUT_MS)
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError):
            self._mark_unregistered()
            return
        except ServiceNotRegisteredError:
            self._mark_unregistered()
            with contextlib.suppress(ServiceSessionExpiredError):
                self._try_register()
            return
        except ServiceSessionExpiredError:
            self._mark_superseded()
            return

        decode_ack_response(responses, method)
        self._clear_reported_degradation()

    def _stop_lease_loop(self) -> None:
        with self._lease_lock:
            lease_thread = self._lease_thread
            if lease_thread is None:
                return
            self._lease_stop.set()

        if lease_thread is not threading.current_thread():
            lease_thread.join()

        with self._lease_lock:
            if self._lease_thread is lease_thread:
                self._lease_thread = None

    # ==============================
    # Scheduler service operations
    # ==============================

    # Scheduler methods follow the vLLM request lifecycle. Recoverable failures
    # become cache misses or no-ops, allowing scheduling to continue correctly
    # without the remote cache.

    def lookup(
        self,
        request: Request,
        num_computed_tokens: int,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> tuple[int, bool]:
        responses = self._try_scheduler_rpc(
            KVCacheMethod.LOOKUP,
            lambda registration: encode_lookup_request(registration, request, num_computed_tokens),
            timeout_ms,
        )
        return decode_lookup_response(responses) if responses is not None else (0, False)

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> None:
        responses = self._try_scheduler_rpc(
            KVCacheMethod.UPDATE_STATE_AFTER_ALLOC,
            lambda registration: encode_update_state_after_alloc(registration, request, blocks, num_external_tokens),
            timeout_ms,
        )
        if responses is not None:
            decode_ack_response(responses, KVCacheMethod.UPDATE_STATE_AFTER_ALLOC)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        new_token_ids: dict[str, list[int]],
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> tuple | None:
        """Return (metadata, touch_block_ids) or None when the remote call fails."""
        responses = self._try_scheduler_rpc(
            KVCacheMethod.BUILD_CONNECTOR_META,
            lambda registration: encode_build_connector_meta_request(registration, scheduler_output, new_token_ids),
            timeout_ms,
        )
        return decode_build_connector_meta_response(responses) if responses is not None else None

    def request_finished(
        self,
        request_id: str,
        block_ids,
        all_groups: bool = False,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> tuple[bool, dict | None]:
        responses = self._try_scheduler_rpc(
            KVCacheMethod.REQUEST_FINISHED,
            lambda registration: encode_request_finished(registration, request_id, block_ids, all_groups),
            timeout_ms,
        )
        return decode_request_finished_response(responses) if responses is not None else (False, None)

    def update_connector_output(
        self,
        completed_events: dict[int, int],
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> list[int]:
        """Report worker completion counts; return block ids to free locally."""
        responses = self._try_scheduler_rpc(
            KVCacheMethod.UPDATE_CONNECTOR_OUTPUT,
            lambda registration: encode_update_connector_output(registration, completed_events),
            timeout_ms,
        )
        return decode_update_connector_output_response(responses) if responses is not None else []

    # ==============================
    # Worker service operations
    # ==============================

    # Worker RPCs follow the connector's transfer phases. Most coordination calls
    # return an empty or unsuccessful result after a recoverable failure, while
    # layer-by-layer wait and save calls report infrastructure errors to the
    # connector.

    def wait_for_save(
        self,
        metadata: AscendConnectorMetadata,
        event_spec: NPUEventSpec,
        timeout_ms: int | None = None,
    ) -> bool:
        """Wait without a default deadline so the source Event outlives accepted Store work."""
        responses = self._try_worker_rpc(
            KVCacheMethod.WAIT_FOR_SAVE,
            lambda registration: encode_wait_for_save_request(registration, metadata, event_spec),
            timeout_ms,
        )
        if responses is None:
            return False
        decode_ack_response(responses, KVCacheMethod.WAIT_FOR_SAVE)
        return True

    def get_finished(
        self,
        finished_req_ids: set[str],
        metadata: AscendConnectorMetadata,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> tuple[set[str], set[str]]:
        responses = self._try_worker_rpc(
            KVCacheMethod.GET_FINISHED,
            lambda registration: encode_get_finished_request(registration, finished_req_ids, metadata),
            timeout_ms,
        )
        return decode_get_finished_response(responses) if responses is not None else (set(), set())

    def build_connector_worker_meta(
        self,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> AscendStoreKVConnectorWorkerMetadata | None:
        responses = self._try_worker_rpc(
            KVCacheMethod.BUILD_CONNECTOR_WORKER_META,
            encode_build_connector_worker_meta_request,
            timeout_ms,
        )
        return decode_build_connector_worker_meta_response(responses) if responses is not None else None

    def get_kv_events(self, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> list[BlockStored]:
        responses = self._try_worker_rpc(KVCacheMethod.GET_KV_EVENTS, encode_get_kv_events_request, timeout_ms)
        return decode_get_kv_events_response(responses) if responses is not None else []

    def start_load_kv(self, metadata: AscendConnectorMetadata, timeout_ms: int | None = None) -> bool:
        responses = self._try_worker_rpc(
            KVCacheMethod.START_LOAD_KV,
            lambda registration: encode_start_load_kv_request(registration, metadata),
            timeout_ms,
        )
        if responses is None:
            return False
        decode_ack_response(responses, KVCacheMethod.START_LOAD_KV)
        return True

    def wait_for_layer_load(self, timeout_ms: int | None = None) -> None:
        responses = self._worker_rpc(KVCacheMethod.WAIT_FOR_LAYER_LOAD, encode_wait_for_layer_load_request, timeout_ms)
        decode_ack_response(responses, KVCacheMethod.WAIT_FOR_LAYER_LOAD)

    def save_kv_layer(self, event_spec: NPUEventSpec, timeout_ms: int | None = None) -> None:
        responses = self._worker_rpc(
            KVCacheMethod.SAVE_KV_LAYER,
            lambda registration: encode_save_kv_layer_request(registration, event_spec),
            timeout_ms,
        )
        decode_ack_response(responses, KVCacheMethod.SAVE_KV_LAYER)

    def get_block_ids_with_load_errors(self, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> set[int] | None:
        responses = self._try_worker_rpc(
            KVCacheMethod.GET_BLOCK_IDS_WITH_LOAD_ERRORS,
            encode_get_block_ids_with_load_errors_request,
            timeout_ms,
        )
        return decode_get_block_ids_with_load_errors_response(responses) if responses is not None else None

    # ==============================
    # Registered requests and failure handling
    # ==============================

    # Every business RPC checks the configured role and obtains a registered
    # session before sending. Busy affects only the current call; a timeout,
    # lost transport, or missing service triggers registration again, while a
    # stale session permanently invalidates this client.

    def _scheduler_rpc(
        self,
        method: KVCacheMethod,
        encode: Callable[[SchedulerRegistration], tuple[bytes, ...]],
        timeout_ms: int | None,
    ) -> list[bytes]:
        self._raise_if_superseded()
        registration = self._get_scheduler_registration()
        payloads = encode(registration)
        return self._request_registered_service(method, payloads, timeout_ms)

    def _try_scheduler_rpc(
        self,
        method: KVCacheMethod,
        encode: Callable[[SchedulerRegistration], tuple[bytes, ...]],
        timeout_ms: int | None,
    ) -> list[bytes] | None:
        try:
            return self._scheduler_rpc(method, encode, timeout_ms)
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError, ServiceNotRegisteredError) as exc:
            self._report_degradation(method, exc)
            return None

    def _worker_rpc(
        self,
        method: KVCacheMethod,
        encode: Callable[[WorkerRegistration], tuple[bytes, ...]],
        timeout_ms: int | None,
    ) -> list[bytes]:
        self._raise_if_superseded()
        registration = self._get_worker_registration()
        payloads = encode(registration)
        return self._request_registered_service(method, payloads, timeout_ms)

    def _try_worker_rpc(
        self,
        method: KVCacheMethod,
        encode: Callable[[WorkerRegistration], tuple[bytes, ...]],
        timeout_ms: int | None,
    ) -> list[bytes] | None:
        try:
            return self._worker_rpc(method, encode, timeout_ms)
        except (MPRequestTimeoutError, MPServerBusyError, MPServerUnavailableError, ServiceNotRegisteredError) as exc:
            self._report_degradation(method, exc)
            return None

    def _request_registered_service(
        self,
        method: KVCacheMethod,
        payloads: tuple[bytes, ...],
        timeout_ms: int | None,
    ) -> list[bytes]:
        if not self.is_registered and not self._register():
            raise MPServerUnavailableError("KV cache service registration is unavailable")

        try:
            responses = self._send_service_request(method, payloads, timeout_ms)
        except (MPRequestTimeoutError, MPServerUnavailableError, ServiceNotRegisteredError):
            self._mark_unregistered()
            raise
        except ServiceSessionExpiredError:
            self._mark_superseded()
            raise
        self._clear_reported_degradation()
        return responses

    def _send_service_request(
        self,
        method: KVCacheMethod,
        payloads: tuple[bytes, ...],
        timeout_ms: int | None,
    ) -> list[bytes]:
        """Send one request and translate errors defined by the KV cache service."""
        try:
            return self._rpc_client.request(method, payloads, timeout_ms=timeout_ms)
        except MPRemoteError as exc:
            message = str(exc)
            if message.startswith(SERVICE_NOT_REGISTERED_PREFIX):
                raise ServiceNotRegisteredError(message) from exc
            if message.startswith(STALE_SESSION_PREFIX):
                raise ServiceSessionExpiredError(message) from exc
            raise

    def _get_scheduler_registration(self) -> SchedulerRegistration:
        with self._client_lifecycle_lock:
            configured_registration = self._registration

        if configured_registration is None or not isinstance(configured_registration[0], SchedulerRegistration):
            raise RuntimeError("KVCacheClient is not configured as a Scheduler client")
        return configured_registration[0]

    def _get_worker_registration(self) -> WorkerRegistration:
        with self._client_lifecycle_lock:
            configured_registration = self._registration

        if configured_registration is None or not isinstance(configured_registration[0], WorkerRegistration):
            raise RuntimeError("KVCacheClient is not configured as a Worker client")
        return configured_registration[0]

    def _mark_unregistered(self) -> None:
        with self._client_lifecycle_lock:
            if self._registration_state is not _RegistrationState.SUPERSEDED:
                self._registration_state = _RegistrationState.UNREGISTERED

    def _mark_superseded(self) -> None:
        with self._client_lifecycle_lock:
            self._registration_state = _RegistrationState.SUPERSEDED

    def _raise_if_superseded(self) -> None:
        with self._client_lifecycle_lock:
            if self._registration_state is _RegistrationState.SUPERSEDED:
                raise ServiceSessionExpiredError("KV cache service session has been superseded")

    def _report_degradation(self, method: KVCacheMethod | str, error: BaseException) -> None:
        signature = type(error), str(error)
        with self._client_lifecycle_lock:
            if self._last_reported_degradation == signature:
                return
            self._last_reported_degradation = signature

        method_name = method.value if isinstance(method, KVCacheMethod) else method
        logger.warning("KV cache RPC %s degraded. type=%s, error=%s", method_name, type(error).__name__, error)

    def _clear_reported_degradation(self) -> None:
        with self._client_lifecycle_lock:
            self._last_reported_degradation = None

    # ==============================
    # Client shutdown
    # ==============================

    # Shutdown stops lease recovery and waits for any service registration already
    # in progress, including Worker cache setup after recovery. It unregisters
    # before closing the transport, so a timed-out registration that may have
    # succeeded remotely still gets one cleanup attempt.

    def _unregister(self) -> None:
        with self._client_lifecycle_lock:
            configured_registration = self._registration
            should_unregister = self._registration_state not in {
                _RegistrationState.UNCONFIGURED,
                _RegistrationState.SUPERSEDED,
            }
            if should_unregister:
                self._registration_state = _RegistrationState.UNREGISTERED

        if configured_registration is None or not should_unregister or not self._rpc_client.is_transport_connected:
            return

        registration = configured_registration[0]
        if isinstance(registration, SchedulerRegistration):
            method = KVCacheMethod.UNREGISTER_SCHEDULER
            payloads = encode_scheduler_session(registration.identity, registration.session_id)
        else:
            method = KVCacheMethod.UNREGISTER_WORKER
            payloads = encode_worker_session(registration.identity, registration.session_id)

        try:
            responses = self._send_service_request(method, payloads, _UNREGISTER_TIMEOUT_MS)
            decode_ack_response(responses, method)
        except Exception:
            # close() is best-effort cleanup; transport recovery is no longer useful once the client is closing.
            logger.debug("Failed to unregister KV cache service during client close", exc_info=True)

    def close(self) -> None:
        with self._client_lifecycle_lock:
            if self._closed:
                return
            self._closed = True

        self._stop_lease_loop()
        # Wait for the current service registration and its Worker cache setup
        # before deciding whether unregister is needed.
        with self._registration_attempt_lock:
            self._unregister()
        self._rpc_client.close()
        with self._client_lifecycle_lock:
            self._worker_kv_cache_registration = None
