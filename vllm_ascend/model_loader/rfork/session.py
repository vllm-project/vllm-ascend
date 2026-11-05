# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import atexit
import threading
import time
from contextlib import contextmanager

from vllm.logger import logger

from vllm_ascend.model_loader.rfork.config import RForkConfig, RForkConfigurationError
from vllm_ascend.model_loader.rfork.planner_client import InvalidSeedLeaseError, RForkPlannerClient, lease_log_id
from vllm_ascend.model_loader.rfork.seed_client import build_seed_url, fetch_seed_transfer_info
from vllm_ascend.model_loader.rfork.seed_server import (
    RForkSeedServerHandle,
    create_rfork_server,
)
from vllm_ascend.model_loader.rfork.transfer_backend import RForkTransferBackend
from vllm_ascend.model_loader.rfork.types import (
    LeaseReleaseResult,
    RForkIdentity,
    RForkLifecycleState,
    SeedLease,
    SeedRemovalResult,
    SeedReportStatus,
    SeedTransferInfo,
)

HEARTBEAT_STOP_GRACE_SEC = 1.0
LEASE_RENEW_MIN_INTERVAL_SEC = 0.1
LEASE_RENEW_MAX_INTERVAL_SEC = 30.0
LEASE_RELEASE_DEGRADED_RETRY_INTERVAL_SEC = 60.0
RFORK_SEED_PORT_SLOTS_PER_RANK = 2


def _resolve_seed_server_port(config: RForkConfig, identity: RForkIdentity) -> int:
    if config.seed_port_base == 0:
        return 0
    model_slot = 1 if identity.is_draft_model else 0
    return config.seed_port_base + identity.global_rank * RFORK_SEED_PORT_SLOTS_PER_RANK + model_slot


def _registered_structural_digest(backend: RForkTransferBackend) -> str:
    """Return the full-inventory digest captured with the registration scan and its NPU formats."""
    digest = backend.registered_structural_digest
    if not isinstance(digest, str) or not digest:
        raise RuntimeError("RFork structural digest is unavailable after memory registration.")
    return digest


class RForkSessionUnavailableError(Exception):
    """The session cannot start another model load."""


class RForkSession:
    """Sole owner of one worker process's RFork runtime resources.

    Lock ordering: _seed_lifecycle_lock must be acquired before _lock when both are needed.
    _seed_lifecycle_lock serializes seed service start/stop across planner I/O.
    _lock protects session state, lease, and server handle.
    """

    def __init__(self, config: RForkConfig, identity: RForkIdentity) -> None:
        if not config.planner_url:
            raise RForkConfigurationError(
                "rfork_scheduler_url is required; configure it with model_loader_extra_config or RFORK_SCHEDULER_URL"
            )
        if not config.model_url or not config.model_deploy_strategy_name:
            raise RForkConfigurationError("RFork requires non-empty model_url and model_deploy_strategy_name")

        self.config = config
        self.identity = identity
        self.planner = RForkPlannerClient(config, identity)
        self.transfer_backend = RForkTransferBackend(tp_rank=identity.tp_rank)
        self.state = RForkLifecycleState.INITIALIZED
        self.seed_lease: SeedLease | None = None
        self.seed_server: RForkSeedServerHandle | None = None
        self.heartbeat_thread: threading.Thread | None = None
        self.heartbeat_stop_event = threading.Event()
        self._seed_worker_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self.lease_renew_stop_event = threading.Event()
        self._lease_release_in_flight = False
        self._lease_release_stopped = False
        self._source_transfer_session_id: str | None = None
        self._seed_publication_pending = False
        self._lock = threading.RLock()
        # Acquire before _lock; release responses must not wait on removal I/O.
        self._seed_lifecycle_lock = threading.RLock()
        self._shutdown_in_progress = False
        self._load_started = False

        def _atexit_shutdown() -> bool:
            # Only the interpreter-exit path treats unsupported planner removal as expected.
            return self.shutdown(final=True)

        self._atexit_callback = _atexit_shutdown
        atexit.register(self._atexit_callback)

    @contextmanager
    def loading(self):
        """Allow one model load and keep its resources alive until loading finishes."""
        with self._seed_lifecycle_lock:
            if self._shutdown_event.is_set():
                raise RForkSessionUnavailableError("RFork session is shutting down")
            if self._load_started:
                raise RForkSessionUnavailableError("RFork session has already loaded a model")
            self._load_started = True
            yield

    def register_receiver_weights(self, model) -> bool:
        """Register prepared NPU weight buffers with TransferEngine to receive seed weights."""
        with self._lock:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            if not self._register_weights(model, purpose="destination"):
                return False
            # Seed misses are silent 404s; the bound key and digest make them debuggable.
            log_identity = logger.info if self.identity.tp_rank == 0 else logger.debug
            log_identity(
                "RFork %s seed identity bound: seed_key=%s, structural_digest=%s",
                "draft" if self.identity.is_draft_model else "main",
                self.planner.seed_key,
                self.planner.structural_digest,
            )
            return True

    def acquire_seed_lease(self) -> bool:
        """Acquire a seed lease; return False if the planner supplies no usable lease."""
        with self._lock:
            acquisition_started = time.monotonic()
            try:
                self.seed_lease = self.planner.acquire_seed()
            except InvalidSeedLeaseError as exc:
                logger.warning("RFork planner seed acquisition failed: %s", exc)
                self.seed_lease = exc.lease
                self._release_seed_locked()
                return False
            if self.seed_lease is None:
                return False
            self._start_lease_renewal_locked(self.seed_lease)
            logger.debug(
                "RFork lease acquired: lease=%s global_rank=%s request_elapsed=%.3fs",
                lease_log_id(self.seed_lease),
                self.identity.global_rank,
                time.monotonic() - acquisition_started,
            )
            return True

    def read_weights_from_seed(self) -> bool:
        """Read seed weights into the registered buffers."""
        # Hold _lock through metadata/RDMA so cleanup cannot unregister buffers mid-transfer.
        with self._lock:
            # Shutdown releases the lease outside _lock; refuse transfers that would outlive it.
            if self._shutdown_event.is_set():
                logger.error("RFork transfer is refused because shutdown started; the seed lease will be released.")
                return False
            metadata_started = time.monotonic()
            seed_info = fetch_seed_transfer_info(
                build_seed_url(self.seed_lease.seed_ip, self.seed_lease.seed_port),
                self.planner.seed_key,
                self.config.request_timeout_sec,
            )
            if seed_info is None:
                return False
            metadata_elapsed = time.monotonic() - metadata_started
            read_started = time.monotonic()
            if not self.transfer_backend.read_weights_from_seed(seed_info):
                return False
            self._source_transfer_session_id = seed_info.session_id
            logger.debug(
                "RFork transfer stages: lease=%s global_rank=%s metadata=%.3fs read=%.3fs",
                lease_log_id(self.seed_lease),
                self.identity.global_rank,
                metadata_elapsed,
                time.monotonic() - read_started,
            )
            return True

    def validate_transferred_model_layout(self, model) -> None:
        """Verify the final model before it can be returned or published."""
        with self._lock:
            try:
                self.transfer_backend.validate_final_layout(model, peer_session_id=self._source_transfer_session_id)
            except Exception as exc:
                raise RuntimeError(f"RFork final tensor layout validation failed: {exc}") from exc
            self.state = RForkLifecycleState.READY

    def register_local_weights_for_seed(self, model) -> bool:
        """Register a locally loaded model before publishing it as a seed."""
        with self._lock:
            if self._shutdown_event.is_set() or (self.seed_lease is not None and self._lease_release_stopped):
                return False
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            try:
                registered = self._register_weights(model, purpose="seed_service")
            except Exception as exc:
                logger.warning("RFork seed registration failed or structure changed: %s", exc)
                registered = False
            if not registered:
                self._reset_transfer_locked()
                return False
            self.state = RForkLifecycleState.READY
            return True

    def schedule_seed_service(self) -> bool:
        """Publish the registered model after the source lease is released."""
        with self._seed_lifecycle_lock, self._lock:
            if self._shutdown_event.is_set() or self.state is not RForkLifecycleState.READY:
                return False
            self._seed_publication_pending = True
            if not self._ensure_seed_worker_locked():
                self._seed_publication_pending = False
                if self.seed_lease is None:
                    self._reset_transfer_locked()
                return False
            return True

    def prepare_for_fallback(self) -> bool:
        """Release the failed receiver's resources before loading locally."""
        with self._seed_lifecycle_lock, self._lock:
            self._seed_publication_pending = False
            self._release_seed_locked()
            return self._reset_transfer_locked()

    def shutdown(self, *, final: bool = False) -> bool:
        """Stop seed activity and release resources that are safe to release."""
        self._shutdown_event.set()
        with self._seed_lifecycle_lock:
            self.lease_renew_stop_event.set()
            with self._lock:
                if self.state is RForkLifecycleState.FINALIZED and self.seed_lease is None:
                    return True
                if self._shutdown_in_progress:
                    logger.debug("RFork shutdown is already in progress; skipping a reentrant cleanup request.")
                    return False
                self._shutdown_in_progress = True
                self._seed_publication_pending = False
            try:
                self._release_legacy_lease_on_shutdown()
                with self._lock:
                    if self.state is RForkLifecycleState.FINALIZED:
                        return self.seed_lease is None
                service_ok = self._stop_seed_service(final=final)
                with self._lock:
                    release_ok = self.seed_lease is None
                    legacy_lease_pending = self.seed_lease is not None and self.seed_lease.lease_ttl_sec is None
                    # The lock serializes local reads. Unpublished receiver memory is independent of legacy quota.
                    finalize_ok = (
                        self.transfer_backend.finalize_transfer_engine()
                        if service_ok and (release_ok or legacy_lease_pending)
                        else False
                    )
                    if finalize_ok:
                        self.state = RForkLifecycleState.FINALIZED
                        if release_ok:
                            atexit.unregister(self._atexit_callback)
                        else:
                            logger.warning(
                                "RFork finalized the receiver backend but its legacy planner allocation remains "
                                "unreleased. The allocation has no lease expiry; planner recovery requires put_seed "
                                "or external reclamation."
                            )
                    elif not service_ok:
                        logger.warning(
                            "RFork shutdown retained registered memory because seed service cleanup is incomplete."
                        )
                    elif not release_ok and not legacy_lease_pending:
                        logger.warning(
                            "RFork shutdown retained TransferEngine state because the source lease is unresolved. "
                            "No new release retries will be started; an in-flight request may still acknowledge. "
                            "Otherwise lease recovery depends on the planner's expiry/reclamation policy."
                        )
                    else:
                        logger.warning(
                            "RFork shutdown retained TransferEngine state because finalization did not complete."
                        )
                    return service_ok and release_ok and finalize_ok
            finally:
                with self._lock:
                    self._shutdown_in_progress = False

    def _register_weights(self, model, *, purpose: str) -> bool:
        if not self.transfer_backend.register_memory_region(model, purpose=purpose, identity=self.identity):
            return False
        self.planner.bind_structural_digest(_registered_structural_digest(self.transfer_backend))
        return True

    def _start_lease_renewal_locked(self, lease: SeedLease) -> None:
        if lease.lease_ttl_sec is None:
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._renew_seed_lease,
            args=(lease, stop_event),
            daemon=True,
            name="RForkLeaseRenewal",
        )
        self.lease_renew_stop_event = stop_event
        try:
            thread.start()
        except RuntimeError as exc:
            stop_event.set()
            raise RuntimeError("RFork seed lease renewal could not be started") from exc

    def _renew_seed_lease(self, lease: SeedLease, stop_event: threading.Event) -> None:
        assert lease.lease_ttl_sec is not None
        interval = min(
            max(lease.lease_ttl_sec / 3, LEASE_RENEW_MIN_INTERVAL_SEC),
            LEASE_RENEW_MAX_INTERVAL_SEC,
        )
        while not stop_event.wait(interval):
            self.planner.renew_seed_once(lease)

    def _ensure_seed_worker_locked(self) -> bool:
        if self.seed_lease is not None:
            self.lease_renew_stop_event.set()
        if self._shutdown_event.is_set() or self._lease_release_stopped:
            return False
        if self._seed_worker_thread is not None:
            return True
        self._seed_worker_thread = threading.Thread(target=self._run_seed_worker, daemon=True, name="RForkSeedWorker")
        try:
            self._seed_worker_thread.start()
            return True
        except RuntimeError:
            self._seed_worker_thread = None
            if self.seed_lease is not None:
                self._lease_release_stopped = True
            logger.exception("RFork could not start the seed worker.")
            return False

    def _run_seed_worker(self) -> None:
        try:
            self._retry_seed_lease_release()
            with self._lock:
                if self.seed_lease is not None or self._shutdown_event.is_set():
                    return
            with self._seed_lifecycle_lock:
                with self._lock:
                    if not self._seed_publication_pending or self._shutdown_event.is_set():
                        self._seed_worker_thread = None
                        return
                    self._seed_publication_pending = False
                try:
                    started = self._serve_seed()
                except Exception:
                    logger.exception("RFork background seed startup raised; inference can continue.")
                    started = False
                if not started:
                    self._cleanup_failed_seed_start()
                    logger.warning("RFork background seed startup failed; inference can continue.")
        finally:
            with self._lock:
                if self._seed_worker_thread is threading.current_thread():
                    self._seed_worker_thread = None

    def _retry_seed_lease_release(self) -> None:
        attempts = 0
        while not self._shutdown_event.is_set():
            with self._lock:
                if self.seed_lease is None:
                    return
                lease = self.seed_lease
            if attempts:
                interval = self.config.lease_release_retry_interval_sec
                if attempts >= self.config.lease_release_max_attempts:
                    interval = max(interval, LEASE_RELEASE_DEGRADED_RETRY_INTERVAL_SEC)
                if self._shutdown_event.wait(interval):
                    return
            with self._lock:
                if self._shutdown_event.is_set():
                    return
                self._lease_release_in_flight = True
            attempts += 1
            finished = self._release_lease_once(lease, "RFork background lease release raised.")
            if finished:
                if self.seed_lease is None and attempts > 1:
                    logger.info("RFork lease release recovered: lease=%s attempts=%d", lease_log_id(lease), attempts)
                return
            if attempts == 1 or attempts == self.config.lease_release_max_attempts:
                logger.warning("RFork lease release is temporarily unavailable; background retries continue.")

    def _release_lease_once(self, lease: SeedLease, exception_log: str) -> bool:
        """Release a lease outside the lock; return whether retries should stop."""
        result = LeaseReleaseResult.REJECTED
        try:
            result = self.planner.release_seed_once(lease)
        except Exception:
            logger.exception(exception_log)
        finally:
            with self._lock:
                self._lease_release_in_flight = False
                if result is LeaseReleaseResult.RELEASED:
                    self.seed_lease = None
                    if self.state is RForkLifecycleState.FINALIZED:
                        atexit.unregister(self._atexit_callback)
                elif result is LeaseReleaseResult.REJECTED:
                    self._lease_release_stopped = True
                    self._seed_publication_pending = False
                    logger.error("RFork lease release was rejected; the seed will not be published.")
        return result is not LeaseReleaseResult.RETRYABLE

    def _release_seed_locked(self) -> bool:
        """Schedule release without waiting; True only after an acknowledged release."""
        if self.seed_lease is None:
            return True
        self._ensure_seed_worker_locked()
        return False

    def _release_legacy_lease_on_shutdown(self) -> None:
        """Legacy allocations have no expiry; make one final bounded request unless one is in flight."""
        with self._lock:
            lease = self.seed_lease
            if lease is None or lease.lease_ttl_sec is not None:
                return
            if self._lease_release_in_flight:
                return
            self._lease_release_in_flight = True
        self._release_lease_once(lease, "RFork legacy lease release raised during shutdown.")

    def _seed_transfer_info(self) -> SeedTransferInfo:
        session_id = self.transfer_backend.transfer_session_id
        weights = self.transfer_backend.weight_manifest
        if not isinstance(session_id, str) or not session_id or not isinstance(weights, dict) or not weights:
            raise RuntimeError("RFork transfer metadata is unavailable after memory registration.")
        return SeedTransferInfo(session_id=session_id, weights=weights)

    def _serve_seed(self) -> bool:
        # Keep health/planner I/O outside _lock while shutdown waits on _seed_lifecycle_lock.
        with self._lock:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
        try:
            info = self._seed_transfer_info()
            # Reserve adjacent main/draft slots for every distributed worker.
            port = _resolve_seed_server_port(self.config, self.identity)
            if port > 65535:
                logger.warning(
                    "RFork seed port exceeds 65535: base=%d global_rank=%d model_kind=%s resolved_port=%d; "
                    "falling back to an OS-assigned port",
                    self.config.seed_port_base,
                    self.identity.global_rank,
                    "draft" if self.identity.is_draft_model else "main",
                    port,
                )
                port = 0

            handle = create_rfork_server(
                self.planner.seed_key,
                info,
                bind_host=self.config.seed_bind_host,
                port=port,
            )
            with self._lock:
                self.seed_server = handle
                if self._shutdown_event.is_set():
                    raise RuntimeError("shutdown requested during seed server startup")
            handle.start(self.config.seed_timeout_sec)
            initial_report = self.planner.report_seed_once(handle.port, seed_ip=self.config.seed_advertise_host)
            pending_report = not initial_report
            if pending_report and getattr(initial_report, "status", None) is not SeedReportStatus.RETRYABLE:
                raise RuntimeError(
                    "planner rejected the initial seed advertisement: "
                    f"{getattr(initial_report, 'reason', 'unknown error')}"
                )

            with self._lock:
                if self._shutdown_event.is_set():
                    raise RuntimeError("shutdown requested during seed advertisement")
                self.heartbeat_stop_event = threading.Event()
                self.heartbeat_thread = threading.Thread(
                    target=self._run_seed_heartbeat,
                    args=(handle, self.heartbeat_stop_event, int(pending_report)),
                    daemon=True,
                    name="RForkHeartbeat",
                )
                try:
                    self.heartbeat_thread.start()
                except RuntimeError:
                    self.heartbeat_thread = None
                    raise
                self.state = RForkLifecycleState.SERVING
            logger.debug(
                "RFork seed service started for global_rank=%s, port=%s",
                self.identity.global_rank,
                handle.port,
            )
            if pending_report:
                logger.warning(
                    "RFork seed service is running but planner advertisement is pending for seed_key=%s: %s; "
                    "background heartbeats will retry.",
                    self.planner.seed_key,
                    initial_report.reason,
                )
            return True
        except Exception as exc:
            with self._lock:
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork seed service startup failed for global_rank=%s: %s", self.identity.global_rank, exc)
            return False

    def _run_seed_heartbeat(
        self, handle: RForkSeedServerHandle, stop_event: threading.Event, report_failures: int = 0
    ) -> None:
        # Do not take _seed_lifecycle_lock: shutdown owns it while joining this thread; native cleanup stays elsewhere.
        unavailable = bool(report_failures)
        withdrawal_reason = "seed service failure"
        while not stop_event.wait(self.config.heartbeat_interval_sec):
            if not handle.is_alive:
                break
            try:
                reported = self.planner.report_seed_once(handle.port, seed_ip=self.config.seed_advertise_host)
            except Exception:
                logger.exception("RFork heartbeat raised; withdrawing the seed.")
                break
            # Server may exit while add_seed is in flight; revoke that advertisement even if the report succeeded.
            if not handle.is_alive:
                break
            if getattr(reported, "status", None) is SeedReportStatus.REJECTED:
                withdrawal_reason = (
                    f"planner rejected seed heartbeat for seed_key={self.planner.seed_key}: {reported.reason}"
                )
                break
            if not reported and not unavailable:
                logger.warning("RFork seed heartbeat is temporarily unavailable; background retries continue.")
                unavailable = True
            elif reported and unavailable:
                logger.info("RFork seed heartbeat recovered for seed_key=%s", self.planner.seed_key)
                unavailable = False
        else:
            return

        with self._lock:
            if stop_event.is_set() or self.seed_server is not handle:
                return
            stop_event.set()
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
        logger.error("RFork seed heartbeat stopped after %s; inference can continue.", withdrawal_reason)
        try:
            removed = self.planner.remove_seed()
        except Exception:
            logger.exception("RFork failed to withdraw the seed after a service failure.")
            removed = SeedRemovalResult.RETRYABLE
        if not removed:
            logger.warning("RFork seed removal remains pending; heartbeats stopped, retaining registered memory.")
        # Retain the handle and registrations until normal cleanup; removal alone does not prove native reads finished.

    def _cleanup_failed_seed_start(self) -> None:
        # The caller owns _seed_lifecycle_lock, but must not hold _lock here.
        service_ok = self._stop_seed_service()
        with self._lock:
            if service_ok:
                self._reset_transfer_locked()
            else:
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
                logger.warning("RFork seed cleanup is incomplete; registered memory remains pinned.")

    def _stop_seed_service(self, *, final: bool = False) -> bool:
        """Caller serializes seed lifecycle; join and removal run outside the state lock."""
        with self._lock:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            self.heartbeat_stop_event.set()
            heartbeat = self.heartbeat_thread
            server = self.seed_server
        if heartbeat is not None and heartbeat is not threading.current_thread():
            # Join add_seed before removal; Requests timeout bounds inactivity, not total duration.
            join_timeout_sec = max(
                float(self.config.seed_timeout_sec),
                2 * float(self.config.request_timeout_sec) + HEARTBEAT_STOP_GRACE_SEC,
            )
            heartbeat.join(timeout=join_timeout_sec)
            if heartbeat.is_alive():
                logger.warning("RFork heartbeat thread did not stop in time.")
                return False

        try:
            removed = self.planner.remove_seed()
        except Exception as exc:
            logger.warning("RFork planner seed removal raised: %s", exc)
            removed = SeedRemovalResult.RETRYABLE
        final_without_removal = removed is SeedRemovalResult.UNSUPPORTED and final
        if not removed and not final_without_removal:
            if removed in (SeedRemovalResult.UNSUPPORTED, SeedRemovalResult.UNCONFIRMED):
                # Stop admitting metadata requests, but previously issued addresses may still be in use.
                if server is not None:
                    try:
                        server.stop()
                    except Exception as exc:
                        logger.warning("RFork seed metadata service shutdown failed: %s", exc)
                logger.warning(
                    "RFork planner cannot confirm seed withdrawal; metadata service is stopping, "
                    "but published memory remains registered until safe cleanup or process exit."
                )
            else:
                logger.warning("RFork planner seed removal failed; keeping the local seed service available for retry.")
            return False
        if server is not None:
            try:
                if not server.stop():
                    return False
            except Exception as exc:
                logger.warning("RFork seed server shutdown failed: %s", exc)
                return False
        with self._lock:
            self.heartbeat_thread = None
            self.seed_server = None
        if final_without_removal:
            logger.info(
                "RFork planner does not support seed removal; the metadata service stopped and "
                "local transfer resources are released for process exit."
            )
        return True

    def _reset_transfer_locked(self) -> bool:
        if self.seed_server is not None:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork refuses to unregister memory while the seed server is owned.")
            return False
        try:
            reset = self.transfer_backend.unregister_memory_region()
        except Exception as exc:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork memory unregistration raised: %s", exc)
            return False
        if not reset:
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork memory remains registered; retaining transfer state for retry.")
            return False
        self.state = RForkLifecycleState.INITIALIZED
        return True
