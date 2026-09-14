# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import atexit
import threading
import time
from typing import Any

from vllm.logger import logger

from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.planner_client import RForkPlannerClient, lease_log_id
from vllm_ascend.model_loader.rfork.seed_client import build_seed_url, fetch_seed_transfer_info
from vllm_ascend.model_loader.rfork.seed_server import (
    RForkSeedServerHandle,
    RForkSeedServerStartupError,
    start_rfork_server,
)
from vllm_ascend.model_loader.rfork.transfer_backend import RForkTransferBackend
from vllm_ascend.model_loader.rfork.types import (
    LeaseReleaseResult,
    RForkFallbackCleanupResult,
    RForkIdentity,
    RForkLifecycleState,
    RForkSeedServiceStartResult,
    SeedLease,
    SeedTransferInfo,
)

HEARTBEAT_STOP_GRACE_SEC = 1.0
LEASE_RENEW_MIN_INTERVAL_SEC = 0.1
LEASE_RENEW_MAX_INTERVAL_SEC = 30.0


class RForkSession:
    """Sole owner of one worker process's RFork runtime resources."""

    def __init__(self, config: RForkConfig, identity: RForkIdentity) -> None:
        if not config.planner_url:
            raise ValueError(
                "rfork_scheduler_url is required; configure it with model_loader_extra_config or RFORK_SCHEDULER_URL"
            )
        if not config.model_url or not config.model_deploy_strategy_name:
            raise ValueError("RFork requires non-empty model_url and model_deploy_strategy_name")

        self.config = config
        self.identity = identity
        self.planner = RForkPlannerClient(config, identity)
        self.transfer_backend = RForkTransferBackend()
        self.state = RForkLifecycleState.INITIALIZED
        self.seed_lease: SeedLease | None = None
        self.seed_server: RForkSeedServerHandle | None = None
        self.heartbeat_thread: threading.Thread | None = None
        self.heartbeat_stop_event = threading.Event()
        self.lease_release_thread: threading.Thread | None = None
        self.lease_release_stop_event = threading.Event()
        self.lease_renew_thread: threading.Thread | None = None
        self.lease_renew_stop_event = threading.Event()
        self._lease_release_attempts = 0
        self._lease_release_exhausted = False
        self._lease_acquired_at: float | None = None
        self._registration_elapsed = 0.0
        self._deferred_seed_start: tuple[Any, bool, list[tuple[int, int]] | None] | None = None
        self._lock = threading.RLock()
        # Acquire before _lock; release responses must not wait on removal I/O.
        self._seed_lifecycle_lock = threading.RLock()
        atexit.register(self.shutdown)

    def register_destination(
        self, model, processed_layout: bool, exclude_blocks: list[tuple[int, int]] | None = None
    ) -> bool:
        """Prepare local buffers on the NPU caller thread before acquiring a lease."""
        with self._lock:
            if (
                self.state is not RForkLifecycleState.INITIALIZED
                or self.seed_lease is not None
                or self.lease_release_stop_event.is_set()
            ):
                return False
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
            started_at = time.monotonic()
            if not self.transfer_backend.register_memory_region(model, processed_layout, exclude_blocks):
                return False
            self._registration_elapsed = time.monotonic() - started_at
            self.state = RForkLifecycleState.REGISTERED
            return True

    def acquire_seed(self) -> bool:
        with self._lock:
            if self.lease_release_stop_event.is_set():
                return False
            if self.lease_release_thread is not None and self.lease_release_thread.is_alive():
                return False
            # Retry unresolved leases only for release; acquisition also needs registered buffers.
            if self.seed_lease is not None:
                self._release_seed_locked()
                return False
            if self.state is not RForkLifecycleState.REGISTERED:
                logger.error("RFork seed acquisition requires registered buffers; state=%s", self.state.name)
                return False
            acquisition_started = time.monotonic()
            self.seed_lease = self.planner.acquire_seed()
            if self.seed_lease is None:
                return False
            self._lease_acquired_at = acquisition_started
            self._lease_release_attempts = 0
            self._lease_release_exhausted = False
            if not self._start_lease_renewal_locked(self.seed_lease):
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
                self._release_seed_locked()
                return False
            logger.debug(
                "RFork lease acquired: lease=%s global_rank=%s request_elapsed=%.3fs",
                lease_log_id(self.seed_lease),
                self.identity.global_rank,
                time.monotonic() - acquisition_started,
            )
            self.state = RForkLifecycleState.LEASED
            return True

    def _start_lease_renewal_locked(self, lease: SeedLease) -> bool:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._renew_seed_lease,
            args=(lease, stop_event),
            daemon=True,
            name="RForkLeaseRenewal",
        )
        self.lease_renew_stop_event = stop_event
        self.lease_renew_thread = thread
        try:
            thread.start()
        except RuntimeError:
            self.lease_renew_thread = None
            stop_event.set()
            logger.exception("RFork could not start the lease renewal worker.")
            return False
        return True

    def _renew_seed_lease(self, lease: SeedLease, stop_event: threading.Event) -> None:
        interval = min(
            max(float(lease.lease_ttl_sec) / 3, LEASE_RENEW_MIN_INTERVAL_SEC),
            LEASE_RENEW_MAX_INTERVAL_SEC,
        )
        try:
            while not stop_event.wait(interval):
                self.planner.renew_seed_once(lease)
        finally:
            with self._lock:
                if self.lease_renew_thread is threading.current_thread():
                    self.lease_renew_thread = None

    def _stop_lease_renewal_locked(self) -> None:
        self.lease_renew_stop_event.set()

    def can_reuse_shared_weights(self, model, processed_layout: bool, exclude_blocks: list[tuple[int, int]]) -> bool:
        with self._lock:
            if (
                self.state is not RForkLifecycleState.INITIALIZED
                or self.seed_lease is not None
                or self.lease_release_stop_event.is_set()
            ):
                return False
            return self.transfer_backend.can_reuse_shared_weights(model, processed_layout, exclude_blocks)

    def transfer_from_seed(
        self,
        model,
        processed_layout: bool,
    ) -> bool:
        # Hold _lock through metadata/RDMA so cleanup cannot unregister buffers mid-transfer.
        with self._lock:
            if self.state is not RForkLifecycleState.LEASED or self.seed_lease is None:
                logger.error("RFork transfer requires an acquired seed lease.")
                return False
            # Metadata/read failures require cleanup because destination registration already succeeded.
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
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
            if not self.transfer_backend.read_weights_from_seed(
                model=model,
                seed_info=seed_info,
                processed_layout=processed_layout,
            ):
                return False
            self.state = RForkLifecycleState.TRANSFERRED
            logger.debug(
                "RFork transfer stages: lease=%s global_rank=%s registration=%.3fs metadata=%.3fs read=%.3fs",
                lease_log_id(self.seed_lease),
                self.identity.global_rank,
                self._registration_elapsed,
                metadata_elapsed,
                time.monotonic() - read_started,
            )
            # Lease bookkeeping must never put planner network latency on the startup thread.
            self._ensure_lease_release_retry_locked()
            return True

    def _ensure_lease_release_retry_locked(self) -> None:
        self._stop_lease_renewal_locked()
        if (
            self.seed_lease is None
            or self._lease_release_exhausted
            or self.lease_release_stop_event.is_set()
            or (self.lease_release_thread is not None and self.lease_release_thread.is_alive())
        ):
            return
        self.lease_release_thread = threading.Thread(
            target=self._retry_seed_lease_release,
            daemon=True,
            name="RForkLeaseRelease",
        )
        try:
            self.lease_release_thread.start()
        except RuntimeError:
            self.lease_release_thread = None
            self._lease_release_exhausted = True
            logger.exception("RFork could not start lease release worker; retaining unresolved lease.")

    def _retry_seed_lease_release(self) -> None:
        try:
            while not self.lease_release_stop_event.is_set():
                # Wait only between attempts, never before the initial release.
                if self._lease_release_attempts and self.lease_release_stop_event.wait(
                    self.config.lease_release_retry_interval_sec
                ):
                    return
                with self._lock:
                    if self.state is RForkLifecycleState.FINALIZED or self.seed_lease is None:
                        return
                    lease = self.seed_lease
                    self._lease_release_attempts += 1
                # Perform release HTTP outside _lock, then reacquire it to update state.
                try:
                    result = self.planner.release_seed_once(lease)
                except Exception:
                    logger.exception("RFork background lease release raised; retaining unresolved lease.")
                    result = LeaseReleaseResult.REJECTED
                with self._lock:
                    if self.seed_lease is not lease:
                        return
                    finished = self._record_lease_release_locked(lease, result)
                if finished:
                    self._promote_deferred_seed()
                    return

        finally:
            with self._lock:
                if self.lease_release_thread is threading.current_thread():
                    self.lease_release_thread = None

    def _record_lease_release_locked(self, lease: SeedLease, result: LeaseReleaseResult) -> bool:
        """Apply one release response; True stops retries."""
        acquired_at = self._lease_acquired_at
        logger.debug(
            "RFork lease release outcome: lease=%s attempt=%d/%d result=%s held_elapsed=%.3fs",
            lease_log_id(lease),
            self._lease_release_attempts,
            self.config.lease_release_max_attempts,
            result.name,
            time.monotonic() - acquired_at if acquired_at is not None else 0.0,
        )
        if result is LeaseReleaseResult.RELEASED:
            self.seed_lease = None
            self._lease_acquired_at = None
            if self.state is RForkLifecycleState.LEASED:
                self.state = RForkLifecycleState.REGISTERED
            return True
        if (
            result is LeaseReleaseResult.REJECTED
            or self._lease_release_attempts >= self.config.lease_release_max_attempts
        ):
            self._lease_release_exhausted = True
            self._deferred_seed_start = None
            logger.error(
                "RFork lease release stopped: lease=%s attempts=%d; lease remains unresolved, "
                "worker will not advertise a seed. Model loading/inference may continue.",
                lease_log_id(lease),
                self._lease_release_attempts,
            )
            return True
        return False

    def _release_seed_locked(self) -> bool:
        """Schedule release without waiting; True only after an acknowledged release."""
        if self.seed_lease is None:
            return True
        self._ensure_lease_release_retry_locked()
        return False

    def _promote_deferred_seed(self) -> None:
        with self._seed_lifecycle_lock:
            with self._lock:
                if (
                    self._deferred_seed_start is None
                    or self.seed_lease is not None
                    or self.state is RForkLifecycleState.FINALIZED
                    or self.lease_release_stop_event.is_set()
                ):
                    return
                model, processed_layout, exclude_blocks = self._deferred_seed_start
                self._deferred_seed_start = None
            try:
                promoted = self._start_seed_service(model, processed_layout, exclude_blocks)
            except Exception:
                logger.exception(
                    "RFork deferred seed promotion raised after the source lease was released; "
                    "the model remains available for inference."
                )
                promoted = False
            if not promoted:
                self._cleanup_failed_seed_start()
                logger.warning(
                    "RFork deferred seed promotion failed after the source lease was released; "
                    "the model remains available for inference."
                )

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
        if self.state is not RForkLifecycleState.FINALIZED:
            self.state = RForkLifecycleState.INITIALIZED
        return True

    def prepare_for_fallback(self) -> RForkFallbackCleanupResult:
        """Return to an initialized, reusable session without finalizing TransferEngine."""
        with self._seed_lifecycle_lock:
            with self._lock:
                if self.state is RForkLifecycleState.FINALIZED:
                    logger.error("RFork cannot prepare a finalized session for fallback.")
                    return RForkFallbackCleanupResult(False, False, False)
                self._deferred_seed_start = None
            service_ok = self._stop_seed_service()
            with self._lock:
                release_ok = self._release_seed_locked()
                reset_ok = self._reset_transfer_locked() if service_ok else False
                return RForkFallbackCleanupResult(service_ok, release_ok, reset_ok)

    def _seed_transfer_info(self) -> SeedTransferInfo:
        session_id = self.transfer_backend.transfer_session_id
        weights = self.transfer_backend.weight_manifest
        if not isinstance(session_id, str) or not session_id or not isinstance(weights, dict) or not weights:
            raise RuntimeError("RFork transfer metadata is unavailable after memory registration.")
        return SeedTransferInfo(
            session_id=session_id,
            weights=weights,
            formats=dict(self.transfer_backend.weight_formats) if self.transfer_backend.weight_formats else None,
        )

    def start_seed_service(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> RForkSeedServiceStartResult:
        with self._seed_lifecycle_lock:
            with self._lock:
                if self.lease_release_stop_event.is_set():
                    return RForkSeedServiceStartResult.FAILED
                if self.state is RForkLifecycleState.SERVING:
                    return RForkSeedServiceStartResult.STARTED
                if self.seed_lease is not None and self._lease_release_exhausted:
                    return RForkSeedServiceStartResult.FAILED
                if self.state not in (
                    RForkLifecycleState.INITIALIZED,
                    RForkLifecycleState.LEASED,
                    RForkLifecycleState.TRANSFERRED,
                    RForkLifecycleState.READY,
                ):
                    logger.error("RFork seed service requires a complete model; state=%s", self.state.name)
                    return RForkSeedServiceStartResult.FAILED
                if self.state is not RForkLifecycleState.READY:
                    # Refresh checkpoint-layout registration after post-load storage replacement.
                    if self.state is not RForkLifecycleState.TRANSFERRED or not processed_layout:
                        self.state = RForkLifecycleState.CLEANUP_REQUIRED
                        try:
                            registered = self.transfer_backend.register_memory_region(
                                model, processed_layout, exclude_blocks
                            )
                        except Exception:
                            logger.exception(
                                "RFork seed memory registration raised; cleaning up before continuing inference."
                            )
                            registered = False
                        if not registered:
                            self._reset_transfer_locked()
                            return RForkSeedServiceStartResult.FAILED
                    self.state = RForkLifecycleState.READY
                if self.seed_lease is not None:
                    if self._lease_release_exhausted or self.lease_release_stop_event.is_set():
                        return RForkSeedServiceStartResult.FAILED
                    self._deferred_seed_start = (model, processed_layout, exclude_blocks)
                    self._ensure_lease_release_retry_locked()
                    logger.debug(
                        "RFork seed promotion is deferred until the source seed lease is released; "
                        "the transferred model remains available for inference."
                    )
                    return RForkSeedServiceStartResult.DEFERRED
            started = self._start_seed_service(model, processed_layout, exclude_blocks)
            if not started:
                self._cleanup_failed_seed_start()
            return RForkSeedServiceStartResult.STARTED if started else RForkSeedServiceStartResult.FAILED

    def _start_seed_service(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> bool:
        # Keep health/planner I/O outside _lock; transitional state blocks transfer and registration.
        with self._lock:
            if self.state is RForkLifecycleState.SERVING:
                return True
            if self.state is not RForkLifecycleState.READY:
                logger.error("RFork seed service cannot start from state=%s", self.state.name)
                return False
            if self.seed_lease is not None or self.lease_release_stop_event.is_set():
                logger.error("RFork seed service cannot start with an unresolved lease or during shutdown.")
                return False
            self.state = RForkLifecycleState.CLEANUP_REQUIRED
        try:
            info = self._seed_transfer_info()
            handle = start_rfork_server(
                self.planner.seed_key,
                info,
                health_timeout_sec=self.config.seed_timeout_sec,
                bind_host=self.config.seed_bind_host,
            )
            with self._lock:
                self.seed_server = handle
                if self.lease_release_stop_event.is_set():
                    raise RuntimeError("shutdown requested during seed server startup")
            if not self.planner.report_seed_once(handle.port, seed_ip=self.config.seed_advertise_host):
                raise RuntimeError("planner rejected the initial seed advertisement")

            with self._lock:
                if self.lease_release_stop_event.is_set():
                    raise RuntimeError("shutdown requested during seed advertisement")
                self.heartbeat_stop_event = threading.Event()
                self.heartbeat_thread = threading.Thread(
                    target=self.planner.run_seed_heartbeat,
                    args=(handle.port,),
                    kwargs={
                        "sleep_interval": self.config.heartbeat_interval_sec,
                        "stop_event": self.heartbeat_stop_event,
                        "seed_ip": self.config.seed_advertise_host,
                        "initial_delay": True,
                    },
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
            return True
        except Exception as exc:
            with self._lock:
                if isinstance(exc, RForkSeedServerStartupError):
                    self.seed_server = exc.handle
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork seed service startup failed for global_rank=%s: %s", self.identity.global_rank, exc)
            return False

    def _cleanup_failed_seed_start(self) -> None:
        # The caller owns _seed_lifecycle_lock, but must not hold _lock here.
        service_ok = self._stop_seed_service()
        with self._lock:
            if service_ok:
                self._reset_transfer_locked()
            else:
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
                logger.warning("RFork seed cleanup is incomplete; registered memory remains pinned.")

    def _stop_seed_service(self) -> bool:
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
            removed = False
        if not removed:
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
        return True

    def shutdown(self) -> bool:
        # Avoid release I/O; existing attempts may finish, otherwise planner expiry reclaims the lease.
        self.lease_release_stop_event.set()
        self.lease_renew_stop_event.set()
        with self._seed_lifecycle_lock:
            with self._lock:
                if self.state is RForkLifecycleState.FINALIZED:
                    return True
                self._deferred_seed_start = None
            service_ok = self._stop_seed_service()
            with self._lock:
                release_ok = self.seed_lease is None
                finalize_ok = self.transfer_backend.finalize_transfer_engine() if service_ok and release_ok else False
                if finalize_ok:
                    self.state = RForkLifecycleState.FINALIZED
                elif not service_ok:
                    logger.warning(
                        "RFork shutdown retained registered memory because seed service cleanup is incomplete."
                    )
                elif not release_ok:
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
