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
from vllm_ascend.model_loader.rfork.tensor_layout import (
    build_structural_digest,
    collect_transferable_tensors,
)
from vllm_ascend.model_loader.rfork.transfer_backend import RForkTransferBackend
from vllm_ascend.model_loader.rfork.types import (
    LeaseReleaseResult,
    RForkFallbackCleanupResult,
    RForkIdentity,
    RForkLifecycleState,
    RForkSeedServiceStartResult,
    SeedLease,
    SeedReportStatus,
    SeedTransferInfo,
)

HEARTBEAT_STOP_GRACE_SEC = 1.0
HEARTBEAT_LOG_EVERY_N = 4
HEARTBEAT_FAILURE_LOG_EVERY_N = 20
LEASE_RENEW_MIN_INTERVAL_SEC = 0.1
LEASE_RENEW_MAX_INTERVAL_SEC = 30.0
LEASE_RELEASE_DEGRADED_RETRY_INTERVAL_SEC = 60.0
LEASE_RELEASE_DEGRADED_LOG_EVERY_N = 10
RFORK_SEED_PORT_SLOTS_PER_RANK = 2


def _resolve_seed_server_port(config: RForkConfig, identity: RForkIdentity) -> int:
    if config.seed_port_base == 0:
        return 0
    model_slot = 1 if identity.is_draft_model else 0
    return config.seed_port_base + identity.global_rank * RFORK_SEED_PORT_SLOTS_PER_RANK + model_slot


def _compute_structural_digest(model, processed_layout: bool) -> str:
    return build_structural_digest(collect_transferable_tensors(model, processed_layout))


class RForkSession:
    """Sole owner of one worker process's RFork runtime resources.

    Lock ordering: _seed_lifecycle_lock must be acquired before _lock when both are needed.
    _seed_lifecycle_lock serializes seed service start/stop across planner I/O.
    _lock protects session state, lease, and server handle.
    """

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
        self.transfer_backend = RForkTransferBackend(tp_rank=identity.tp_rank)
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
        self._source_transfer_session_id: str | None = None
        self._deferred_seed_start: tuple[Any, bool, list[tuple[int, int]] | None] | None = None
        self._lock = threading.RLock()
        # Acquire before _lock; release responses must not wait on removal I/O.
        self._seed_lifecycle_lock = threading.RLock()
        self._shutdown_in_progress = False
        self._atexit_callback = self.shutdown
        atexit.register(self._atexit_callback)

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
            self._source_transfer_session_id = None
            started_at = time.monotonic()
            if not self.transfer_backend.register_memory_region(model, processed_layout, exclude_blocks):
                return False
            self.planner.bind_structural_digest(_compute_structural_digest(model, processed_layout))
            self._registration_elapsed = time.monotonic() - started_at
            self.state = RForkLifecycleState.REGISTERED
            # Seed misses are silent 404s; the bound key and digest make them debuggable.
            log_identity = logger.info if self.identity.tp_rank == 0 else logger.debug
            log_identity(
                "RFork %s seed identity bound: seed_key=%s, structural_digest=%s",
                "draft" if self.identity.is_draft_model else "main",
                self.planner.seed_key,
                self.planner.structural_digest,
            )
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
        """Signal the lease renewal thread to stop without waiting under _lock."""
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
            self._source_transfer_session_id = seed_info.session_id
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

    def log_transferred_model_layout(self, model, processed_layout: bool) -> None:
        """Log the final receiver layout without affecting transfer state."""
        with self._lock:
            peer_session_id = self._source_transfer_session_id
        self.transfer_backend.log_model_layout_summary(
            model,
            processed_layout,
            stage=("receiver_after_transfer_finalize" if processed_layout else "receiver_after_post_load"),
            peer_session_id=peer_session_id,
        )

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
                retry_interval = self.config.lease_release_retry_interval_sec
                if self._lease_release_attempts >= self.config.lease_release_max_attempts:
                    retry_interval = max(retry_interval, LEASE_RELEASE_DEGRADED_RETRY_INTERVAL_SEC)
                if self._lease_release_attempts and self.lease_release_stop_event.wait(retry_interval):
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
            if self._lease_release_attempts > 1:
                logger.info(
                    "RFork planner lease release recovered: lease=%s attempts=%d",
                    lease_log_id(lease),
                    self._lease_release_attempts,
                )
            self.seed_lease = None
            self._lease_acquired_at = None
            if self.state is RForkLifecycleState.LEASED:
                self.state = RForkLifecycleState.REGISTERED
            return True
        if result is LeaseReleaseResult.REJECTED:
            self._lease_release_exhausted = True
            self._deferred_seed_start = None
            logger.error(
                "RFork lease release stopped: lease=%s attempts=%d; lease remains unresolved, "
                "worker will not advertise a seed. Model loading/inference may continue.",
                lease_log_id(lease),
                self._lease_release_attempts,
            )
            return True
        if (
            self._lease_release_attempts == 1
            or self._lease_release_attempts == self.config.lease_release_max_attempts
            or (
                self._lease_release_attempts > self.config.lease_release_max_attempts
                and (self._lease_release_attempts - self.config.lease_release_max_attempts)
                % LEASE_RELEASE_DEGRADED_LOG_EVERY_N
                == 0
            )
        ):
            logger.warning(
                "RFork planner lease release is temporarily unavailable: lease=%s attempts=%d; "
                "background retries continue.",
                lease_log_id(lease),
                self._lease_release_attempts,
            )
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
        if self.identity.is_draft_model:
            # A draft's topology is not final when the loader returns: the proposer
            # then shares embed_tokens / lm_head / topk buffers with its target, and
            # a DSpark draft rotates fc in place. Weights registered here would be
            # rebound or rewritten afterwards, so a peer reading them could see
            # pre-sharing addresses or a mix of rotated and unrotated bytes. Drafts
            # therefore stay receive-only until registration can follow sharing.
            logger.debug("RFork draft models do not advertise a seed; the transferred model serves inference only.")
            return RForkSeedServiceStartResult.FAILED
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
                        if registered:
                            try:
                                self.planner.bind_structural_digest(_compute_structural_digest(model, processed_layout))
                            except RuntimeError as exc:
                                # The structure drifted from destination registration; a seed
                                # advertised under the stale key could never pass manifest checks.
                                logger.error(
                                    "RFork refuses to advertise a seed whose structure changed "
                                    "after destination registration: %s. Inference can continue.",
                                    exc,
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
            # Production check of the pre/post post-load symmetry invariant: the
            # digest recomputed at advertisement must equal the one bound at
            # destination registration.
            self.planner.verify_structural_digest(_compute_structural_digest(model, processed_layout))
            # Reserve adjacent main/draft slots for every distributed worker.
            port = _resolve_seed_server_port(self.config, self.identity)
            if port > 0:
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

            handle = start_rfork_server(
                self.planner.seed_key,
                info,
                health_timeout_sec=self.config.seed_timeout_sec,
                bind_host=self.config.seed_bind_host,
                port=port,
                fallback_to_dynamic_port=port > 0,
            )
            with self._lock:
                self.seed_server = handle
                if self.lease_release_stop_event.is_set():
                    raise RuntimeError("shutdown requested during seed server startup")
            if not handle.is_alive:
                raise RuntimeError("seed HTTP server exited before advertisement")
            initial_report = self.planner.report_seed_once(handle.port, seed_ip=self.config.seed_advertise_host)
            pending_report = not initial_report
            if pending_report and getattr(initial_report, "status", None) is not SeedReportStatus.RETRYABLE:
                raise RuntimeError(
                    "planner rejected the initial seed advertisement: "
                    f"{getattr(initial_report, 'reason', 'unknown error')}"
                )

            with self._lock:
                if self.lease_release_stop_event.is_set():
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
                if isinstance(exc, RForkSeedServerStartupError):
                    self.seed_server = exc.handle
                self.state = RForkLifecycleState.CLEANUP_REQUIRED
            logger.warning("RFork seed service startup failed for global_rank=%s: %s", self.identity.global_rank, exc)
            return False

    def _run_seed_heartbeat(
        self, handle: RForkSeedServerHandle, stop_event: threading.Event, report_failures: int = 0
    ) -> None:
        # Do not take _seed_lifecycle_lock: shutdown owns it while joining this thread; native cleanup stays elsewhere.
        heartbeat_index = 0
        withdrawal_reason = "seed service failure"
        while not stop_event.wait(self.config.heartbeat_interval_sec):
            if not handle.is_alive:
                break
            heartbeat_index += 1
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
            if not reported:
                report_failures += 1
                if report_failures == 1 or report_failures % HEARTBEAT_FAILURE_LOG_EVERY_N == 0:
                    logger.warning(
                        "RFork planner seed heartbeat is temporarily unavailable for seed_key=%s: %s "
                        "(consecutive failures=%d); background retries continue.",
                        self.planner.seed_key,
                        getattr(reported, "reason", "unknown error"),
                        report_failures,
                    )
            elif heartbeat_index % HEARTBEAT_LOG_EVERY_N == 0:
                logger.debug("RFork heartbeat accepted for seed_key=%s", self.planner.seed_key)
            if reported and report_failures:
                logger.info(
                    "RFork planner seed heartbeat recovered for seed_key=%s after %d failed reports.",
                    self.planner.seed_key,
                    report_failures,
                )
                report_failures = 0
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
            removed = False
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
                if self._shutdown_in_progress:
                    logger.debug("RFork shutdown is already in progress; skipping a reentrant cleanup request.")
                    return False
                self._shutdown_in_progress = True
                self._deferred_seed_start = None
            try:
                service_ok = self._stop_seed_service()
                with self._lock:
                    release_ok = self.seed_lease is None
                    finalize_ok = (
                        self.transfer_backend.finalize_transfer_engine() if service_ok and release_ok else False
                    )
                    if finalize_ok:
                        self.state = RForkLifecycleState.FINALIZED
                        atexit.unregister(self._atexit_callback)
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
            finally:
                with self._lock:
                    self._shutdown_in_progress = False
