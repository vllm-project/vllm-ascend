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
from vllm_ascend.model_loader.rfork.seed_server import RForkSeedServerHandle, start_rfork_server
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
        self._lease_release_attempts = 0
        self._lease_release_exhausted = False
        self._lease_acquired_at: float | None = None
        self._deferred_seed_start: tuple[Any, bool, list[tuple[int, int]] | None] | None = None
        self._lock = threading.RLock()
        atexit.register(self.shutdown)

    def acquire_seed(self) -> bool:
        with self._lock:
            if self.lease_release_stop_event.is_set():
                return False
            if self.lease_release_thread is not None and self.lease_release_thread.is_alive():
                return False
            # Retain a failed-release lease for retry even if memory reset returned the session to initialized.
            if self.seed_lease is not None:
                if not self._release_seed_locked() or self.seed_lease is not None:
                    return False
            if self.state is not RForkLifecycleState.INITIALIZED:
                logger.error("RFork seed acquisition requires an initialized session; state=%s", self.state.name)
                return False
            acquisition_started = time.monotonic()
            self.seed_lease = self.planner.acquire_seed()
            if self.seed_lease is None:
                return False
            self._lease_acquired_at = acquisition_started
            self._lease_release_attempts = 0
            self._lease_release_exhausted = False
            logger.info(
                "RFork lease acquired: lease=%s global_rank=%s request_elapsed=%.3fs",
                lease_log_id(self.seed_lease),
                self.identity.global_rank,
                time.monotonic() - acquisition_started,
            )
            self.state = RForkLifecycleState.LEASED
            return True

    def transfer_from_seed(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> bool:
        with self._lock:
            if self.state is not RForkLifecycleState.LEASED or self.seed_lease is None:
                logger.error("RFork transfer requires an acquired seed lease.")
                return False
            registration_started = time.monotonic()
            if not self.transfer_backend.register_memory_region(model, processed_layout, exclude_blocks):
                return False
            registration_elapsed = time.monotonic() - registration_started
            self.state = RForkLifecycleState.REGISTERED
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
            logger.info(
                "RFork transfer stages: lease=%s global_rank=%s registration=%.3fs metadata=%.3fs read=%.3fs",
                lease_log_id(self.seed_lease),
                self.identity.global_rank,
                registration_elapsed,
                metadata_elapsed,
                time.monotonic() - read_started,
            )
            # Lease bookkeeping must never put planner network latency on the startup thread.
            self._ensure_lease_release_retry_locked()
            return True

    def _ensure_lease_release_retry_locked(self) -> None:
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
                with self._lock:
                    if self.state is RForkLifecycleState.FINALIZED or self.seed_lease is None:
                        return
                    lease = self.seed_lease
                    self._lease_release_attempts += 1
                    attempt = self._lease_release_attempts
                    acquired_at = self._lease_acquired_at
                # Only this worker sends releases. No network I/O under the session lock.
                try:
                    result = self.planner.release_seed_once(lease)
                except Exception:
                    logger.exception("RFork background lease release raised; retaining unresolved lease.")
                    result = LeaseReleaseResult.REJECTED
                with self._lock:
                    if self.seed_lease is not lease:
                        return
                    logger.info(
                        "RFork lease release outcome: lease=%s attempt=%d/%d result=%s held_elapsed=%.3fs",
                        lease_log_id(lease),
                        attempt,
                        self.config.lease_release_max_attempts,
                        result.name,
                        time.monotonic() - acquired_at if acquired_at is not None else 0.0,
                    )
                    if result is LeaseReleaseResult.RELEASED:
                        self.seed_lease = None
                        self._lease_acquired_at = None
                        if self.state is RForkLifecycleState.LEASED:
                            self.state = RForkLifecycleState.INITIALIZED
                        self._promote_deferred_seed_locked()
                        return
                    if result is LeaseReleaseResult.REJECTED or attempt >= self.config.lease_release_max_attempts:
                        self._lease_release_exhausted = True
                        self._deferred_seed_start = None
                        logger.error(
                            "RFork lease release stopped: lease=%s attempts=%d; lease remains unresolved, "
                            "worker will not advertise a seed. Model loading/inference may continue.",
                            lease_log_id(lease),
                            attempt,
                        )
                        return
                self.lease_release_stop_event.wait(self.config.lease_release_retry_interval_sec)
        finally:
            with self._lock:
                if self.lease_release_thread is threading.current_thread():
                    self.lease_release_thread = None

    def _release_seed_locked(self) -> bool:
        """Schedule release without waiting; True only after an acknowledged release."""
        if self.seed_lease is None:
            return True
        self._ensure_lease_release_retry_locked()
        return False

    def _promote_deferred_seed_locked(self) -> None:
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
            promoted = self._start_seed_service_locked(model, processed_layout, exclude_blocks)
        except Exception:
            logger.exception(
                "RFork deferred seed promotion raised after the source lease was released; "
                "the model remains available for inference."
            )
            return
        if not promoted:
            logger.warning(
                "RFork deferred seed promotion failed after the source lease was released; "
                "the model remains available for inference."
            )

    def release_seed(self) -> bool:
        with self._lock:
            return self._release_seed_locked()

    def _reset_transfer_locked(self) -> bool:
        if self.seed_server is not None:
            logger.warning("RFork refuses to unregister memory while the seed server is owned.")
            return False
        try:
            reset = self.transfer_backend.unregister_memory_region()
        except Exception as exc:
            logger.warning("RFork memory unregistration raised: %s", exc)
            return False
        if not reset:
            logger.warning("RFork memory remains registered; retaining transfer state for retry.")
            return False
        if self.state is not RForkLifecycleState.FINALIZED:
            self.state = RForkLifecycleState.LEASED if self.seed_lease is not None else RForkLifecycleState.INITIALIZED
        return True

    def prepare_for_fallback(self) -> RForkFallbackCleanupResult:
        """Return to an initialized, reusable session without finalizing TransferEngine."""
        with self._lock:
            if self.state is RForkLifecycleState.FINALIZED:
                logger.error("RFork cannot prepare a finalized session for fallback.")
                return RForkFallbackCleanupResult(False, False, False)
            self._deferred_seed_start = None
            service_ok = self._stop_seed_service_locked()
            release_ok = self._release_seed_locked()
            reset_ok = self._reset_transfer_locked() if service_ok else False
            if not release_ok:
                self._ensure_lease_release_retry_locked()
            return RForkFallbackCleanupResult(service_ok, release_ok, reset_ok)

    def _seed_transfer_info(self) -> SeedTransferInfo:
        session_id = self.transfer_backend.transfer_session_id
        weights = self.transfer_backend.weight_manifest
        shapes = self.transfer_backend.weight_shapes
        if not isinstance(session_id, str) or not session_id or not isinstance(weights, dict):
            raise RuntimeError("RFork transfer metadata is unavailable after memory registration.")
        return SeedTransferInfo(session_id=session_id, weights=weights, shapes=shapes)

    def start_seed_service(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> RForkSeedServiceStartResult:
        with self._lock:
            if self.lease_release_stop_event.is_set():
                return RForkSeedServiceStartResult.FAILED
            if self.state is RForkLifecycleState.SERVING:
                return RForkSeedServiceStartResult.STARTED
            if self.seed_lease is not None:
                if self._lease_release_exhausted or self.lease_release_stop_event.is_set():
                    return RForkSeedServiceStartResult.FAILED
                if self.state not in (RForkLifecycleState.LEASED, RForkLifecycleState.REGISTERED):
                    logger.error("RFork seed promotion cannot be deferred from state=%s", self.state.name)
                    return RForkSeedServiceStartResult.FAILED
                # Prepare fallback model memory on the caller's NPU thread. The release
                # worker only publishes ready metadata; it must not register on its default device.
                if self.state is RForkLifecycleState.LEASED:
                    if not self.transfer_backend.register_memory_region(model, processed_layout, exclude_blocks):
                        return RForkSeedServiceStartResult.FAILED
                    self.state = RForkLifecycleState.REGISTERED
                self._deferred_seed_start = (model, processed_layout, exclude_blocks)
                self._ensure_lease_release_retry_locked()
                logger.warning(
                    "RFork seed promotion is deferred until the source seed lease is released; "
                    "the transferred model remains available for inference."
                )
                return RForkSeedServiceStartResult.DEFERRED
            if self.state not in (RForkLifecycleState.INITIALIZED, RForkLifecycleState.REGISTERED):
                logger.error("RFork seed service cannot start from state=%s", self.state.name)
                return RForkSeedServiceStartResult.FAILED
            started = self._start_seed_service_locked(model, processed_layout, exclude_blocks)
            return RForkSeedServiceStartResult.STARTED if started else RForkSeedServiceStartResult.FAILED

    def _start_seed_service_locked(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> bool:
        if self.state is RForkLifecycleState.SERVING:
            return True
        if self.state not in (RForkLifecycleState.INITIALIZED, RForkLifecycleState.REGISTERED):
            logger.error("RFork seed service cannot start from state=%s", self.state.name)
            return False
        if self.seed_lease is not None:
            logger.error("RFork seed service cannot start while a source seed lease is outstanding.")
            return False
        if self.state is not RForkLifecycleState.REGISTERED:
            if not self.transfer_backend.register_memory_region(model, processed_layout, exclude_blocks):
                return False
            self.state = RForkLifecycleState.REGISTERED

        try:
            handle = start_rfork_server(
                self.planner.seed_key,
                self._seed_transfer_info(),
                health_timeout_sec=self.config.seed_timeout_sec,
                bind_host=self.config.seed_bind_host,
            )
            self.seed_server = handle
            if not self.planner.report_seed_once(
                handle.port,
                seed_ip=self.config.seed_advertise_host,
            ):
                raise RuntimeError("planner rejected the initial seed advertisement")

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
            self.heartbeat_thread.start()
            self.state = RForkLifecycleState.SERVING
            logger.info(
                "RFork seed service started for global_rank=%s, port=%s",
                self.identity.global_rank,
                handle.port,
            )
            return True
        except Exception as exc:
            logger.warning("RFork seed service startup failed for global_rank=%s: %s", self.identity.global_rank, exc)
            service_ok = self._stop_seed_service_locked()
            if service_ok:
                self._reset_transfer_locked()
            else:
                self.state = RForkLifecycleState.SERVING
                logger.warning("RFork seed cleanup is incomplete; registered memory remains pinned.")
            return False

    def _stop_seed_service_locked(self) -> bool:
        self.heartbeat_stop_event.set()
        heartbeat = self.heartbeat_thread
        heartbeat_ok = True
        if heartbeat is not None and heartbeat is not threading.current_thread():
            # Requests applies the timeout separately to connection and read.
            # Allow both phases plus thread-exit overhead before removing the seed.
            # This is a bounded wait, not a hard HTTP deadline; retain resources
            # for retry if the request still outlives it (e.g. a trickling response).
            connect_timeout_sec = read_timeout_sec = float(self.config.request_timeout_sec)
            join_timeout_sec = max(
                float(self.config.seed_timeout_sec),
                connect_timeout_sec + read_timeout_sec + HEARTBEAT_STOP_GRACE_SEC,
            )
            heartbeat.join(timeout=join_timeout_sec)
            heartbeat_ok = not heartbeat.is_alive()
            if not heartbeat_ok:
                logger.warning("RFork heartbeat thread did not stop in time.")

        if not heartbeat_ok:
            return False

        if not self.planner.remove_seed():
            logger.warning("RFork planner seed removal failed; keeping the local seed service available for retry.")
            return False
        server_ok = True
        if self.seed_server is not None:
            try:
                server_ok = self.seed_server.stop()
            except Exception as exc:
                logger.warning("RFork seed server shutdown failed: %s", exc)
                server_ok = False

        stopped = heartbeat_ok and server_ok
        if stopped:
            self.heartbeat_thread = None
            self.seed_server = None
            if self.state is RForkLifecycleState.SERVING:
                self.state = RForkLifecycleState.REGISTERED
        return stopped

    def stop_seed_service(self) -> bool:
        with self._lock:
            self._deferred_seed_start = None
            return self._stop_seed_service_locked()

    def shutdown(self) -> bool:
        self.lease_release_stop_event.set()
        with self._lock:
            if self.state is RForkLifecycleState.FINALIZED:
                return True
            self._deferred_seed_start = None
            service_ok = self._stop_seed_service_locked()
            release_ok = self._release_seed_locked()
            finalize_ok = self.transfer_backend.finalize_transfer_engine() if service_ok and release_ok else False
            if finalize_ok:
                self.state = RForkLifecycleState.FINALIZED
            elif not service_ok:
                logger.warning("RFork shutdown retained registered memory because seed service cleanup is incomplete.")
            elif not release_ok:
                logger.warning("RFork shutdown retained TransferEngine state until the seed lease can be released.")
            else:
                logger.warning("RFork shutdown retained TransferEngine state because finalization did not complete.")
            return service_ok and release_ok and finalize_ok
