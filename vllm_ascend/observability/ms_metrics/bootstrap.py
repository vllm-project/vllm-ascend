# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap MS Service Metric for Ascend.

Runs early from ``NPUPlatform.pre_register_and_update`` so worker processes
inherit ``PROMETHEUS_MULTIPROC_DIR``. When ``ms_service_metric`` is installed,
collection defaults to ON (same shared-memory switch as ``ms-service-metric
on``); operators can still ``ms-service-metric off`` at runtime.
"""

from __future__ import annotations

import contextlib
import os

from vllm.logger import logger

_bootstrapped = False


def ensure_prometheus_multiproc_dir() -> str | None:
    """Ensure ``PROMETHEUS_MULTIPROC_DIR`` is set and the directory exists.

    Reuses a user-provided path when present. Otherwise delegates to vLLM's
    ``setup_multiprocess_prometheus`` so Ascend and native metrics share one
    multiprocess directory. MS hooks run in engine/worker processes, so this
    is required even for a single API server.
    """
    existing = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if existing:
        try:
            os.makedirs(existing, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "[ms_metrics] PROMETHEUS_MULTIPROC_DIR=%s is not usable: %s",
                existing,
                exc,
            )
            return existing
        return existing

    try:
        from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

        setup_multiprocess_prometheus()
    except Exception as exc:
        logger.warning(
            "[ms_metrics] Failed to auto-set PROMETHEUS_MULTIPROC_DIR via vLLM: %s",
            exc,
        )
        return None

    path = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if path:
        logger.info_once(
            "[ms_metrics] Auto-set PROMETHEUS_MULTIPROC_DIR=%s",
            path,
            scope="process",
        )
    return path


def enable_ms_metric_collection(force: bool = False) -> bool:
    """Set the MS Service Metric shared-memory switch to ON.

    Equivalent to ``ms-service-metric on``: creates the control shared memory
    when missing, then marks collection ON. Already-running vLLM processes are
    signaled; processes that start later read ON during ``MetricControlWatch``
    startup. Returns False when the package is missing or the switch cannot be
    written (for example non-Linux without ``posix_ipc``).
    """
    try:
        from ms_service_metric.utils.shm_manager import (  # type: ignore[import-not-found]
            POSIX_IPC_AVAILABLE,
            SharedMemoryManager,
        )
    except ImportError:
        logger.debug("[ms_metrics] ms_service_metric is not installed; skip default-on")
        return False

    if not POSIX_IPC_AVAILABLE:
        logger.warning("[ms_metrics] posix_ipc is unavailable; cannot enable MS Service Metric")
        return False

    manager = SharedMemoryManager()
    try:
        if not manager.connect(create=True):
            logger.warning("[ms_metrics] Failed to connect MS metric control shared memory")
            return False
        success, signaled, _cleaned, changed = manager.send_control_command(
            is_start=True,
            force=force,
            send_signal=True,
        )
        if not success:
            logger.warning("[ms_metrics] Failed to send MS metric ON command")
            return False
        if changed:
            logger.info(
                "[ms_metrics] Default-on enabled MS Service Metric (signaled=%s processes)",
                signaled,
            )
        else:
            logger.info_once(
                "[ms_metrics] MS Service Metric already ON",
                scope="process",
            )
        return True
    except Exception as exc:
        logger.warning("[ms_metrics] Failed to enable MS Service Metric: %s", exc)
        return False
    finally:
        with contextlib.suppress(Exception):
            manager.disconnect()


def bootstrap_ms_metrics() -> None:
    """Idempotent startup bootstrap for Ascend MS metrics.

    1. Auto-set ``PROMETHEUS_MULTIPROC_DIR`` when unset.
    2. If ``ms_service_metric`` is installed, default collection to ON.
       Use ``ms-service-metric off`` to disable at runtime.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    ensure_prometheus_multiproc_dir()
    enable_ms_metric_collection()
