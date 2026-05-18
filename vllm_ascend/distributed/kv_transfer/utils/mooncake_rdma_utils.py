# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Adapted from vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/
# kv_connector/v1/mooncake/rdma_utils.py
"""Mooncake requester / disk-offload config helpers for vllm-ascend.

Provides:

* ``get_requester_local_hostname``: overrideable local hostname (env var
  ``MOONCAKE_REQUESTER_LOCAL_HOSTNAME``).
* ``get_configured_preferred_segment``: extracts ``preferred_segment``
  override from the kv-connector ``extra_config`` (per-request) and falls
  back to the ``MOONCAKE_PREFERRED_SEGMENT`` env var. Used to pin PUTs to
  a specific owner segment so the SSD tier on that owner receives the
  spill (required by Mooncake's standalone-store + disk-offload mode).
* ``get_configured_worker_rnic``: per-NPU RNIC selection from a CSV
  ``device_name`` list (so multiple DP ranks on the same host don't
  saturate a single NIC).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import vllm_ascend.envs as ascend_envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def normalize_string_override(value: Any) -> str | None:
    """Return ``value`` stripped if it's a non-empty string, else ``None``."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def get_current_physical_npu_index() -> int | None:
    """Return the current process's physical NPU index, or ``None``.

    Mirrors the upstream GPU helper but uses ``torch.npu.current_device``
    (NPU equivalent). Returns ``None`` if torch_npu is unavailable or the
    current process has not bound an NPU device.
    """
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return int(torch.npu.current_device())
    except Exception:  # noqa: BLE001 — best-effort detection
        return None
    return None


def get_requester_local_hostname(local_ip: str) -> str:
    """Hostname this rank registers as a Mooncake requester.

    Resolution order:
      1. ``MOONCAKE_REQUESTER_LOCAL_HOSTNAME`` env override (non-empty).
      2. ``local_ip`` argument.
    """
    override = normalize_string_override(
        ascend_envs.MOONCAKE_REQUESTER_LOCAL_HOSTNAME
    )
    if override is not None:
        return override
    return local_ip


def get_configured_preferred_segment(
    extra_config: Mapping[str, Any],
) -> str | None:
    """Return the configured Mooncake ``preferred_segment`` or ``None``.

    Resolution order:
      1. ``extra_config["preferred_segment"]`` (non-empty string).
      2. ``MOONCAKE_PREFERRED_SEGMENT`` env var.

    Raises ``ValueError`` if ``preferred_segment`` is present in
    ``extra_config`` but not a non-empty string.
    """
    raw = extra_config.get("preferred_segment")
    preferred_segment = normalize_string_override(raw)
    if preferred_segment is not None:
        return preferred_segment
    if raw is not None:
        raise ValueError(
            "Mooncake preferred_segment override must be a non-empty string"
        )

    env_value = normalize_string_override(ascend_envs.MOONCAKE_PREFERRED_SEGMENT)
    if env_value is not None:
        logger.info(
            "Mooncake preferred_segment from MOONCAKE_PREFERRED_SEGMENT: %s",
            env_value,
        )
        return env_value
    return None


def _get_explicit_worker_rnic(device_list: str) -> str:
    """Pick the local NPU's RNIC from a CSV list (indexed by physical NPU)."""
    entries = [entry.strip() for entry in device_list.split(",")]
    if any(not entry for entry in entries):
        raise ValueError(
            "Mooncake worker device_name contains an empty RDMA device entry"
        )
    if len(entries) == 1:
        return entries[0]

    npu_index = get_current_physical_npu_index()
    if npu_index is None:
        raise RuntimeError(
            "Mooncake RDMA requester could not determine the local "
            "physical NPU index for per-rank RNIC selection"
        )
    if npu_index >= len(entries):
        raise ValueError(
            "Mooncake worker device list does not cover local NPU "
            f"{npu_index}: {device_list}"
        )
    device_name = entries[npu_index]
    logger.info(
        "Mooncake selected worker RNIC %s from explicit device list for local NPU %s",
        device_name,
        npu_index,
    )
    return device_name


def get_configured_worker_rnic(
    *,
    protocol: str,
    configured_device: str,
) -> str:
    """Resolve the RDMA NIC for this rank.

    * If ``configured_device`` is a single name, return it.
    * If it is a CSV indexed by physical NPU, return this rank's slot.
    * Otherwise (no override + non-RDMA protocol), return ``""`` so
      Mooncake falls back to its own auto-pick.
    """
    normalized_device = normalize_string_override(configured_device)
    if normalized_device is not None:
        return _get_explicit_worker_rnic(normalized_device)

    # ``ascend`` here means the NPU/HCCL transport, which is RDMA-like.
    if protocol not in {"rdma", "efa", "ascend"}:
        return ""

    logger.warning(
        "No RDMA devices specified for Mooncake backend (protocol=%s). "
        "Set 'device_name' in mooncake_config.json to a single RNIC name "
        "or a comma-separated CSV indexed by physical NPU; falling back to "
        "Mooncake's built-in auto-selection, which may converge on the "
        "same NIC across all DP ranks and saturate bandwidth.",
        protocol,
    )
    return ""
