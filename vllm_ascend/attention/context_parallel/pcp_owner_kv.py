# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Owner-sharded PCP KV placement: configuration and support checks.

The device side lives in :mod:`pcp_owner_vmm`: every rank reserves the complete
logical address range over a VMM slab and maps only ``1 / pcp_size`` of its
physical two-megabyte chunks, so the stock attention kernel reads and writes
the whole logical cache without knowing which rank owns a page.  The consumer
side (long-KV and per-chunk SWA staging) lives in :mod:`pcp_owner_stage`.

This module only selects and validates the placement mode.  It performs no
device work and imports no device libraries, so the cache planner can call it
before the worker exists.
"""

from __future__ import annotations

from vllm.logger import logger

PCP_KV_SHARING_REPLICATED = "replicated"
PCP_KV_SHARING_OWNER = "owner"
PCP_KV_SHARING_MODES = (PCP_KV_SHARING_REPLICATED, PCP_KV_SHARING_OWNER)


def pcp_kv_sharing_mode_from_config(vllm_config) -> str:
    """Return the typed ``additional_config.pcp_config.kv_sharing`` value."""
    from vllm_ascend.ascend_config import pcp_config_from_vllm_config

    return pcp_config_from_vllm_config(vllm_config).kv_sharing


def owner_sharding_enabled(vllm_config=None) -> bool:
    """Whether ``pcp_config.kv_sharing == "owner"`` (default: replicated)."""
    if vllm_config is None:
        return False
    return pcp_kv_sharing_mode_from_config(vllm_config) == PCP_KV_SHARING_OWNER


def validate_owner_sharing(vllm_config) -> tuple[bool, str]:
    """Check whether owner sharing can be enabled for this configuration."""
    parallel = vllm_config.parallel_config
    if parallel.prefill_context_parallel_size <= 1:
        return False, "owner sharing requires prefill_context_parallel_size > 1"
    if parallel.decode_context_parallel_size != 1:
        return False, "owner sharing requires decode_context_parallel_size == 1"
    if parallel.tensor_parallel_size != 1:
        return False, "owner sharing currently requires tensor_parallel_size == 1 (PCP replicated-cache contract)"
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        return False, "owner sharing does not support an external KV-transfer connector"
    return True, ""


def resolve_owner_sharing(vllm_config) -> tuple[bool, str]:
    """Whether this process may run the owner-sharded PCP prefill path."""
    supported, reason = validate_owner_sharing(vllm_config)
    if not supported:
        return False, reason
    # The device backend is probed without touching a device: it only checks
    # that the CANN V2 symbols and the torch_npu external-storage constructors
    # exist.  The actual export/import/map happens once during KV allocation
    # and falls back collectively if any step fails.
    from .pcp_owner_vmm import owner_slab_support

    available, reason = owner_slab_support()
    if not available:
        return False, "owner-slab allocator is not installed in this build: " + reason
    return True, ""


def log_owner_sharing_fallback(reason: str) -> None:
    logger.warning_once("PCP owner-sharded KV is unavailable (%s); using the replicated cache path.", reason)
