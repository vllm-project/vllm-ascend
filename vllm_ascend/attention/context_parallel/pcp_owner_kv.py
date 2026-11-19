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


def get_pcp_kv_sharing_mode() -> str:
    """Return the requested PCP KV sharing mode (default: replicated)."""
    from vllm_ascend import envs

    return normalize_pcp_kv_sharing_mode(envs.VLLM_ASCEND_PCP_KV_SHARING)


def normalize_pcp_kv_sharing_mode(raw: str | None) -> str:
    mode = (raw or PCP_KV_SHARING_REPLICATED).strip().lower()
    if mode not in PCP_KV_SHARING_MODES:
        raise ValueError(f"pcp_kv_sharing must be one of {PCP_KV_SHARING_MODES}, got {raw!r}")
    return mode


def pcp_kv_sharing_mode_from_config(vllm_config) -> str:
    """Read ``additional_config.pcp_kv_sharing``, then the env override."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    if "pcp_kv_sharing" in additional:
        return normalize_pcp_kv_sharing_mode(additional["pcp_kv_sharing"])
    return get_pcp_kv_sharing_mode()


def owner_sharding_enabled(vllm_config=None) -> bool:
    if vllm_config is None:
        return get_pcp_kv_sharing_mode() == PCP_KV_SHARING_OWNER
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
