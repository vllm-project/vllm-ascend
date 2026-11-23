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

# Connectors that can transfer a complete logical KV block out of an
# owner-sharded cache. They read the full logical address range (every PCP rank
# maps it) and the connector stages the block through a transport-registered
# HBM buffer when the transport engine cannot register the VMM range itself.
# The block id / block table contract is unchanged: each key still carries a
# complete logical page.
OWNER_COMPATIBLE_KV_CONNECTORS = frozenset(
    {
        "AscendStoreConnector",
        "MooncakeConnectorStoreV1",
    }
)
# Direct P2P connectors read the same complete logical block, but their
# transfer thread has not been wired to the owner staging buffer yet. Adding a
# name here requires a PD equality run first; until then they fall back
# collectively to the replicated cache.

# Layerwise transfer hands the engine per-layer slices while the layers are
# still live. That path has not been validated against the owner slab, so it
# must fall back collectively instead of risking a partial page.
_PCP_TRANSFER_CONNECTORS = frozenset(
    {
        "AscendStoreConnector",
        "MooncakeConnectorStoreV1",
        "MooncakeLayerwiseConnector",
        "MooncakeConnectorV1",
        "MooncakeHybridConnector",
        "MooncakeD2RHConnectorV1",
    }
)


def pcp_kv_sharing_mode_from_config(vllm_config) -> str:
    """Return the typed ``additional_config.pcp_config.kv_sharing`` value."""
    from vllm_ascend.ascend_config import pcp_config_from_vllm_config

    return pcp_config_from_vllm_config(vllm_config).kv_sharing


def owner_sharding_enabled(vllm_config=None) -> bool:
    """Whether ``pcp_config.kv_sharing == "owner"`` (default: replicated)."""
    if vllm_config is None:
        return False
    return pcp_kv_sharing_mode_from_config(vllm_config) == PCP_KV_SHARING_OWNER


def _iter_connector_specs(name: str | None, extra) -> list[tuple[str, dict]]:
    """Flatten a ``MultiConnector`` tree into ``(name, extra_config)`` leaves."""
    if not name:
        return [("", {})]
    if name not in ("MultiConnector", "AscendMultiConnector"):
        return [(name, extra if isinstance(extra, dict) else {})]
    children = extra.get("connectors") if isinstance(extra, dict) else None
    if not isinstance(children, (list, tuple)) or not children:
        return [(name, extra if isinstance(extra, dict) else {})]
    leaves: list[tuple[str, dict]] = []
    for child in children:
        if not isinstance(child, dict):
            leaves.append(("", {}))
            continue
        leaves.extend(
            _iter_connector_specs(
                child.get("kv_connector"),
                child.get("kv_connector_extra_config") or {},
            )
        )
    return leaves


def validate_owner_connector(vllm_config) -> tuple[bool, str]:
    """Check whether the configured KV-transfer connector can carry owner KV.

    All processes read the same ``vllm_config``, so a rejection here is
    collective: every PCP rank falls back to the replicated cache together.
    The gate is deliberately conservative. A connector is accepted only when it
    addresses complete logical blocks from a rank whose owner slab maps the
    full logical range (staging is performed inside the connector when the
    transport engine cannot register that range).
    """
    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer is None:
        return True, ""
    name = getattr(kv_transfer, "kv_connector", None)
    extra = getattr(kv_transfer, "kv_connector_extra_config", None) or {}
    leaves = _iter_connector_specs(name, extra)
    if not leaves or any(not leaf_name for leaf_name, _ in leaves):
        return False, "KV-transfer configuration does not name every connector"
    for leaf_name, leaf_extra in leaves:
        if leaf_name not in OWNER_COMPATIBLE_KV_CONNECTORS:
            supported = ", ".join(sorted(OWNER_COMPATIBLE_KV_CONNECTORS))
            return (
                False,
                f"connector {leaf_name} is not owner-aware (supported: {supported})",
            )
        if leaf_name in _PCP_TRANSFER_CONNECTORS and leaf_extra.get("use_layerwise", False):
            return (
                False,
                f"connector {leaf_name} layerwise transfer has not been validated with owner-sharded KV",
            )
    return True, ""


def validate_owner_sharing(vllm_config) -> tuple[bool, str]:
    """Check whether owner sharing can be enabled for this configuration."""
    parallel = vllm_config.parallel_config
    if parallel.prefill_context_parallel_size <= 1:
        return False, "owner sharing requires prefill_context_parallel_size > 1"
    if parallel.decode_context_parallel_size != 1:
        return False, "owner sharing requires decode_context_parallel_size == 1"
    if parallel.tensor_parallel_size != 1:
        return False, "owner sharing currently requires tensor_parallel_size == 1 (PCP replicated-cache contract)"
    connector_ok, connector_reason = validate_owner_connector(vllm_config)
    if not connector_ok:
        return False, connector_reason
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
