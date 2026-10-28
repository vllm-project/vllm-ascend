# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select the PD KV transfer backend used by the Mooncake connector.

The dense connector used to talk to the Mooncake transfer engine only, and that
engine was always initialized with one hardcoded transport. MemFabric exposes
the same surface the connector needs (``batch_transfer_sync_read``,
``register_memory``, ``get_rpc_port``) but moves device memory through SDMA
instead of the Mooncake Ascend transport's ``aclrtMemcpy`` staging, which on a
same-host prefill/decode pair is several times faster.

Which engine a machine can actually use depends on the interconnect (RoCE or
fabric memory availability), not on the model, so the choice is a launch
parameter read from ``kv_connector_extra_config``:

* ``transfer_backend``: ``mooncake`` (default) or ``memfabric``
* ``transfer_protocol``: Mooncake transport, default ``ascend``
* ``memfabric_transfer_protocol``: MemFabric data path, default ``sdma``

Prefill and decode must be configured identically.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import (
    MEMFABRIC_ROLE_DECODE,
    MEMFABRIC_ROLE_PREFILL,
    global_memfabric_te,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    DEFAULT_TRANSFER_PROTOCOL,
    global_te,
)

BACKEND_MOONCAKE = "mooncake"
BACKEND_MEMFABRIC = "memfabric"
DEFAULT_TRANSFER_BACKEND = BACKEND_MOONCAKE
VALID_TRANSFER_BACKENDS = (BACKEND_MOONCAKE, BACKEND_MEMFABRIC)


def _extra_config(vllm_config: VllmConfig, key: str, default: Any) -> Any:
    """Read one extra-config key, tolerating the stubs unit tests build.

    Real deployments always go through ``KVTransferConfig``; test doubles often
    only carry the fields a case needs. An explicit ``None`` also means "not
    set", otherwise ``str(None)`` would reach the native engine as a value.
    """
    getter = getattr(vllm_config.kv_transfer_config, "get_from_extra_config", None)
    value = default if getter is None else getter(key, default)
    return default if value is None else value


def resolve_transfer_backend(vllm_config: VllmConfig) -> str:
    """Pick the transfer backend, defaulting to the historical Mooncake path."""
    backend = _extra_config(vllm_config, "transfer_backend", DEFAULT_TRANSFER_BACKEND)
    backend = str(backend).strip().lower() or DEFAULT_TRANSFER_BACKEND
    if backend not in VALID_TRANSFER_BACKENDS:
        raise ValueError(f"Invalid transfer_backend={backend!r}; expected one of {VALID_TRANSFER_BACKENDS}.")
    return backend


def resolve_mooncake_protocol(vllm_config: VllmConfig) -> str:
    """Mooncake transport, ``ascend`` unless the deployment overrides it."""
    protocol = _extra_config(vllm_config, "transfer_protocol", DEFAULT_TRANSFER_PROTOCOL)
    return str(protocol).strip().lower() or DEFAULT_TRANSFER_PROTOCOL


def _memfabric_role(kv_role: str) -> str:
    """Map the connector role onto the MemFabric prefill/decode role."""
    if kv_role == "kv_producer":
        return MEMFABRIC_ROLE_PREFILL
    if kv_role == "kv_consumer":
        return MEMFABRIC_ROLE_DECODE
    raise ValueError(f"Transfer backend {BACKEND_MEMFABRIC!r} needs a producer/consumer role, got {kv_role!r}.")


def ensure_mooncake_backend(vllm_config: VllmConfig, connector: str) -> None:
    """Reject a backend selection a Mooncake-only connector cannot honour.

    Only :class:`MooncakeConnectorV1` can route through MemFabric. Every other
    Mooncake connector reads ``transfer_backend`` from the same extra config, so
    silently ignoring it there would look like the knob did nothing.
    """
    backend = resolve_transfer_backend(vllm_config)
    if backend != BACKEND_MOONCAKE:
        raise ValueError(
            f"{connector} supports transfer_backend={BACKEND_MOONCAKE!r} only, got {backend!r}. "
            "Set it on MooncakeConnectorV1 or leave the default."
        )
    # These connectors neither read the protocol nor pass it to the engine, so
    # accepting a custom value here would be another knob that does nothing.
    protocol = resolve_mooncake_protocol(vllm_config)
    if protocol != DEFAULT_TRANSFER_PROTOCOL:
        raise ValueError(
            f"{connector} does not support transfer_protocol={protocol!r}; it always uses "
            f"{DEFAULT_TRANSFER_PROTOCOL!r}. Set it on MooncakeConnectorV1 or leave the default."
        )


def build_transfer_engine(
    vllm_config: VllmConfig,
    hostname: str,
    device_name: str | None,
    kv_role: str,
    mooncake_registry: Any | None = None,
) -> tuple[Any, Any]:
    """Initialize the configured backend.

    Returns ``(registry, engine)``. ``registry`` owns the process-wide engine and
    the KV buffer registration; ``engine`` is the object the connector issues
    transfers on. Both backends implement the same three methods the connector
    depends on, so the call sites stay backend agnostic.
    """
    backend = resolve_transfer_backend(vllm_config)
    if backend == BACKEND_MEMFABRIC:
        role = _memfabric_role(kv_role)
        protocol = _extra_config(vllm_config, "memfabric_transfer_protocol", None)
        logger.info(
            "KV transfer backend: backend=%s role=%s protocol=%s",
            backend,
            role,
            protocol or "sdma (MemFabric default)",
        )
        # MemFabric binds its process-wide engine to one role/device/protocol, so
        # it has to be configured before the first get_transfer_engine() call.
        global_memfabric_te.configure(
            role=role,
            device_id=torch.npu.current_device(),
            transfer_protocol=protocol,
        )
        return global_memfabric_te, global_memfabric_te.get_transfer_engine(hostname)

    protocol = resolve_mooncake_protocol(vllm_config)
    logger.info(
        "KV transfer backend: backend=%s role=%s protocol=%s",
        backend,
        kv_role,
        protocol,
    )
    # Bind late: callers (and tests) may replace the module-level singleton.
    registry = global_te if mooncake_registry is None else mooncake_registry
    return registry, registry.get_transfer_engine(
        hostname,
        device_name=device_name,
        protocol=protocol,
    )
