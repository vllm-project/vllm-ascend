# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIXL LLM-DataDist engine wrapper.

Process-level singleton around ``llm_datadist.LLMDataDist`` and its
``CacheManager``. This is the HIXL analogue of
``mooncake_transfer_engine.GlobalTE`` for the Mooncake path.

Responsibilities (engine-coupled by design — HIXL uses block-index
addressing via ``register_blocks_cache``/``pull_blocks``, not the
byte-addressable ``register_memory``/``batch_transfer_sync_read`` of
Mooncake, so the two paths deliberately do not share a transport
interface):

* Lazy import of ``llm_datadist`` (which pulls in ``libcann_hixl.so`` /
  ``libhcomm.so`` / ``torch_npu``), so merely registering the connector
  never crashes a non-HIXL deployment.
* Process-wide singleton init of ``LLMDataDist`` (the underlying engine
  is itself a per-process singleton).
* Idempotent ``link_clusters`` to remote clusters with a lock, since the
  D side pulls from one or more P clusters and links must not be duplicated.
* Best-effort ``shutdown`` (unlink + finalize).
"""

from __future__ import annotations

import threading
from typing import Any

from vllm.logger import logger

# Fixed for the direct-HIXL (no-Mooncake) path. Setting ``transfer_backend``
# requires EnableCacheManager + EnableRemoteCacheAccessible + listen_ip_info,
# all of which ``LLMDataDist.init`` auto-derived from these two options.
_TRANSFER_BACKEND = "hixl"
_LOCAL_COMM_RES = ""
_DEFAULT_LINK_TIMEOUT_MS = 5000


def kv_role_to_llm_role(kv_role: str):
    """Map a vLLM ``kv_role`` string to ``llm_datadist.LLMRole``.

    Args:
        kv_role: ``"kv_producer"`` (Prefill/P, exposes KV) or
            ``"kv_consumer"`` (Decode/D, pulls KV).

    Returns:
        ``LLMRole.PROMPT`` or ``LLMRole.DECODER``.
    """
    from llm_datadist import LLMRole

    if kv_role == "kv_producer":
        return LLMRole.PROMPT
    if kv_role == "kv_consumer":
        return LLMRole.DECODER
    raise ValueError(
        f"Unsupported kv_role for HIXLConnector: {kv_role!r} "
        "(expected 'kv_producer' or 'kv_consumer')"
    )


class HixlDataDist:
    """Owns one ``LLMDataDist`` (process singleton) and its ``CacheManager``."""

    def __init__(
        self,
        kv_role: str,
        cluster_id: int,
        listen_ip: str,
        listen_port: int,
        device_id: int | None = None,
        link_timeout_ms: int = _DEFAULT_LINK_TIMEOUT_MS,
        extra_options: dict[str, Any] | None = None,
    ) -> None:
        # Lazy import: importing llm_datadist loads native libs.
        from llm_datadist import LLMConfig, LLMDataDist

        self._llm_role = kv_role_to_llm_role(kv_role)
        self._cluster_id = int(cluster_id)
        self._listen_ip = listen_ip
        self._listen_port = int(listen_port)
        self._link_timeout_ms = int(link_timeout_ms)
        self._link_lock = threading.Lock()
        self._linked_remote_clusters: set[int] = set()

        cfg = LLMConfig()
        if device_id is not None:
            cfg.device_id = int(device_id)
        cfg.local_comm_res = _LOCAL_COMM_RES  # triggers EnableCacheManager
        # transfer_backend requires EnableRemoteCacheAccessible + listen_ip_info,
        # both auto-enabled by LLMDataDist.init when these are set.
        cfg.transfer_backend = _TRANSFER_BACKEND
        cfg.listen_ip_info = f"{listen_ip}:{self._listen_port}"

        extra = extra_options or {}
        if "link_total_time" in extra:
            cfg.link_total_time = int(extra["link_total_time"])
        if "link_retry_count" in extra:
            cfg.link_retry_count = int(extra["link_retry_count"])
        if "sync_kv_timeout" in extra:
            cfg.sync_kv_timeout = extra["sync_kv_timeout"]
        # Pass-through of arbitrary raw ge/llm options from extra_config.
        for k, v in (extra.get("llm_options") or {}).items():
            cfg.ge_options[k] = str(v)

        self._datadist = LLMDataDist(self._llm_role, self._cluster_id)
        self._datadist.init(cfg.generate_options())
        self._cache_manager = self._datadist.cache_manager
        logger.info(
            "HixlDataDist initialized: role=%s cluster_id=%d listen=%s:%d",
            self._llm_role.name,
            self._cluster_id,
            self._listen_ip,
            self._listen_port,
        )

    @property
    def datadist(self):
        return self._datadist

    @property
    def cache_manager(self):
        return self._cache_manager

    @property
    def cluster_id(self) -> int:
        return self._cluster_id

    @property
    def listen_ip(self) -> str:
        return self._listen_ip

    @property
    def listen_port(self) -> int:
        return self._listen_port

    @property
    def llm_role(self):
        return self._llm_role

    def ensure_linked(
        self, remote_cluster_id: int, remote_ip: str, remote_port: int
    ) -> None:
        """Idempotently ``link_clusters`` to a remote cluster.

        The caller (D side, before its first ``pull_blocks`` to a given P
        cluster) invokes this. Repeat calls for an already-linked
        ``remote_cluster_id`` are no-ops. ``LLMClusterInfo`` must carry both
        local and remote ip_info because ``local_comm_res`` is empty.

        Raises:
            RuntimeError: if ``link_clusters`` does not return ``LLM_SUCCESS``.
        """
        from llm_datadist import LLMClusterInfo, LLMStatusCode

        remote_cluster_id = int(remote_cluster_id)
        if remote_cluster_id in self._linked_remote_clusters:
            return
        with self._link_lock:
            if remote_cluster_id in self._linked_remote_clusters:
                return
            cluster = LLMClusterInfo()
            cluster.remote_cluster_id = remote_cluster_id
            cluster.append_local_ip_info(self._listen_ip, self._listen_port)
            cluster.append_remote_ip_info(remote_ip, int(remote_port))
            code, per_cluster = self._datadist.link_clusters(
                [cluster], self._link_timeout_ms
            )
            if code != LLMStatusCode.LLM_SUCCESS:
                raise RuntimeError(
                    f"HIXL link_clusters failed: remote_cluster_id="
                    f"{remote_cluster_id} ({remote_ip}:{remote_port}), "
                    f"code={code}, per_cluster={per_cluster}"
                )
            self._linked_remote_clusters.add(remote_cluster_id)
            logger.info(
                "HIXL linked remote_cluster_id=%d (%s:%d)",
                remote_cluster_id,
                remote_ip,
                remote_port,
            )

    def shutdown(self) -> None:
        """Unlink all linked clusters and finalize. Safe to call repeatedly."""
        if getattr(self, "_datadist", None) is None:
            return
        from llm_datadist import LLMClusterInfo

        try:
            for rid in list(self._linked_remote_clusters):
                cluster = LLMClusterInfo()
                cluster.remote_cluster_id = rid
                try:
                    self._datadist.unlink_clusters(
                        [cluster], self._link_timeout_ms, force=True
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("HIXL unlink remote_cluster_id=%d failed: %s", rid, e)
            self._linked_remote_clusters.clear()
        finally:
            try:
                self._datadist.finalize()
            except Exception as e:  # noqa: BLE001
                logger.warning("HIXL finalize failed: %s", e)


# ---------------------------------------------------------------------------
# Process singleton. LLMDataDist is itself a per-process singleton
# (llm_engine_instance), so we gate creation with a lock and assert identity
# on reuse.
# ---------------------------------------------------------------------------
_global_lock = threading.Lock()
_global_instance: HixlDataDist | None = None


def get_datadist(
    kv_role: str,
    cluster_id: int,
    listen_ip: str,
    listen_port: int,
    device_id: int | None = None,
    link_timeout_ms: int = _DEFAULT_LINK_TIMEOUT_MS,
    extra_options: dict[str, Any] | None = None,
) -> HixlDataDist:
    """Get or create the process-wide ``HixlDataDist`` singleton.

    On subsequent calls the cached instance is returned; its ``cluster_id`` is
    asserted to match, since the underlying ``LLMDataDist`` cannot be recreated
    with different identity in the same process.
    """
    global _global_instance
    if _global_instance is not None:
        inst = _global_instance
        assert inst.cluster_id == int(cluster_id), (
            f"HIXL cluster_id mismatch: singleton={inst.cluster_id} "
            f"requested={cluster_id}"
        )
        return inst
    with _global_lock:
        if _global_instance is not None:
            return _global_instance
        _global_instance = HixlDataDist(
            kv_role=kv_role,
            cluster_id=cluster_id,
            listen_ip=listen_ip,
            listen_port=listen_port,
            device_id=device_id,
            link_timeout_ms=link_timeout_ms,
            extra_options=extra_options,
        )
    return _global_instance


def shutdown_datadist() -> None:
    """Tear down the singleton (called from ``HIXLConnector.shutdown``)."""
    global _global_instance
    with _global_lock:
        inst = _global_instance
        _global_instance = None
    if inst is not None:
        inst.shutdown()
