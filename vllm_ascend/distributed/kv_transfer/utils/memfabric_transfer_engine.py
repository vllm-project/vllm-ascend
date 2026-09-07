# SPDX-License-Identifier: Apache-2.0
"""Process-wide MemFabric transfer-engine initialization.

This module is intentionally separate from ``mooncake_transfer_engine`` so
the existing Mooncake singleton and its initialization contract remain
unchanged.
"""

from __future__ import annotations

import threading
from typing import Any

from vllm.logger import logger

BACKEND_MEMFABRIC = "memfabric"
MEMFABRIC_ROLE_PREFILL = "Prefill"
MEMFABRIC_ROLE_DECODE = "Decode"
_VALID_MEMFABRIC_ROLES = (MEMFABRIC_ROLE_PREFILL, MEMFABRIC_ROLE_DECODE)
# MemFabric data-path protocol, selected via kv_connector_extra_config so this
# module never needs to branch on machine type: sdma/device_rdma for A3 nodes,
# device_urma for A5 nodes.
_VALID_MEMFABRIC_TRANSFER_PROTOCOLS = ("sdma", "device_rdma", "device_urma")
_DEFAULT_MEMFABRIC_TRANSFER_PROTOCOL = "sdma"


class MemfabricBackend:
    """Normalize the MemFabric API used by the SFA PD connector."""

    def __init__(self, engine: Any, advertised_rpc_port: int):
        self._engine = engine
        self._advertised_rpc_port = advertised_rpc_port

    def get_rpc_port(self) -> int:
        return self._advertised_rpc_port

    def register_memory(self, ptr: int, size: int) -> int:
        return 0 if self._engine.register_memory(ptr, size) == 0 else -1

    def batch_transfer_sync_read(
        self,
        session_id: str,
        local_buffers: list[int],
        peer_buffers: list[int],
        length_list: list[int],
    ) -> int:
        ret = self._engine.batch_transfer_sync_read(
            session_id,
            local_buffers,
            peer_buffers,
            length_list,
        )
        if ret != 0:
            logger.error(
                "MemFabric batch_transfer_sync_read failed (ret=%s) for session %s",
                ret,
                session_id,
            )
            return -1
        return 0


class GlobalMemfabricTE:
    """Lazily create one role-bound MemFabric engine per process."""

    def __init__(self):
        self._engine: MemfabricBackend | None = None
        self._role: str | None = None
        self._device_id: int | None = None
        self._transfer_protocol: str | None = None
        self._hostname: str | None = None
        self._unique_id: str | None = None
        self._is_buffer_registered = False
        self._engine_lock = threading.Lock()
        self._register_buffer_lock = threading.Lock()

    @property
    def unique_id(self) -> str:
        if self._unique_id is None:
            raise RuntimeError("MemFabric transfer engine has not been initialized")
        return self._unique_id

    def configure(
        self,
        *,
        role: str,
        device_id: int,
        transfer_protocol: str | None = None,
    ) -> None:
        """Bind this process singleton to one MemFabric role, device and protocol.

        ``transfer_protocol`` comes from
        ``kv_connector_extra_config["memfabric_transfer_protocol"]`` and selects
        the data path: ``sdma``/``device_rdma`` for A3 nodes, ``device_urma``
        for A5 nodes. It defaults to ``sdma``, the memfabric_hybrid library
        default, so launch scripts pick the value per machine type instead of
        this module branching on hardware.
        """
        if role not in _VALID_MEMFABRIC_ROLES:
            raise ValueError(f"Invalid MemFabric role {role!r}; expected one of {_VALID_MEMFABRIC_ROLES}")
        if device_id < 0:
            raise ValueError(f"MemFabric device_id must be non-negative, got {device_id}")
        protocol = (transfer_protocol or _DEFAULT_MEMFABRIC_TRANSFER_PROTOCOL).strip().lower()
        if protocol not in _VALID_MEMFABRIC_TRANSFER_PROTOCOLS:
            raise ValueError(
                f"Invalid MemFabric transfer_protocol={transfer_protocol!r}; "
                f"expected one of {_VALID_MEMFABRIC_TRANSFER_PROTOCOLS}"
            )

        with self._engine_lock:
            configured = self._role is not None
            if configured and (role, device_id, protocol) != (
                self._role,
                self._device_id,
                self._transfer_protocol,
            ):
                raise RuntimeError(
                    "MemFabric transfer engine is already configured for "
                    f"role={self._role}, device_id={self._device_id}, "
                    f"transfer_protocol={self._transfer_protocol}; cannot "
                    f"reconfigure it for role={role}, device_id={device_id}, "
                    f"transfer_protocol={protocol}"
                )
            self._role = role
            self._device_id = device_id
            self._transfer_protocol = protocol

    def get_transfer_engine(self, hostname: str) -> MemfabricBackend:
        with self._engine_lock:
            if self._engine is None:
                if self._role is None or self._device_id is None:
                    raise RuntimeError("MemFabric transfer engine must be configured before initialization")
                self._hostname = hostname
                self._engine = self._build_engine(hostname)
            elif hostname != self._hostname:
                raise RuntimeError(
                    f"MemFabric transfer engine was initialized for hostname {self._hostname!r}, not {hostname!r}"
                )
            return self._engine

    def _get_transfer_protocol(self):
        """Map the configured protocol name to its ``TransDataOpType`` value.

        Unknown names fail fast here as well: a wrong protocol otherwise only
        surfaces later as an obscure engine initialization failure.
        """
        from memfabric_hybrid import TransDataOpType  # type: ignore

        protocol_map = {
            "sdma": TransDataOpType.SDMA,
            "device_rdma": TransDataOpType.DEVICE_RDMA,
            "device_urma": TransDataOpType.DEVICE_URMA,
        }
        protocol = self._transfer_protocol or _DEFAULT_MEMFABRIC_TRANSFER_PROTOCOL
        if protocol not in protocol_map:
            raise ValueError(
                f"Invalid MemFabric transfer_protocol={protocol!r}; expected one of {sorted(protocol_map)}"
            )
        return protocol_map[protocol]

    def _build_engine(self, hostname: str) -> MemfabricBackend:
        try:
            from memfabric_hybrid import (  # type: ignore
                TransferEngine,
                set_conf_store_tls,
                set_log_level,
            )
        except ImportError as exc:
            raise ImportError(
                "Please install memfabric_hybrid (memfabric-hybrid) to use SfaRemoteD2HConnector."
            ) from exc

        # Match the MemFabric initialization sequence used by its examples.
        set_log_level(2)
        set_conf_store_tls(False, "")
        raw_engine = TransferEngine()
        store_url = f"tcp://{hostname}"

        data_op_type = self._get_transfer_protocol()
        logger.info(
            "MemFabric TransferEngine initialize: store_url=%s, unique_id=%s, role=%s, device_id=%s, data_op_type=%s",
            store_url,
            hostname,
            self._role,
            self._device_id,
            getattr(data_op_type, "name", data_op_type),
        )
        ret = raw_engine.initialize(
            store_url,
            hostname,
            self._role,
            self._device_id,
            store_server_role=MEMFABRIC_ROLE_PREFILL,
            data_op_type=data_op_type,
        )
        if ret != 0:
            raise RuntimeError(
                "MemFabric TransferEngine initialization failed with "
                f"ret_value={ret}; hostname={hostname!r} must be a numeric IPv4 "
                "address reachable by the peer"
            )
        advertised_rpc_port = raw_engine.get_rpc_port()
        self._unique_id = f"{hostname}:{advertised_rpc_port}"
        return MemfabricBackend(raw_engine, advertised_rpc_port)

    def register_buffer(self, ptrs: list[int], sizes: list[int]) -> None:
        if len(ptrs) != len(sizes):
            raise ValueError(f"MemFabric registration pointer/size counts differ: {len(ptrs)} != {len(sizes)}")
        with self._register_buffer_lock:
            if self._engine is None:
                raise RuntimeError("MemFabric transfer engine must be initialized")
            if self._is_buffer_registered:
                return
            for ptr, size in zip(ptrs, sizes):
                ret = self._engine.register_memory(ptr, size)
                if ret != 0:
                    raise RuntimeError(f"MemFabric memory registration failed with ret_value={ret}")
            self._is_buffer_registered = True


global_memfabric_te = GlobalMemfabricTE()
