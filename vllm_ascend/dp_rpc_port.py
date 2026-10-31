#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Preflight check for the data parallel RPC port.

The data parallel master binds ``tcp://<master ip>:<rpc port>`` for the
startup handshake. When a previous engine did not exit cleanly, that stale
process keeps the port (and the NPU devices) alive, so the next engine
started in the same container fails with a bare::

    zmq.error.ZMQError: Address already in use (addr='tcp://<node-ip>:12321')

The bind itself lives in upstream vLLM, so this module only detects the
conflict early and reports it with the information needed to recover from it.
"""

import socket
from typing import Any

from vllm.logger import logger

__all__ = ["check_dp_rpc_port_available", "find_tcp_listener"]


def _resolve_ipv4(host: str) -> set[str]:
    """Return the IPv4 addresses that ``host`` resolves to."""
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return set()
    return {str(info[4][0]) for info in infos}


def _address_matches(listen_ip: str, host: str) -> bool:
    """Whether a socket bound to ``listen_ip`` also owns ``host``."""
    if listen_ip in ("0.0.0.0", ""):
        # A wildcard listener owns every local address of the port.
        return True
    return listen_ip in _resolve_ipv4(host)


def find_tcp_listener(host: str, port: int) -> tuple[int, str] | None:
    """Return the ``(pid, process name)`` listening on ``host:port``.

    ``None`` is returned when nothing listens on the endpoint, or when the
    owner cannot be identified (for example when the socket belongs to a
    process owned by another user).
    """
    try:
        import psutil
    except ImportError:
        return None

    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return None

    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or conn.pid is None:
            continue
        laddr = conn.laddr
        # psutil types the local address as a union of the address
        # namedtuple and an empty tuple for unix sockets.
        listen_port = getattr(laddr, "port", None)
        listen_ip = getattr(laddr, "ip", None)
        if listen_port != port or listen_ip is None:
            continue
        if not _address_matches(str(listen_ip), host):
            continue
        try:
            name = psutil.Process(conn.pid).name()
        except psutil.Error:
            name = "unknown"
        return conn.pid, name
    return None


def _dp_rpc_endpoint(parallel_config: Any) -> tuple[str, int] | None:
    """Return the endpoint the data parallel master binds, if any.

    Upstream only uses the TCP handshake transport when the data parallel
    engines are not all local, and only the rank 0 frontend binds the
    handshake address; the other ranks connect to it. Mirroring that here
    keeps the check from firing on deployments that never bind the port.
    """
    dp_size = getattr(parallel_config, "data_parallel_size", 1)
    if dp_size is None or dp_size <= 1:
        return None

    size_local = getattr(parallel_config, "data_parallel_size_local", None)
    if size_local is not None and size_local == dp_size:
        return None

    if getattr(parallel_config, "data_parallel_rank", 0) != 0:
        return None

    if getattr(parallel_config, "node_rank_within_dp", 0) != 0:
        # Worker nodes of a multi node replica do not bind the address.
        return None

    host = getattr(parallel_config, "data_parallel_master_ip", None)
    port = getattr(parallel_config, "data_parallel_rpc_port", None)
    if not host or not port:
        return None
    return str(host), int(port)


def check_dp_rpc_port_available(parallel_config: Any) -> None:
    """Fail fast when the data parallel RPC port is already taken.

    Raises:
        RuntimeError: when another process already listens on the endpoint
            that this engine is going to bind.
    """
    endpoint = _dp_rpc_endpoint(parallel_config)
    if endpoint is None:
        return
    host, port = endpoint

    holder = find_tcp_listener(host, port)
    if holder is None:
        return

    pid, name = holder
    logger.error(
        "The data parallel RPC port %s:%d is already in use (PID %d, %s).",
        host,
        port,
        pid,
        name,
    )
    raise RuntimeError(
        f"The data parallel RPC port {host}:{port} is already in use by "
        f"PID {pid} ({name}). A previous engine that did not exit cleanly "
        "keeps the port and the NPU devices alive, so this engine cannot "
        "start. Stop the stale process or restart the container, then start "
        "the engine again."
    )
