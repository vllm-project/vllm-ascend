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

Whether a listener really conflicts with the endpoint cannot be decided from
the addresses alone: it depends on the address family, on ``IPV6_V6ONLY`` and
on the wildcard/specific address combination. An IPv6 listener bound to ``::``
with ``IPV6_V6ONLY=1`` for example does not keep an IPv4 endpoint of the same
port busy, while a dual stack one does. The conflict is therefore confirmed by
binding the endpoint the same way the engine binds it, so the check agrees
with the kernel instead of guessing and never rejects a startup that would
succeed.
"""

import errno
import socket
from typing import Any

from vllm.logger import logger

__all__ = ["check_dp_rpc_port_available", "find_tcp_listener"]

# Addresses that stand for every local address of their family.
_WILDCARD_ADDRESSES = ("0.0.0.0", "::", "")


def _resolve_ip(host: str) -> set[str]:
    """Return the IP addresses that ``host`` resolves to."""
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return set()
    return {str(info[4][0]) for info in infos}


def _normalize_ip(address: str) -> str:
    """Return ``address`` without its IPv4-mapped IPv6 prefix."""
    prefix = "::ffff:"
    if address.startswith(prefix):
        return address[len(prefix) :]
    return address


def _address_matches(listen_ip: str, host: str) -> bool:
    """Whether a socket bound to ``listen_ip`` may own ``host``.

    This only tells which listener of a port is the one to blame, whether
    the endpoint is really taken is decided by :func:`_is_endpoint_free`.
    """
    if listen_ip in _WILDCARD_ADDRESSES or host in _WILDCARD_ADDRESSES:
        # A wildcard listener owns every local address of the port, and
        # binding a wildcard address conflicts with any existing listener.
        return True
    listen_ip = _normalize_ip(listen_ip)
    return listen_ip in {_normalize_ip(ip) for ip in _resolve_ip(host)}


def _endpoint_family(host: str) -> int:
    """Return the address family the engine binds the endpoint with."""
    address = host.strip("[]")
    if not address:
        # An empty host stands for the wildcard address of the IPv4 family.
        return socket.AF_INET
    if ":" in address:
        return socket.AF_INET6
    try:
        infos = socket.getaddrinfo(address, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return socket.AF_INET
    for info in infos:
        if info[0] in (socket.AF_INET, socket.AF_INET6):
            return info[0]
    return socket.AF_INET


def _is_endpoint_free(host: str, port: int) -> bool:
    """Return whether the endpoint is free, that is, not held by a socket.

    The socket is created with the same options as the handshake socket
    libzmq binds, so the answer is the one the engine gets when it binds the
    endpoint. An error other than "address already in use" means that the
    endpoint cannot be bound at all, which is not a conflict with another
    process and therefore reported as free.
    """
    family = _endpoint_family(host)
    address = host.strip("[]")
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        # The address family is not available on this host.
        return True
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Windows: libzmq asks for an exclusive address, so a port that
            # is in use cannot be taken over.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # libzmq allows the reuse of the address before it binds.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
            # libzmq enables the IPv4 mapping on its IPv6 sockets.
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind((address, port))
    except OSError as exc:
        # Only "address already in use" means that the endpoint is taken,
        # any other error means that it cannot be bound at all.
        return exc.errno != errno.EADDRINUSE
    finally:
        sock.close()
    return True


def _process_name(psutil: Any, pid: int) -> str:
    """Return the name of the process, or ``unknown`` when it is gone."""
    try:
        return psutil.Process(pid).name()
    except psutil.Error:
        return "unknown"


def find_tcp_listener(host: str, port: int) -> tuple[int, str] | None:
    """Return the ``(pid, process name)`` that holds ``host:port``.

    ``None`` is returned when the endpoint is free, when the owner cannot be
    identified (for example when the socket belongs to a process owned by
    another user) or when the listeners of the port do not really conflict
    with the endpoint.
    """
    try:
        import psutil
    except ImportError:
        return None

    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return None

    listeners: list[tuple[str, int]] = []
    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or conn.pid is None:
            continue
        laddr = conn.laddr
        # psutil types the local address as a union of the address
        # namedtuple and an empty tuple for unix sockets.
        if getattr(laddr, "port", None) != port:
            continue
        listen_ip = getattr(laddr, "ip", None)
        if listen_ip is None:
            continue
        listeners.append((str(listen_ip), conn.pid))

    if not listeners:
        return None

    if _is_endpoint_free(host, port):
        # The port has a listener, but it does not conflict with the
        # endpoint, for example an IPv6 only listener next to an IPv4
        # endpoint. The engine can still bind it.
        return None

    for listen_ip, pid in listeners:
        if _address_matches(listen_ip, host):
            return pid, _process_name(psutil, pid)
    _, pid = listeners[0]
    return pid, _process_name(psutil, pid)


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
