# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Capability-based listener selection for the sample proxy's Uvicorn workers."""

import asyncio
import inspect
import select
import socket
import sys
from contextlib import ExitStack

import uvicorn
from uvicorn.supervisors import Multiprocess

_PROBE_CONNECTIONS = 32
_PROBE_TIMEOUT = 1

try:
    import uvloop

    _Loop = uvloop.Loop
except ImportError:
    _Loop = asyncio.SelectorEventLoop


class _WorkerLoop(_Loop):
    async def create_server(self, *args, **kwargs):
        # Each child binds its own listener; SO_REUSEPORT distributes new TCP
        # connections, not subsequent HTTP requests on a keep-alive connection.
        if kwargs.get("sock") is None:
            kwargs["reuse_port"] = True
        return await super().create_server(*args, **kwargs)


class _WorkerLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        return _WorkerLoop()


class _WorkerConfig(uvicorn.Config):
    def setup_event_loop(self):
        # Uvicorn before the loop-factory API uses an asyncio policy.
        asyncio.set_event_loop_policy(_WorkerLoopPolicy())

    def get_loop_factory(self):
        return _WorkerLoop


def supports_worker_listeners(host: str) -> bool:
    """Check that independent listeners can both receive new TCP connections."""
    reuse_port = getattr(socket, "SO_REUSEPORT", None)
    if reuse_port is None:
        return False
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    address = ("::1" if family == socket.AF_INET6 else "127.0.0.1", 0)
    # A successful bind alone does not establish connection-distribution support.
    try:
        with ExitStack() as stack:
            listeners = [stack.enter_context(socket.socket(family, socket.SOCK_STREAM)) for _ in range(2)]
            for listener in listeners:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.setsockopt(socket.SOL_SOCKET, reuse_port, 1)
                listener.bind(address)
                listener.listen()
                listener.settimeout(_PROBE_TIMEOUT)
                address = listener.getsockname()
            seen = set()
            for _ in range(_PROBE_CONNECTIONS):
                with socket.socket(family, socket.SOCK_STREAM) as client:
                    client.settimeout(_PROBE_TIMEOUT)
                    client.connect(address)
                    ready, _, _ = select.select(listeners, [], [], _PROBE_TIMEOUT)
                    if not ready:
                        return False
                    with ready[0].accept()[0]:
                        seen.add(ready[0])
                if len(seen) == len(listeners):
                    return True
    except OSError:
        return False
    return False


def run_worker_listeners(app: str, *, host: str, port: int, workers: int, app_dir: str | None = None):
    """Use independent listeners when supported, otherwise share Uvicorn's socket."""
    # Reject an occupied port instead of joining another proxy's reuse group.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", socket.SO_REUSEADDR), 1)
        probe.bind((host, port))
        probe.listen()
        port = probe.getsockname()[1]

    if workers <= 1 or not supports_worker_listeners(host):
        uvicorn.run(app, host=host, port=port, workers=workers, factory=True, app_dir=app_dir)
        return

    if app_dir is not None:
        sys.path.insert(0, app_dir)

    config = _WorkerConfig(app, host=host, port=port, workers=workers, factory=True)
    # None takes Server.startup's host/port path instead of sharing an inherited
    # socket. Older Uvicorn supervisors also require an explicit server target.
    if "target" in inspect.signature(Multiprocess).parameters:
        Multiprocess(config, target=uvicorn.Server(config).run, sockets=None).run()
    else:
        Multiprocess(config, sockets=None).run()
