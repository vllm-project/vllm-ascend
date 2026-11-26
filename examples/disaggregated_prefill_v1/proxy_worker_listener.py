# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Independent Linux listeners for the sample proxy's Uvicorn workers."""

import asyncio
import inspect
import socket

import uvicorn
from uvicorn.supervisors import Multiprocess

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


def run_worker_listeners(app: str, *, host: str, port: int, workers: int):
    """Run independently bound listeners with Uvicorn's worker supervision."""
    # Reject an occupied port instead of joining another proxy's reuse group.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        port = probe.getsockname()[1]

    config = _WorkerConfig(app, host=host, port=port, workers=workers, factory=True)
    # None takes Server.startup's host/port path instead of sharing an inherited
    # socket. Older Uvicorn supervisors also require an explicit server target.
    if "target" in inspect.signature(Multiprocess).parameters:
        Multiprocess(config, target=uvicorn.Server(config).run, sockets=None).run()
    else:
        Multiprocess(config, sockets=None).run()
