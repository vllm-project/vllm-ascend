# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in ASGI entry point: --middleware ...frontend.FastEEPMiddleware."""

import asyncio
import json
import time
from types import MethodType

from starlette.datastructures import Headers
from vllm.logger import logger

from vllm_ascend.distributed.fasteep.policy import validate_target


async def respond(send, status, payload):
    body = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


class FastEEPMiddleware:
    """Serialize resize, drain complete streams, and queue new inference calls.

    One API process owns this gate. Mutation continues after caller cancellation;
    a worker error keeps admission closed until the instance is restarted.
    """

    def __init__(self, app):
        self.app = app
        self.condition = asyncio.Condition()
        self.resize_lock = asyncio.Lock()
        self.closed = False
        self.failed = False
        self.inflight = 0
        self.operation = None
        self.client = None
        self.active = None
        self.authentication = None
        inner = app
        while inner is not None:
            if type(inner).__name__ == "AuthenticationMiddleware":
                self.authentication = inner
                break
            inner = getattr(inner, "app", None)

    async def rpc(self, method, **kwargs):
        values = await self.client.engine_core.call_utility_all_async("collective_rpc", method, 240, (), kwargs)
        return [value for rank in values for value in rank]

    async def initialize(self, scope):
        if self.client is not None:
            return
        client = scope["app"].state.engine_client
        engine = client.engine_core
        if getattr(engine, "client_count", None) != 1 or not hasattr(engine, "engine_inflight"):
            raise ValueError("FastEEP requires one Python API process with internal DP load balancing.")
        self.client = client
        try:
            states = await self.rpc("fasteep_state")
            self.maximum = len(engine.core_engines)
            self.check_states(states, self.maximum)
        except BaseException:
            self.client = None
            raise
        self.active = self.maximum
        original = engine.get_core_engine_for_request

        def route(engine, request):
            # Delegate affinity, counters, and abort bookkeeping to upstream.
            rank = request.data_parallel_rank
            if rank is not None and not 0 <= rank < self.active:
                raise ValueError("Requested DP rank is inactive.")
            if rank is None and self.active < self.maximum:
                request.data_parallel_rank = min(
                    range(self.active), key=lambda i: engine.engine_inflight[engine.core_engines[i]]
                )
            try:
                return original(request)
            finally:
                request.data_parallel_rank = rank

        engine.get_core_engine_for_request = MethodType(route, engine)

    def check_states(self, states, target):
        if len(states) != self.maximum or sorted(s["rank"] for s in states) != list(range(self.maximum)):
            raise RuntimeError("Incomplete FastEEP worker state.")
        if any(s["active"] != target or s["failed"] or s["sleeping"] != (s["rank"] >= target) for s in states):
            raise RuntimeError("FastEEP workers did not agree on the committed topology.")

    async def resize(self, scope, target, timeout):
        async with self.resize_lock:
            if self.failed:
                return 503, {"error": "Previous resize failed; restart this instance."}
            start = time.monotonic()
            paused = False
            # Worker RPCs may take multiple scheduling steps. Close admission
            # before awaiting them so arriving requests cannot enter that gap.
            async with self.condition:
                self.closed = True
            try:
                try:
                    await self.initialize(scope)
                    validate_target(target, self.maximum)
                    await self.rpc("fasteep_validate", target=target)
                except ValueError as error:
                    return 400, {"error": str(error)}
                if target == self.active:
                    return 200, {"active": target, "changed": False}
                async with self.condition:
                    await asyncio.wait_for(self.condition.wait_for(lambda: self.inflight == 0), timeout)
                if self.client.engine_core.streaming_reqs:
                    return 409, {"error": "Close streaming-input sessions before resizing."}
                # Keep every scheduler alive; only workers leave the active group.
                paused = True
                await asyncio.wait_for(self.client.pause_generation(mode="wait", clear_cache=True), timeout)
                states = await self.rpc("fasteep_resize", target=target)
                self.check_states(states, target)
                self.active = target
                await self.client.resume_generation()
                return 200, {"active": target, "changed": True, "seconds": time.monotonic() - start}
            except asyncio.TimeoutError:
                self.failed = paused
                return 504, {"error": "Resize timed out."}
            except BaseException:
                self.failed = paused
                raise
            finally:
                async with self.condition:
                    self.closed = self.failed
                    self.condition.notify_all()

    @staticmethod
    def log_failure(task):
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("FastEEP resize failed", exc_info=error)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path in {"/scale_elastic_ep", "/sleep", "/wake_up", "/collective_rpc", "/pause", "/resume"}:
            return await respond(send, 409, {"error": "This instance uses retained FastEEP scaling."})
        if path == "/fasteep_scale_ep" and scope["method"] == "POST":
            if self.authentication is not None and not self.authentication.verify_token(Headers(scope=scope)):
                return await respond(send, 401, {"error": "Unauthorized"})
            body = bytearray()
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                body.extend(event.get("body", b""))
                if len(body) > 16384:
                    return await respond(send, 413, {"error": "Resize request is too large."})
                if not event.get("more_body", False):
                    break
            try:
                payload = json.loads(body)
                target = payload["new_data_parallel_size"]
                timeout = payload.get("drain_timeout", 120)
                if type(timeout) is not int or timeout <= 0:
                    raise ValueError("drain_timeout must be a positive integer.")
            except (ValueError, KeyError, TypeError) as error:
                return await respond(send, 400, {"error": str(error)})
            # Keep a strong reference when the HTTP handler is cancelled.
            if self.operation is not None and not self.operation.done():
                return await respond(send, 409, {"error": "A resize is already in progress."})
            self.operation = asyncio.create_task(self.resize(scope, target, timeout))
            self.operation.add_done_callback(self.log_failure)
            try:
                status, result = await asyncio.shield(self.operation)
            except Exception:
                status, result = 500, {"error": "Resize failed; inspect server logs and restart this instance."}
            return await respond(send, status, result)
        inference = path == "/invocations" or path.startswith(("/v1/", "/v2/", "/inference", "/cohere"))
        if inference and scope["method"] == "POST":
            async with self.condition:
                await self.condition.wait_for(lambda: not self.closed or self.failed)
                if self.failed:
                    return await respond(send, 503, {"error": "FastEEP is unavailable after a failed resize."})
                self.inflight += 1
            try:
                return await self.app(scope, receive, send)
            finally:
                async with self.condition:
                    self.inflight -= 1
                    self.condition.notify_all()
        return await self.app(scope, receive, send)
