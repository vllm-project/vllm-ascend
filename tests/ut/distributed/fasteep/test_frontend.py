# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import asyncio
import json
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

from vllm_ascend.distributed.fasteep.frontend import FastEEPMiddleware


class TestAdmission(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.events = []

        async def app(scope, receive, send):
            self.events.append("inference")
            self.entered.set()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"first", "more_body": True})
            await self.release.wait()
            await send({"type": "http.response.body", "body": b"last"})

        self.gate = FastEEPMiddleware(app)
        self.engine = SimpleNamespace(
            client_count=1, core_engines=[0, 1, 2], engine_inflight=Counter(), streaming_reqs=set()
        )
        self.engine.get_core_engine_for_request = lambda request: request.data_parallel_rank
        self.engine.call_utility_all_async = AsyncMock(side_effect=self.rpc)
        self.client = SimpleNamespace(
            engine_core=self.engine, pause_generation=AsyncMock(), resume_generation=AsyncMock()
        )
        self.scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/completions",
            "app": SimpleNamespace(state=SimpleNamespace(engine_client=self.client)),
        }
        self.target = 3
        self.mutation_entered = asyncio.Event()
        self.mutation_release = asyncio.Event()
        self.mutation_release.set()
        self.fail = False

    async def rpc(self, operation, method, timeout, args, kwargs):
        if method == "fasteep_validate":
            return [[True]] * 3
        if method == "fasteep_resize":
            self.mutation_entered.set()
            await self.mutation_release.wait()
            self.events.append("resize")
            if self.fail:
                raise RuntimeError("native failure")
            self.target = kwargs["target"]
        return [
            [{"rank": rank, "active": self.target, "maximum": 3, "failed": False, "sleeping": rank >= self.target}]
            for rank in range(3)
        ]

    async def request(self, path, payload=None):
        scope = dict(self.scope, path=path)
        sent = []
        receive = AsyncMock(return_value={"type": "http.request", "body": json.dumps(payload).encode()})
        await self.gate(scope, receive, AsyncMock(side_effect=sent.append))
        return sent

    async def test_stream_drains_and_arrivals_wait(self):
        first = asyncio.create_task(self.request("/v1/completions"))
        await self.entered.wait()
        resize = asyncio.create_task(self.request("/fasteep_scale_ep", {"new_data_parallel_size": 2}))
        for _ in range(20):
            if self.gate.closed:
                break
            await asyncio.sleep(0)
        self.assertTrue(self.gate.closed)
        second = asyncio.create_task(self.request("/v1/chat/completions"))
        await asyncio.sleep(0)
        self.assertEqual(self.events, ["inference"])
        self.release.set()
        await asyncio.gather(first, resize, second)
        self.assertEqual(self.events, ["inference", "resize", "inference"])
        self.client.pause_generation.assert_awaited_once_with(mode="wait", clear_cache=True)
        self.assertEqual(self.gate.active, 2)
        self.assertEqual(self.gate.inflight, 0)

    async def test_invalid_requests_do_not_mutate(self):
        for payload in (
            {},
            [],
            {"new_data_parallel_size": True},
            {"new_data_parallel_size": 4},
            {"new_data_parallel_size": 2, "drain_timeout": 0},
        ):
            result = await self.request("/fasteep_scale_ep", payload)
            self.assertEqual(result[0]["status"], 400)
        self.assertNotIn("resize", self.events)
        self.assertFalse(self.gate.closed)

    async def test_invocations_wait_for_admission(self):
        self.gate.closed = True
        request = asyncio.create_task(self.request("/invocations"))
        await asyncio.sleep(0)
        self.assertFalse(self.entered.is_set())
        async with self.gate.condition:
            self.gate.closed = False
            self.gate.condition.notify_all()
        await self.entered.wait()
        self.assertEqual(self.gate.inflight, 1)
        self.release.set()
        await request
        self.assertEqual(self.gate.inflight, 0)

    async def test_arrival_waits_during_worker_validation(self):
        validating = asyncio.Event()
        release_validation = asyncio.Event()

        async def delayed_rpc(*args, **kwargs):
            if args[1] == "fasteep_validate":
                validating.set()
                await release_validation.wait()
            return await self.rpc(*args, **kwargs)

        self.engine.call_utility_all_async.side_effect = delayed_rpc
        resize = asyncio.create_task(self.request("/fasteep_scale_ep", {"new_data_parallel_size": 2}))
        await validating.wait()
        arriving = asyncio.create_task(self.request("/v1/completions"))
        try:
            await asyncio.sleep(0)
            self.assertFalse(self.entered.is_set())
        finally:
            release_validation.set()
            self.release.set()
            await asyncio.gather(resize, arriving)
        self.assertEqual(self.events, ["resize", "inference"])

    async def test_failure_stays_closed(self):
        self.fail = True
        response = await self.request("/fasteep_scale_ep", {"new_data_parallel_size": 2})
        self.assertEqual(response[0]["status"], 500)
        response = await self.request("/v1/completions")
        self.assertEqual(response[0]["status"], 503)
        self.client.resume_generation.assert_not_awaited()

    async def test_cancelled_caller_cannot_reopen_during_mutation(self):
        self.mutation_release.clear()
        task = asyncio.create_task(self.request("/fasteep_scale_ep", {"new_data_parallel_size": 2}))
        await self.mutation_entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.gate.closed)
        self.mutation_release.set()
        await self.gate.operation
        self.assertFalse(self.gate.closed)
        self.assertEqual(self.gate.active, 2)

    async def test_concurrent_resize_rejected(self):
        self.mutation_release.clear()
        first = asyncio.create_task(self.request("/fasteep_scale_ep", {"new_data_parallel_size": 2}))
        await self.mutation_entered.wait()
        result = await self.request("/fasteep_scale_ep", {"new_data_parallel_size": 3})
        self.assertEqual(result[0]["status"], 409)
        self.mutation_release.set()
        await first

    async def test_drain_timeout_reopens_without_mutation(self):
        self.gate.inflight = 1
        status, _ = await self.gate.resize(self.scope, 2, 0.001)
        self.assertEqual(status, 504)
        self.assertFalse(self.gate.closed)
        self.assertNotIn("resize", self.events)

    async def test_rank_selection_delegates_and_restores_request(self):
        await self.gate.initialize(self.scope)
        self.gate.active = 2
        self.engine.engine_inflight[0] = 4
        request = SimpleNamespace(data_parallel_rank=None)
        self.assertEqual(self.engine.get_core_engine_for_request(request), 1)
        self.assertIsNone(request.data_parallel_rank)
        request.data_parallel_rank = 2
        with self.assertRaises(ValueError):
            self.engine.get_core_engine_for_request(request)

    async def test_rejects_missing_rank_receipt(self):
        await self.gate.initialize(self.scope)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            self.gate.check_states([{"rank": 0}], 2)

    async def test_streaming_session_blocks_mutation(self):
        self.engine.streaming_reqs.add("open-session")
        result = await self.request("/fasteep_scale_ep", {"new_data_parallel_size": 2})
        self.assertEqual(result[0]["status"], 409)
        self.assertFalse(self.gate.closed)
        self.assertNotIn("resize", self.events)

    async def test_management_authentication(self):
        self.gate.authentication = SimpleNamespace(
            verify_token=lambda headers: headers.get("Authorization") == "Bearer test"
        )
        self.scope["headers"] = []
        result = await self.request("/fasteep_scale_ep", {"new_data_parallel_size": 2})
        self.assertEqual(result[0]["status"], 401)
        self.engine.call_utility_all_async.assert_not_awaited()
        self.scope["headers"] = [(b"authorization", b"Bearer test")]
        result = await self.request("/fasteep_scale_ep", {"new_data_parallel_size": 2})
        self.assertEqual(result[0]["status"], 200)

    async def test_single_frontend_and_native_endpoint_guard(self):
        self.engine.client_count = 2
        with self.assertRaises(ValueError):
            await self.gate.initialize(self.scope)
        response = await self.request("/scale_elastic_ep", {"new_data_parallel_size": 2})
        self.assertEqual(response[0]["status"], 409)
