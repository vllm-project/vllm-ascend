# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""HTTP protocol tests using synthetic arrays; these do not validate NPU inference."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import aiohttp
from aiohttp import web

from tests.e2e.common.multi_node.glm53_logits.gate import load_golden
from tests.e2e.common.multi_node.glm53_logits.runner import capture_and_compare, wait_ready
from tests.ut.glm53_logits_helpers import make_fixture


class TestLogitsProtocol(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        make_fixture(self.root)
        self.manifest, self.corpus = load_golden("disable_pp", root=self.root)
        self.entries = {(e["sample"], e["role"]): e for e in self.manifest["cases"]["disable_pp"]}
        self.bad_handoff = False
        self.requests = []
        self.runners = []
        self.urls = {}
        for role in ("P", "D"):
            app = web.Application()
            app["role"] = role
            app.router.add_post("/capture", self.capture)
            app.router.add_get("/artifacts/{name}", self.artifact)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            self.urls[role] = f"http://127.0.0.1:{port}"
            self.runners.append(runner)

    async def asyncTearDown(self):
        for runner in self.runners:
            await runner.cleanup()

    async def capture(self, request):
        data = await request.json()
        role = request.app["role"]
        self.requests.append((role, data))
        name = f"len{len(data['prompt_ids'])}"
        entry = self.entries[name, role]
        result = {"file": entry["file"], "logits_mode": "raw_logits"}
        if role == "P" and not self.bad_handoff:
            result["kv_transfer_params"] = dict(do_remote_prefill=True, remote_block_ids=[7])
        if role == "D":
            self.assertEqual(data["kv_transfer_params"]["remote_block_ids"], [7])
        return web.json_response(result)

    async def artifact(self, request):
        return web.FileResponse(self.root / "disable_pp" / request.match_info["name"])

    async def test_artifacts_are_fetched_and_compared_under_the_fixed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            async with aiohttp.ClientSession() as session:
                metrics = await capture_and_compare(
                    session, self.urls, "disable_pp", self.manifest, self.corpus, Path(directory), self.root
                )
            self.assertEqual(len(metrics), 8)
            self.assertTrue(all(m["passed"] for m in metrics))
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 8)
        self.assertEqual([r for r, _ in self.requests], ["P", "D"] * 4)
        for item, (_, p), (_, d) in zip(self.corpus, self.requests[::2], self.requests[1::2]):
            self.assertEqual(p["trace"], item["trace"][:1])
            self.assertEqual(d["trace"], item["trace"])
            self.assertEqual(p["prompt_ids"], d["prompt_ids"])

    async def test_missing_handoff_fails_before_decode(self):
        self.bad_handoff = True
        with tempfile.TemporaryDirectory() as directory:
            async with aiohttp.ClientSession() as session:
                with self.assertRaisesRegex(RuntimeError, "KV handoff"):
                    await capture_and_compare(
                        session, self.urls, "disable_pp", self.manifest, self.corpus, Path(directory), self.root
                    )
        self.assertEqual(len(self.requests), 1)

    async def test_dead_server_fails_immediately(self):
        process = Mock()
        process.poll.return_value = 1
        async with aiohttp.ClientSession() as session:
            with self.assertRaisesRegex(RuntimeError, "server exited"):
                await wait_ready(session, self.urls, process, 10)

    async def test_readiness_timeout_is_a_failure(self):
        process = Mock()
        process.poll.return_value = None
        async with aiohttp.ClientSession() as session:
            with self.assertRaises(TimeoutError):
                await wait_ready(session, self.urls, process, 0)


if __name__ == "__main__":
    unittest.main()
