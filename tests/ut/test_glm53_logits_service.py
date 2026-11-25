# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise rendered main launcher commands and the capture HTTP extension on CPU."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from tests.e2e.common.multi_node.external_dp.external_dp_config import ExternalDPConfigLoader, RankResolver
from tests.e2e.common.multi_node.external_dp.runtime import build_all_server_commands
from tests.e2e.common.multi_node.glm53_logits.config import CASES


class TestServeCommands(unittest.TestCase):
    def test_all_cases_render_full_external_dp_without_overlapping_devices(self):
        root = Path(__file__).resolve().parents[1] / "SIT/cases/models/configs/GLM"
        for case in CASES:
            with self.subTest(case=case):
                path = root / f"GLM5_3-W8A8-A5-Logits-{case}.yaml"
                config = ExternalDPConfigLoader.from_yaml(str(path), cluster_ips=["192.0.2.1", "192.0.2.2"])
                ranks = RankResolver(config).resolve()
                commands = build_all_server_commands(config, ranks)
                self.assertEqual(len(commands), 5)
                self.assertEqual([r.visible_devices for r in ranks[1:]], ["0,1", "2,3", "4,5", "6,7"])
                p = commands[0]
                self.assertEqual(p.cmd[p.cmd.index("--prefill-context-parallel-size") + 1], "2")
                self.assertEqual(p.env.get("VLLM_PP_LAYER_PARTITION"), None if case == "disable_pp" else "4,4")
                for command in commands:
                    self.assertEqual(command.cmd[:2], ["vllm", "serve"])
                    self.assertNotIn("${", " ".join(command.cmd))
                    self.assertNotIn("enable_dsa_cp", " ".join(command.cmd))
                    self.assertIn("--no-enable-prefix-caching", command.cmd)
                    self.assertIn("--middleware", command.cmd)
                    transfer = json.loads(command.cmd[command.cmd.index("--kv-transfer-config") + 1])
                    self.assertEqual(transfer["kv_connector"], "MooncakeConnectorV2")
                    self.assertEqual(transfer["kv_connector_extra_config"]["decode"]["dp_size"], 4)
                    partition = transfer["kv_connector_extra_config"]["prefill"].get("pp_layer_partition")
                    self.assertEqual(partition, None if case == "disable_pp" else "4,4")


class TestCaptureMiddleware(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fake = ModuleType("vllm")
        fake.SamplingParams = SimpleNamespace
        with patch.dict("sys.modules", {"vllm": fake}):
            module = importlib.import_module("tests.e2e.common.multi_node.glm53_logits.middleware")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bad_trace = False

        async def generate(prompt, params, name):
            self.assertEqual(params.extra_args["regression_trace"], [1])
            self.assertEqual(prompt["prompt_token_ids"], [2, 3])
            yield SimpleNamespace(
                finished=True,
                kv_transfer_params={"do_remote_prefill": True, "remote_block_ids": [7]},
                outputs=[
                    SimpleNamespace(
                        token_ids=[0 if self.bad_trace else 1],
                        logprobs=[{i: SimpleNamespace(logprob=float(i + 1)) for i in range(4)}],
                    )
                ],
            )

        self.app = Starlette(routes=[Route("/health", lambda request: JSONResponse({"standard": True}))])
        self.app.state.engine_client = SimpleNamespace(
            generate=generate,
            engine_core=SimpleNamespace(call_utility_all_async=AsyncMock(return_value=[[{"worker_rank": 0}]])),
        )
        self.app.state.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(get_vocab_size=lambda: 4, logprobs_mode="raw_logits")
        )
        self.app.add_middleware(module.LogitsCaptureMiddleware)
        self.env = patch.dict("os.environ", {"LOG_PREFIX": self.temporary.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def test_standard_api_capture_artifact_and_completion(self):
        self.assertEqual((await self.client.get("/health")).json(), {"standard": True})
        health = (await self.client.get("/glm53_logits/health")).json()
        self.assertEqual(health["features"], [{"worker_rank": 0}])
        response = await self.client.post(
            "/glm53_logits/capture", json={"id": "sample", "prompt_ids": [2, 3], "trace": [1]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["shape"], [1, 4])
        self.assertEqual((await self.client.get("/glm53_logits/artifacts/sample.npz")).status_code, 200)
        await self.client.post("/glm53_logits/finish", json={"passed": False})
        self.assertEqual((await self.client.get("/glm53_logits/completion")).json(), {"passed": False})
        # Completion must not shut down the standard server before peer cleanup.
        self.assertEqual((await self.client.get("/health")).status_code, 200)

    async def test_bad_capture_id_cannot_write_outside_artifact_directory(self):
        response = await self.client.post("/glm53_logits/capture", json={"id": "../escape"})
        self.assertEqual(response.status_code, 400)

    async def test_wrong_trace_fails_instead_of_saving_a_capture(self):
        self.bad_trace = True
        with self.assertRaisesRegex(RuntimeError, "Trace replay"):
            await self.client.post("/glm53_logits/capture", json={"id": "sample", "prompt_ids": [2, 3], "trace": [1]})
        self.assertEqual(list(Path(self.temporary.name).rglob("*.npz")), [])
