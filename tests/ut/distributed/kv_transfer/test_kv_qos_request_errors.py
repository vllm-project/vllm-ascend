# SPDX-License-Identifier: Apache-2.0
"""CPU contracts: actual upstream async/error mapping, without an engine/NPU.

Executes AsyncLLM.generate and streaming-error serialization from source, with
fake queue/engine submission. Uses real upstream exception and response types.
Logger and unrelated message sanitization are fixtures; no HTTP server is run.
"""

import __future__

import ast
import asyncio
import copy
import importlib.util
import json
import sys
import unittest
import uuid
from http import HTTPStatus
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[4]
VLLM = ROOT.parent / "vllm/vllm"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_node(path, owner, name, namespace):
    tree = ast.parse(path.read_text())
    body = tree.body
    if owner is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in body if getattr(n, "name", None) == name)
    unit = ast.Module(body=[node], type_ignores=[])
    code = compile(ast.fix_missing_locations(unit), str(path), "exec", flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    return namespace[name]


class RequestErrors(unittest.TestCase):
    def setUp(self):
        # Keep upstream error serialization real while isolating engine submission.
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        upstream = VLLM
        if not (upstream / "exceptions.py").is_file():
            spec = importlib.util.find_spec("vllm")
            if spec is None:
                raise RuntimeError("Pinned vLLM source or installation required")
            upstream = Path(spec.origin).parent
        for name in (
            "vllm",
            "vllm.utils",
            "vllm.v1",
            "vllm.v1.engine",
            "vllm.entrypoints",
            "vllm.entrypoints.serve",
            "vllm.entrypoints.serve.engine",
            "vllm.entrypoints.serve.exception_handling",
            "vllm_ascend",
        ):
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        sys.modules["vllm.utils"].random_uuid = lambda: uuid.uuid4().hex
        logger = ModuleType("vllm.logger")
        logger.init_logger = lambda name: MagicMock()
        sys.modules[logger.__name__] = logger
        self.errors = load("vllm.exceptions", upstream / "exceptions.py")
        self.engine_errors = load("vllm.v1.engine.exceptions", upstream / "v1/engine/exceptions.py")
        load("vllm.entrypoints.serve.engine.protocol", upstream / "entrypoints/serve/engine/protocol.py")
        utils = ModuleType("vllm.entrypoints.serve.exception_handling.utils")
        utils.sanitize_message = lambda message: message
        sys.modules[utils.__name__] = utils
        self.response = load(
            "vllm.entrypoints.serve.exception_handling.error_response",
            upstream / "entrypoints/serve/exception_handling/error_response.py",
        ).create_error_response
        load("vllm_ascend.ai_qos", ROOT / "vllm_ascend/ai_qos.py")
        self.policy = load(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos",
            ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/qos.py",
        )
        self.hook = load(
            "vllm_ascend.patch.platform.patch_ai_qos_request",
            ROOT / "vllm_ascend/patch/platform/patch_ai_qos_request.py",
        )

        class Input:
            def __init__(self):
                self.original_calls = []
                self.failure = None

            def _validate_params(self, params, supported_tasks):
                if self.failure is not None:
                    raise self.failure
                self.original_calls.append((params, supported_tasks))

        module = ModuleType("vllm.v1.engine.input_processor")
        module.InputProcessor = Input
        sys.modules[module.__name__] = module
        self.hook.install_request_validation()
        self.processor = Input()
        self.qos = dict(
            priority_to_qos={"0": 0, "3": 3, "7": 7},
            default_priority=0,
            level_names=True,
            request_priority=True,
            log=False,
        )
        self.processor.vllm_config = NS(
            kv_transfer_config=NS(get_from_extra_config=lambda key, default: self.qos if key == "kv_qos" else default)
        )

        class Output:
            finished = True

        self.Output = Output
        namespace = dict(
            asyncio=asyncio,
            logger=MagicMock(),
            RequestOutput=Output,
            STREAM_FINISHED=object(),
            VLLMClientError=self.errors.VLLMClientError,
            GracefulHTTPError=self.errors.GracefulHTTPError,
            EngineDeadError=self.engine_errors.EngineDeadError,
            EngineGenerateError=self.engine_errors.EngineGenerateError,
        )
        source_node(upstream / "v1/engine/async_llm.py", None, "InputStreamError", namespace)
        self.generate = source_node(upstream / "v1/engine/async_llm.py", "AsyncLLM", "generate", namespace)
        self.streaming_error = source_node(
            upstream / "entrypoints/generate/base/serving.py",
            "GenerateBaseServing",
            "create_streaming_error_response",
            dict(json=json, HTTPStatus=HTTPStatus),
        )
        self.submitted = []
        self.closed = []

        async def add_request(request_id, prompt, params, **kwargs):
            self.processor._validate_params(params, ("generate",))
            self.submitted.append(request_id)
            return NS(request_id=request_id, get_nowait=lambda: Output(), close=lambda: self.closed.append(request_id))

        self.engine = NS(add_request=add_request, log_requests=False)

    async def consume(self, value, request_id="test"):
        params = NS(extra_args={"kv_transfer_params": value})
        return [output async for output in self.generate(self.engine, {}, params, request_id)]

    def test_all_invalid_labels_are_client_errors_before_submission(self):
        for value in (True, 1, 2, -1, 8, "urgent", "", "7", None, [], {}):
            with self.subTest(value=value), self.assertRaises(self.errors.VLLMValidationError) as caught:
                asyncio.run(self.consume({"kv_priority": value}))
            error = self.response(caught.exception)
            self.assertEqual(error.error.code, 400)
            self.assertEqual(error.error.type, "BadRequestError")
            self.assertEqual(error.error.param, "kv_transfer_params.kv_priority")
            self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(self.submitted, [])
        self.assertEqual(self.processor.original_calls, [])

    def test_non_object_transfer_params_are_client_errors(self):
        for value in (True, [], "high", 7):
            with self.subTest(value=value), self.assertRaises(self.errors.VLLMValidationError) as caught:
                asyncio.run(self.consume(value))
            self.assertEqual(self.response(caught.exception).error.param, "kv_transfer_params")
        self.assertEqual(self.submitted, [])

    def test_valid_inputs_forward_unchanged(self):
        for value in (
            None,
            {},
            {"kv_priority": "LoW"},
            {"kv_priority": "medium"},
            {"kv_priority": "HIGH"},
            {"kv_priority": 0},
            {"kv_priority": 3},
            {"kv_priority": 7},
        ):
            original = copy.deepcopy(value)
            outputs = asyncio.run(self.consume(value))
            self.assertEqual(len(outputs), 1)
            self.assertEqual(value, original)
        self.assertEqual(len(self.submitted), 8)
        self.assertEqual(len(self.closed), 8)

    def test_disabled_request_override_retains_existing_policy(self):
        self.qos.update(request_priority=False, default_priority=3)
        self.assertEqual(len(asyncio.run(self.consume({"kv_priority": "urgent"}))), 1)
        self.assertEqual(self.policy.KvQosPolicy.from_config(self.qos).resolve_priority({"kv_priority": "urgent"}), 3)

    def test_no_qos_and_legacy_paths_are_unchanged(self):
        for qos in (None, {"level_names": False}):
            self.qos = qos
            self.assertEqual(len(asyncio.run(self.consume({"kv_priority": "urgent"}))), 1)

    def test_bad_server_policy_is_not_mislabeled_as_client_error(self):
        self.qos["default_priority"] = 5
        with self.assertRaises(self.engine_errors.EngineGenerateError) as caught:
            asyncio.run(self.consume({"kv_priority": "low"}))
        self.assertEqual(self.response(caught.exception).error.code, 500)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_original_processing_errors_still_use_server_path(self):
        original = ValueError("internal processing failure")
        self.processor.failure = original
        with self.assertRaises(self.engine_errors.EngineGenerateError) as caught:
            asyncio.run(self.consume({"kv_priority": "low"}))
        self.assertEqual(self.response(caught.exception).error.code, 500)
        self.assertIs(caught.exception.__cause__, original)

    def test_invalid_then_valid_same_engine_remains_usable(self):
        with self.assertRaises(self.errors.VLLMValidationError):
            asyncio.run(self.consume({"kv_priority": True}, "bad"))
        self.assertEqual(len(asyncio.run(self.consume({"kv_priority": "high"}, "good"))), 1)
        self.assertEqual(self.submitted, ["good"])
        self.assertEqual(self.closed, ["good"])

    def test_streaming_error_payload_uses_client_code(self):
        with self.assertRaises(self.errors.VLLMValidationError) as caught:
            asyncio.run(self.consume({"kv_priority": "urgent"}))
        body = self.streaming_error(NS(create_error_response=self.response), caught.exception)
        self.assertEqual(json.loads(body)["error"]["code"], 400)

    def test_install_is_idempotent(self):
        before = type(self.processor)._validate_params
        self.hook.install_request_validation()
        self.assertIs(type(self.processor)._validate_params, before)


if __name__ == "__main__":
    unittest.main()
