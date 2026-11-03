# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm.sampling_params import SamplingParams
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p import hybrid_prefill_capture as capture


class TestCaptureRequestBodyMiddleware(unittest.IsolatedAsyncioTestCase):
    async def _request(self, body, endpoint="/v1/messages"):
        messages = [
            {"type": "http.request", "body": body[:7], "more_body": True},
            {"type": "http.request", "body": body[7:], "more_body": False},
        ]
        received = []
        output_scope = {}

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def app(scope, receive, send):
            output_scope.update(scope)
            while True:
                message = await receive()
                received.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break

        async def send(message):
            pass

        await capture.CaptureRequestBodyMiddleware(app)(
            {
                "type": "http",
                "method": "POST",
                "path": endpoint,
                "headers": [(b"content-length", str(len(body)).encode())],
            },
            receive,
            send,
        )
        result = b"".join(received)
        self.assertEqual(int(dict(output_scope["headers"])[b"content-length"]), len(result))
        return result

    async def test_original_body_and_existing_extensions_are_preserved(self):
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "long input " * 10000}],
                "max_tokens": 32000,
                "stream": True,
                "vllm_xargs": {"existing": 7},
            },
            indent=2,
        ).encode()
        for endpoint in capture.CAPTURE_ENDPOINTS:
            with self.subTest(endpoint=endpoint):
                amended = json.loads(await self._request(body, endpoint))
                self.assertEqual(amended["vllm_xargs"].pop(capture.RAW_BODY_KEY).encode(), body)
                self.assertEqual(amended["vllm_xargs"].pop(capture.ENDPOINT_KEY), endpoint)
                self.assertEqual(amended, json.loads(body))

    async def test_invalid_and_unrelated_requests_are_unchanged(self):
        for body in (b"invalid json", b"[]", b'{"vllm_xargs": 1}'):
            with self.subTest(body=body):
                self.assertEqual(await self._request(body), body)
        self.assertEqual(await self._request(b"{}", "/tokenize"), b"{}")


class TestPrefillCaptureWriter(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        # Test snapshots and writes deterministically without starting a daemon.
        with patch.object(capture.threading.Thread, "start"):
            self.writer = capture.PrefillCaptureWriter(self.directory.name, "test-d")
        self.original = '{\n "model": "test", "max_tokens": 32000\n}'
        self.request = SimpleNamespace(
            request_id="request/with/slashes",
            status=RequestStatus.RUNNING,
            sampling_params=SamplingParams(
                max_tokens=32000, extra_args={capture.RAW_BODY_KEY: self.original, capture.ENDPOINT_KEY: "/v1/messages"}
            ),
            kv_transfer_params=None,
            prompt_token_ids=list(range(100)),
            output_token_ids=[],
            num_prompt_tokens=100,
            num_computed_tokens=0,
            num_in_flight_tokens=0,
            num_preemptions=0,
            prompt_embeds=None,
            mm_features=[],
        )

    def test_snapshot_is_independent_and_original_body_is_byte_exact(self):
        self.writer.submit(self.request, 64, 64, {"has_kv_params": False})
        snapshot, original = self.writer.pending.get_nowait()
        self.request.prompt_token_ids[0] = 999
        self.assertEqual(snapshot["prompt_token_ids"], list(range(100)))
        self.assertEqual(self.request.sampling_params.extra_args[capture.RAW_BODY_KEY], self.original)
        self.assertNotIn(capture.RAW_BODY_KEY, snapshot["sampling_params"]["extra_args"])
        self.writer._write(snapshot, original)
        folders = list(Path(self.directory.name).glob("capture-*"))
        self.assertEqual(len(folders), 1)
        self.assertEqual((folders[0] / "request.json").read_bytes(), self.original.encode())
        self.assertEqual(json.loads((folders[0] / "engine.json").read_text()), snapshot)
        self.assertFalse(list(Path(self.directory.name).glob(".partial-*")))

    def test_missing_middleware_is_reported_without_fake_original(self):
        self.request.sampling_params.extra_args = {}
        self.writer.submit(self.request, 2, 2, {})
        snapshot, original = self.writer.pending.get_nowait()
        self.assertFalse(snapshot["original_request_available"])
        self.writer._write(snapshot, original)
        folder = next(Path(self.directory.name).glob("capture-*"))
        self.assertFalse((folder / "request.json").exists())

    def test_full_queue_reports_failure_without_interrupting_generation(self):
        for _ in range(capture.MAX_PENDING_CAPTURES):
            self.writer.pending.put_nowait(None)
        with patch.object(capture, "logger") as logger:
            self.writer.submit(self.request, 2, 2, {})
        self.assertEqual(self.writer.pending.qsize(), capture.MAX_PENDING_CAPTURES)
        self.assertIn("event=capture_failed", logger.warning.call_args.args[0])
        self.assertEqual(logger.warning.call_args.args[-1], "Full")

    def test_serialization_failure_is_reported(self):
        self.request.sampling_params.extra_args = {"unsupported": object()}
        with patch.object(capture, "logger") as logger:
            self.writer.submit(self.request, 2, 2, {})
        self.assertTrue(self.writer.pending.empty())
        self.assertIn("event=capture_failed", logger.warning.call_args.args[0])
