# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostics for unexpected prefill on a P/D decode endpoint."""

import json
import os
import queue
import tempfile
import threading
import time
import uuid

import msgspec
from vllm.logger import logger

RAW_BODY_KEY = "_hybrid_prefill_capture_raw_body"
ENDPOINT_KEY = "_hybrid_prefill_capture_endpoint"
CAPTURE_ENDPOINTS = frozenset({"/v1/messages", "/v1/chat/completions", "/v1/completions"})
MAX_PENDING_CAPTURES = 16


class CaptureRequestBodyMiddleware:
    """Carry the original D HTTP body through the existing vllm_xargs path."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in CAPTURE_ENDPOINTS:
            return await self.app(scope, receive, send)
        messages = []
        chunks = []
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        try:
            original = body.decode("utf-8")
            payload = json.loads(original)
            extra = payload.get("vllm_xargs")
            if extra is None:
                extra = {}
            if not isinstance(extra, dict):
                raise ValueError("Invalid vllm_xargs; leave validation to the API")
            payload["vllm_xargs"] = {**extra, RAW_BODY_KEY: original, ENDPOINT_KEY: scope["path"]}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (UnicodeDecodeError, ValueError, AttributeError):
            # Preserve invalid input and the API's normal validation behavior.
            pass
        else:
            if messages[-1]["type"] != "http.disconnect":
                messages = [{"type": "http.request", "body": body, "more_body": False}]
                scope = dict(scope)
                scope["headers"] = [
                    (key, value) for key, value in scope.get("headers", []) if key.lower() != b"content-length"
                ] + [(b"content-length", str(len(body)).encode())]

        async def replay_receive():
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)


class PrefillCaptureWriter:
    """Snapshot CPU request state; perform bounded, asynchronous file writes."""

    def __init__(self, directory, engine_id):
        self.directory = os.path.abspath(directory)
        self.engine_id = engine_id
        self.pending = queue.Queue(maxsize=MAX_PENDING_CAPTURES)
        self.thread = threading.Thread(target=self._run, name="hybrid-prefill-capture", daemon=True)
        self.thread.start()

    def submit(self, request, num_scheduled_tokens, scheduled_prompt_tokens, admission_flags):
        try:
            sampling = msgspec.to_builtins(request.sampling_params)
            extra = sampling.get("extra_args") or {}
            original = extra.pop(RAW_BODY_KEY, None)
            endpoint = extra.pop(ENDPOINT_KEY, None)
            snapshot = {
                "schema_version": 1,
                "event": "multi_token_prefill_scheduled_on_d",
                "captured_at_unix_ns": time.time_ns(),
                "engine_id": self.engine_id,
                "request_id": request.request_id,
                "endpoint": endpoint,
                "status": request.status.name,
                "admission_flags": dict(admission_flags),
                "kv_transfer_params_at_capture": msgspec.to_builtins(request.kv_transfer_params),
                "prompt_token_ids": list(request.prompt_token_ids) if request.prompt_token_ids is not None else None,
                "output_token_ids": list(request.output_token_ids),
                "sampling_params": sampling,
                "num_prompt_tokens": request.num_prompt_tokens,
                "num_computed_tokens": request.num_computed_tokens,
                "num_in_flight_tokens": request.num_in_flight_tokens,
                "num_preemptions": request.num_preemptions,
                "num_scheduled_tokens": num_scheduled_tokens,
                "scheduled_prompt_tokens": scheduled_prompt_tokens,
                "remaining_prompt_tokens": max(request.num_prompt_tokens - request.num_computed_tokens, 0),
                "original_request_available": isinstance(original, str),
                "has_prompt_embeds": request.prompt_embeds is not None,
                "has_multimodal_features": bool(request.mm_features),
            }
            self.pending.put_nowait((snapshot, original))
        except Exception as exc:
            # Diagnostics must not fail generation or silently claim success.
            logger.warning(
                "[HybridDPrefillCapture] event=capture_failed request_id=%s reason=%s",
                request.request_id,
                type(exc).__name__,
            )

    def _run(self):
        while True:
            snapshot, original = self.pending.get()
            try:
                self._write(snapshot, original)
            except Exception as exc:
                logger.warning(
                    "[HybridDPrefillCapture] event=write_failed request_id=%s reason=%s",
                    snapshot["request_id"],
                    type(exc).__name__,
                )
            finally:
                self.pending.task_done()

    def _write(self, snapshot, original):
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        temporary = tempfile.mkdtemp(prefix=".partial-", dir=self.directory)
        with open(os.path.join(temporary, "engine.json"), "w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2)
        if isinstance(original, str):
            with open(os.path.join(temporary, "request.json"), "wb") as stream:
                stream.write(original.encode("utf-8"))
        destination = os.path.join(self.directory, "capture-" + uuid.uuid4().hex)
        os.rename(temporary, destination)
        logger.warning(
            "[HybridDPrefillCapture] event=saved request_id=%s computed_tokens=%d "
            "remaining_prompt_tokens=%d scheduled_prompt_tokens=%d original_request_available=%s path=%s",
            snapshot["request_id"],
            snapshot["num_computed_tokens"],
            snapshot["remaining_prompt_tokens"],
            snapshot["scheduled_prompt_tokens"],
            snapshot["original_request_available"],
            destination,
        )
