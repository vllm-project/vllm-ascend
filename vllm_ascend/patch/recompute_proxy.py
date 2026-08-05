# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import copy
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

RECOMPUTE_TOKEN_IDS_KEY = "_vllm_ascend_recompute_token_ids"
_STREAM_METADATA_KEYS = ("id", "object", "created", "model")


async def iter_sse_events(chunks: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        buffer = buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            if event:
                yield event + b"\n\n"
    if buffer.strip():
        yield buffer


def _is_recomputed(payload: dict[str, Any]) -> bool:
    return any(choice.get("stop_reason") == "recomputed" for choice in payload.get("choices", []))


@dataclass
class RecomputeContext:
    stream: bool
    return_token_ids: bool
    max_tokens_field: str
    max_tokens: int
    prompt_token_ids: list[int] | None = None
    output_token_ids: list[int] = field(default_factory=list)
    retry_count: int = 0
    _message: dict[str, Any] = field(default_factory=dict)
    _stream_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> RecomputeContext:
        max_tokens_field = "max_completion_tokens" if request.get("max_completion_tokens") is not None else "max_tokens"
        context = cls(
            stream=bool(request.get("stream", False)),
            return_token_ids=bool(request.get("return_token_ids", False)),
            max_tokens_field=max_tokens_field,
            max_tokens=request.get(max_tokens_field) or 16,
        )
        request["return_token_ids"] = True
        return context

    def observe_response(self, payload: dict[str, Any]) -> bool:
        if self.stream and not self._stream_metadata:
            self._stream_metadata = {key: payload[key] for key in _STREAM_METADATA_KEYS if key in payload}

        prompt_token_ids = payload.get("prompt_token_ids")
        if prompt_token_ids is not None:
            if self.prompt_token_ids is None:
                self.prompt_token_ids = list(prompt_token_ids)
            else:
                expected_token_ids = self.prompt_token_ids + self.output_token_ids
                if prompt_token_ids != expected_token_ids:
                    raise RuntimeError("Recompute retry did not preserve the exact token history")

        choices = payload.get("choices", [])
        if choices:
            choice = choices[0]
            self.output_token_ids.extend(choice.get("token_ids") or [])
            if not self.stream:
                self._merge_message(choice.get("message") or {})

        recomputed = _is_recomputed(payload)
        if recomputed:
            self.retry_count += 1
        return recomputed

    def _merge_message(self, message: dict[str, Any]) -> None:
        for key, value in message.items():
            if value is None:
                continue
            if key == "role":
                self._message[key] = value
            elif isinstance(value, str) and isinstance(self._message.get(key), str):
                self._message[key] += value
            elif key not in self._message:
                self._message[key] = copy.deepcopy(value)

    def prepare_retry(self, request: dict[str, Any]) -> None:
        if self.prompt_token_ids is None:
            raise RuntimeError("Recompute response did not include prompt_token_ids")

        token_ids = self.prompt_token_ids + self.output_token_ids
        if "messages" in request:
            request["kv_transfer_params"] = {RECOMPUTE_TOKEN_IDS_KEY: token_ids}
        else:
            request["prompt"] = token_ids
            request.pop("kv_transfer_params", None)
        request[self.max_tokens_field] = self.max_tokens - len(self.output_token_ids)
        request["return_token_ids"] = True

    def response_for_client(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if _is_recomputed(payload):
            return None

        result = copy.deepcopy(payload)
        if self.stream:
            result.update(self._stream_metadata)
        choices = result.get("choices", [])
        if self.stream and self.retry_count and any((choice.get("delta") or {}).get("role") for choice in choices):
            return None

        if not self.stream and choices:
            choices[0]["message"] = copy.deepcopy(self._message)
            if self.return_token_ids:
                choices[0]["token_ids"] = self.output_token_ids.copy()

        if self.return_token_ids:
            if not self.stream and self.prompt_token_ids is not None:
                result["prompt_token_ids"] = self.prompt_token_ids.copy()
        else:
            result.pop("prompt_token_ids", None)
            for choice in choices:
                choice.pop("token_ids", None)

        usage = result.get("usage")
        if usage is not None and self.prompt_token_ids is not None:
            prompt_tokens = len(self.prompt_token_ids)
            completion_tokens = len(self.output_token_ids)
            usage["prompt_tokens"] = prompt_tokens
            usage["completion_tokens"] = completion_tokens
            usage["total_tokens"] = prompt_tokens + completion_tokens
        return result


def replace_rendered_chat_inputs(request: Any, result: Any) -> Any:
    kv_transfer_params = getattr(request, "kv_transfer_params", None) or {}
    token_ids = kv_transfer_params.pop(RECOMPUTE_TOKEN_IDS_KEY, None)
    if token_ids is None or not isinstance(result, tuple):
        return result

    conversation, engine_inputs = result
    if len(engine_inputs) != 1:
        return result

    engine_input = dict(engine_inputs[0])
    engine_input["type"] = "token"
    engine_input["prompt_token_ids"] = list(token_ids)
    engine_input["cache_salt"] = request.cache_salt
    engine_input.pop("prompt", None)
    return conversation, [engine_input]
