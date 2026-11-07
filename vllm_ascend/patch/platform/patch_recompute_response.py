# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Expose Ascend PD recomputation as an API error with a resumable payload."""

import inspect
import json
from contextlib import aclosing
from functools import wraps
from http import HTTPStatus
from typing import Any

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.logger import logger

from vllm_ascend.core.recompute_protocol import RECOMPUTE_STOP_REASON


class RecomputeErrorResponse(ErrorResponse):
    partial_response: dict[str, Any]


def _error(serving, payload):
    if not any(choice.get("stop_reason") == RECOMPUTE_STOP_REASON for choice in payload.get("choices", [])):
        return None
    error = serving.create_error_response(
        "Generation interrupted; prefill recomputation is required",
        err_type="RecomputeRequiredError",
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    return RecomputeErrorResponse(error=error.error, partial_response=payload)


async def _stream(serving, source):
    error = None
    async with aclosing(source):
        async for frame in source:
            # Native serving yields complete events, before HTTP fragmentation.
            if frame.startswith("data: ") and str(RECOMPUTE_STOP_REASON) in frame:
                error = _error(serving, json.loads(frame[6:]))
                if error is not None:
                    break
            yield frame
    if error is not None:
        yield f"data: {error.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


def _wrap_response(original, *, guard_stream=True):
    @wraps(original)
    async def wrapped(self, request, *args, **kwargs):
        response = await original(self, request, *args, **kwargs)
        if isinstance(response, ErrorResponse):
            return response
        if inspect.isasyncgen(response):
            return _stream(self, response) if guard_stream else response
        if any(choice.stop_reason == RECOMPUTE_STOP_REASON for choice in response.choices):
            return _error(self, response.model_dump(mode="json"))
        return response

    return wrapped


def _wrap_generator(original):
    signature = inspect.signature(original)

    @wraps(original)
    async def wrapped(*args, **kwargs):
        result_generator = signature.bind(*args, **kwargs).arguments["result_generator"]
        try:
            async with aclosing(original(*args, **kwargs)) as source:
                async for item in source:
                    yield item
        finally:
            await result_generator.aclose()

    return wrapped


def _wrap_chat_stream(original):
    @wraps(original)
    async def wrapped(self, request, result_generator, *args, **kwargs):
        error = None
        try:
            source = self.chat_completion_stream_generator_raw(request, result_generator, *args, **kwargs)
            async with aclosing(result_generator), aclosing(source):
                async for chunk in source:
                    if not isinstance(chunk, ErrorResponse):
                        if any(choice.stop_reason == RECOMPUTE_STOP_REASON for choice in chunk.choices):
                            error = _error(self, chunk.model_dump(mode="json", exclude_unset=True))
                            break
                    data = (
                        chunk.model_dump_json()
                        if isinstance(chunk, ErrorResponse)
                        else chunk.model_dump_json(exclude_unset=True, exclude_none=not chunk.choices)
                    )
                    yield f"data: {data}\n\n"
        except Exception as exc:
            logger.exception("Error serializing chat completion stream.")
            yield f"data: {self.create_streaming_error_response(exc)}\n\n"
        else:
            if error is not None:
                yield f"data: {error.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return wrapped


def _apply_patch():
    if getattr(OpenAIServingChat.create_chat_completion, "_ascend_recompute_response", False):
        return
    has_raw_chat = hasattr(OpenAIServingChat, "chat_completion_stream_generator_raw")
    for cls, name in (
        (OpenAIServingChat, "create_chat_completion"),
        (OpenAIServingCompletion, "create_completion"),
    ):
        wrapped = _wrap_response(getattr(cls, name), guard_stream=cls is OpenAIServingCompletion or not has_raw_chat)
        wrapped._ascend_recompute_response = True  # type: ignore[attr-defined]
        setattr(cls, name, wrapped)
    OpenAIServingChat.chat_completion_stream_generator = (_wrap_chat_stream if has_raw_chat else _wrap_generator)(
        OpenAIServingChat.chat_completion_stream_generator
    )
    OpenAIServingCompletion.completion_stream_generator = _wrap_generator(
        OpenAIServingCompletion.completion_stream_generator
    )


_apply_patch()
