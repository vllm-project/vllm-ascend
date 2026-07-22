#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from collections.abc import AsyncGenerator
from functools import wraps
import logging

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.trace import apply_trace_patches
from vllm_ascend.trace.utils import log_action

logger = logging.getLogger(__name__)

_WRAPPED_ATTR = "__vllm_ascend_trace_wrapped__"


def _patch_request_status() -> None:
    if getattr(Request, "_vllm_ascend_trace_status_patched", False):
        return

    def status_get(self):
        if "_vllm_ascend_trace_status" in self.__dict__:
            return self.__dict__["_vllm_ascend_trace_status"]
        if "status" in self.__dict__:
            return self.__dict__["status"]
        return None

    def status_set(self, value):
        self.__dict__.pop("status", None)
        self.__dict__["_vllm_ascend_trace_status"] = value
        if value == RequestStatus.WAITING_FOR_REMOTE_KVS:
            request_id = getattr(self, "request_id", "unknown")
            log_action("Add need pulling sequence", request_id)

    Request.status = property(status_get, status_set)  # type: ignore[assignment]
    Request._vllm_ascend_trace_status_patched = True  # type: ignore[attr-defined]
    logger.info("Trace patched Request.status")


def _patch_chat_stream_generator() -> None:
    original = OpenAIServingChat.chat_completion_stream_generator
    if getattr(original, _WRAPPED_ATTR, False):
        return

    @wraps(original)
    async def trace_chat_completion_stream_generator(
        self, *args, **kwargs
    ) -> AsyncGenerator[str, None]:
        request_id = kwargs.get("request_id")
        if request_id is None and len(args) > 2:
            request_id = args[2]
        if request_id is None:
            request_id = "unknown"

        yield_count = 0
        async for item in original(self, *args, **kwargs):
            yield_count += 1
            if yield_count == 2:
                log_action("First decode output token", request_id)
            elif yield_count == 3:
                log_action("Second decode output token", request_id)
            elif yield_count == 4:
                log_action("Third decode output token", request_id)
            if item == "data: [DONE]\n\n":
                log_action("Finish decode pickle and start response", request_id)
            yield item

    setattr(trace_chat_completion_stream_generator, _WRAPPED_ATTR, True)
    OpenAIServingChat.chat_completion_stream_generator = (
        trace_chat_completion_stream_generator
    )
    logger.info("Trace patched OpenAIServingChat.chat_completion_stream_generator")


apply_trace_patches("platform")
_patch_request_status()
_patch_chat_stream_generator()
