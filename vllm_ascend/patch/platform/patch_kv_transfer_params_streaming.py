#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
from collections.abc import AsyncGenerator, AsyncIterator
from functools import wraps
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.outputs import RequestOutput

_PATCHED_ATTR = "_ascend_kv_transfer_params_stream_patched"
_DONE_CHUNK = "data: [DONE]\n\n"

_original_stream_generator = OpenAIServingChat.chat_completion_stream_generator


def _inject_kv_transfer_params(
    chunk: str,
    kv_transfer_params: dict[str, Any],
) -> str:
    """Add ``kv_transfer_params`` to a single SSE ``data:`` payload."""
    prefix = "data: "
    if not chunk.startswith(prefix):
        return chunk
    try:
        payload = json.loads(chunk[len(prefix) :].rstrip("\n"))
    except json.JSONDecodeError:
        return chunk
    if not isinstance(payload, dict):
        return chunk
    payload["kv_transfer_params"] = kv_transfer_params
    return f"{prefix}{json.dumps(payload, ensure_ascii=False)}\n\n"


@wraps(_original_stream_generator)
async def _patched_chat_completion_stream_generator(
    self: OpenAIServingChat,
    request: ChatCompletionRequest,
    result_generator: AsyncIterator[RequestOutput],
    request_id: str,
    model_name: str,
    *args: Any,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Emit ``kv_transfer_params`` on the final streaming chunk.

    Upstream vLLM only sets ``kv_transfer_params`` on the non-streaming
    ``ChatCompletionResponse``; the streaming path never emits it, even though
    the value (e.g. the UCM prefetch handle produced on the decode node) is
    needed by clients that consume chunks incrementally.

    The value is produced by the scheduler when the request finishes, so it is
    only available on the last ``RequestOutput``. We therefore observe the
    result stream, and once it is drained, inject the captured value into the
    final ``data:`` chunk (the one right before ``data: [DONE]``). Buffering is
    limited to a single chunk, so token-by-token streaming is preserved.
    """
    captured: dict[str, Any] = {}

    async def _tracking_result_generator() -> AsyncIterator[RequestOutput]:
        async for res in result_generator:
            kv_transfer_params = getattr(res, "kv_transfer_params", None)
            if kv_transfer_params is not None:
                captured["kv_transfer_params"] = kv_transfer_params
            yield res

    pending_data: str | None = None
    done_chunk: str | None = None

    async for chunk in _original_stream_generator(
        self,
        request,
        _tracking_result_generator(),
        request_id,
        model_name,
        *args,
        **kwargs,
    ):
        if chunk == _DONE_CHUNK:
            done_chunk = chunk
            continue
        if pending_data is not None:
            yield pending_data
        pending_data = chunk

    kv_transfer_params = captured.get("kv_transfer_params")
    if pending_data is not None:
        if kv_transfer_params is not None:
            pending_data = _inject_kv_transfer_params(pending_data, kv_transfer_params)
        yield pending_data

    yield done_chunk if done_chunk is not None else _DONE_CHUNK


if not getattr(OpenAIServingChat, _PATCHED_ATTR, False):
    OpenAIServingChat.chat_completion_stream_generator = _patched_chat_completion_stream_generator
    setattr(OpenAIServingChat, _PATCHED_ATTR, True)
