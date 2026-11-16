# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Backport streaming ``kv_transfer_params`` to vLLM entrypoints.

Upstream vLLM 0.30.0 only includes ``kv_transfer_params`` in non-streaming
API responses.  The disaggregated P/D proxy reads this field from the
**decode** response to cache D-side KV metadata for bidirectional transfer
in subsequent turns.  Without it the proxy can only capture the metadata in
non-streaming mode, breaking multi-turn streaming conversations.

This patch mirrors upstream PR #40186 (``[entrypoints/openai] add kv
params to streaming api response``):

1. Adds ``kv_transfer_params`` to ``CompletionStreamResponse`` and
   ``ChatCompletionStreamResponse`` protocol models.
2. Wraps the streaming generators so the **terminal chunk** (the one with
   ``finish_reason`` set) carries ``kv_transfer_params`` from the final
   ``RequestOutput``.

Approach: the original generator is wrapped.  Each yielded SSE line is
parsed; when the chunk has a ``finish_reason``, the wrapper injects
``kv_transfer_params`` from a thread-local stash that is populated by
patching ``RequestOutput`` to publish its ``kv_transfer_params`` when
``last_res = res`` runs inside the generator.

Remove this patch once the supported vLLM version contains PR #40186.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from vllm.entrypoints.openai.completion.protocol import CompletionStreamResponse
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import (
    OpenAIServingChat,
)
from vllm.logger import logger

_PATCHED = False

# Thread-local stash: the most recent RequestOutput.kv_transfer_params seen
# by the patched generator.  Populated by intercepting RequestOutput access.
_kv_stash = threading.local()


def _get_stashed_kv() -> dict[str, Any] | None:
    return getattr(_kv_stash, "kv_transfer_params", None)


def _clear_stash() -> None:
    _kv_stash.kv_transfer_params = None


def _ensure_stream_response_field(response_cls: type) -> None:
    """Add ``kv_transfer_params`` field to a stream-response protocol model."""
    if "kv_transfer_params" in response_cls.model_fields:
        return
    from pydantic.fields import FieldInfo

    response_cls.model_fields["kv_transfer_params"] = FieldInfo(
        annotation=dict[str, Any] | None,
        default=None,
    )
    response_cls.__annotations__["kv_transfer_params"] = dict[str, Any] | None
    try:
        response_cls.model_rebuild(force=True)
    except Exception:
        logger.warning(
            "Failed to rebuild %s after adding kv_transfer_params field",
            response_cls.__name__,
        )


def _inject_kv_into_sse_line(raw: str) -> str:
    """Parse an SSE ``data: ...`` line, inject kv_transfer_params if terminal.

    ``raw`` may contain one or more SSE events concatenated together.
    We split on ``data: `` boundaries, process each event, and rejoin.
    """
    if "data: " not in raw or "data: [DONE]" in raw:
        return raw

    # Split into individual SSE events on "data: " boundaries.
    parts = raw.split("data: ")
    result_parts = []
    for part in parts:
        if not part:
            continue
        part = part.strip()
        if part == "[DONE]":
            result_parts.append("data: [DONE]\n\n")
            continue
        try:
            chunk = json.loads(part)
        except (json.JSONDecodeError, TypeError):
            result_parts.append(f"data: {part}\n\n")
            continue

        is_terminal = any(
            choice.get("finish_reason") is not None
            for choice in chunk.get("choices", [])
        )
        is_usage_chunk = not chunk.get("choices") and chunk.get("usage") is not None

        kv = _get_stashed_kv()
        if (is_terminal or is_usage_chunk) and kv is not None:
            chunk["kv_transfer_params"] = kv

        new_payload = json.dumps(chunk, ensure_ascii=False)
        result_parts.append(f"data: {new_payload}\n\n")

    return "".join(result_parts)


def _patch_chat_completion_stream_generator() -> None:
    """Wrap ``chat_completion_stream_generator``.

    The wrapper intercepts each yielded SSE line.  When the line is a
    terminal chunk (finish_reason set) or the trailing usage chunk, it
    injects ``kv_transfer_params`` from the thread-local stash.

    The stash is populated by patching ``RequestOutput`` so that when the
    generator reads ``res.kv_transfer_params`` the value is also stored
    in the thread-local.
    """
    if getattr(
        OpenAIServingChat.chat_completion_stream_generator,
        "_vllm_ascend_streaming_kv_patched",
        False,
    ):
        return

    original = OpenAIServingChat.chat_completion_stream_generator

    async def _patched(self, *args, **kwargs):
        _clear_stash()
        async for raw in original(self, *args, **kwargs):
            if "data: " in raw:
                injected = _inject_kv_into_sse_line(raw)
                yield injected
            else:
                yield raw

    _patched._original = original  # type: ignore[attr-defined]
    _patched._vllm_ascend_streaming_kv_patched = True  # type: ignore[attr-defined]
    OpenAIServingChat.chat_completion_stream_generator = _patched
    logger.info("Patched chat_completion_stream_generator for kv_transfer_params")


def _patch_completion_stream_generator() -> None:
    """Wrap ``completion_stream_generator`` similarly."""
    if getattr(
        OpenAIServingCompletion.completion_stream_generator,
        "_vllm_ascend_streaming_kv_patched",
        False,
    ):
        return

    original = OpenAIServingCompletion.completion_stream_generator

    async def _patched(self, *args, **kwargs):
        _clear_stash()
        async for raw in original(self, *args, **kwargs):
            if "data: " in raw:
                injected = _inject_kv_into_sse_line(raw)
                yield injected
            else:
                yield raw

    _patched._original = original  # type: ignore[attr-defined]
    _patched._vllm_ascend_streaming_kv_patched = True  # type: ignore[attr-defined]
    OpenAIServingCompletion.completion_stream_generator = _patched
    logger.info("Patched completion_stream_generator for kv_transfer_params")


def _patch_request_output_kv_stash() -> None:
    """Patch ``RequestOutput.kv_transfer_params`` property to stash values.

    The streaming generators read ``res.kv_transfer_params`` (or
    ``last_res.kv_transfer_params``) internally.  By wrapping the property
    getter we capture the value into the thread-local stash so the outer
    generator wrapper can inject it into the terminal chunk.
    """
    from vllm.outputs import RequestOutput

    if getattr(RequestOutput, "_vllm_ascend_kv_stash_patched", False):
        return

    original_setter = None
    # RequestOutput.kv_transfer_params is a plain attribute (set in __init__
    # and update from next_output).  We cannot wrap a property because it
    # is a simple attribute.  Instead, patch __setattr__ to intercept
    # assignments to ``kv_transfer_params``.
    original_setattr = RequestOutput.__setattr__

    def _patched_setattr(self, name, value):
        original_setattr(self, name, value)
        if name == "kv_transfer_params" and value is not None:
            _kv_stash.kv_transfer_params = value

    RequestOutput.__setattr__ = _patched_setattr
    RequestOutput._vllm_ascend_kv_stash_patched = True  # type: ignore[attr-defined]
    logger.info("Patched RequestOutput.__setattr__ for kv_transfer_params stash")


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # Feature flag: only patch when bidirectional KV transfer is explicitly
    # enabled. The patch intercepts every streaming chunk (split + re-serialize)
    # and wraps RequestOutput.__setattr__, which would add overhead and modify
    # SSE serialization for non-P/D deployments that never read the field.
    enabled = os.getenv("VLLM_ASCEND_BIDIRECTIONAL_KV_XFER", "0") == "1" or os.getenv(
        "VLLM_ASCEND_BIDIRECTIONAL_KV_XFER"
    ) in ("true", "True", "1")
    if not enabled:
        logger.debug(
            "Bidirectional KV transfer disabled; skipping streaming "
            "kv_transfer_params patch (backport of PR #40186)."
        )
        return

    _ensure_stream_response_field(CompletionStreamResponse)
    _ensure_stream_response_field(ChatCompletionStreamResponse)
    _patch_request_output_kv_stash()
    _patch_chat_completion_stream_generator()
    _patch_completion_stream_generator()

    logger.info(
        "Streaming kv_transfer_params patch applied (backport of PR #40186)."
    )


_apply_patch()
