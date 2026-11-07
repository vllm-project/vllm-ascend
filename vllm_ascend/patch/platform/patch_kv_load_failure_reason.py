# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Surface remote KV load failures as explicit, retryable client errors.

In P/D disaggregated deployments, a D-side request whose remote KV load
fails is terminated with FINISHED_ERROR and the client receives a bare
``data: [DONE]`` (streaming) or an empty 200 (non-streaming) with no
indication that the request died. This patch attributes the failure per
request, classified one-to-one by the connector via block-intersection
proof (never guessed):

* ``kv_handle_expired; please retry`` - the mooncake handle deadline had
  passed, so the pull was refused to avoid reading force-freed blocks;
* ``kv_load_failed; please retry`` - any other proven load failure (the
  exact exception is already logged per request by the receiving thread).

Two stages are combined. Engine side: ``update_from_output`` own-definers
(upstream ``Scheduler`` and ``RecomputeScheduler``; async variants
inherit the wrap via MRO) rewrite the terminating ``EngineCoreOutput``
of a proven-failed request to STOP + the reason so it survives to the
serving layer (same rendering path as the ``"recomputed"`` protocol).
The classifier (``get_kv_load_failure_reason``) is found by walking the
connector graph (wrappers such as ``MultiConnector`` neither define nor
delegate it); no-op for graphs without it.

Serving side: the streaming generators drop chunks of requests that
have not produced any token yet ("chunked prefill" filter), which
swallows the finish chunk of a zero-token KV-load failure, and the
non-streaming generators render it as an empty 200. The wraps below
turn both into an explicit error so OpenAI-compatible SDK clients
surface it verbatim instead of collapsing to an unknown finish:

* streaming (/v1/chat/completions and /v1/completions): tee the engine
  result stream to record marker stop_reasons, forward upstream SSE
  lines verbatim, and emit an OpenAI-style error chunk ``{"error":
  {"message": ..., "type": "server_error", "param": null, "code":
  "kv_handle_expired" | "kv_load_failed"}}`` in front of
  ``data: [DONE]`` - only for the markers above and only when they were
  not forwarded by the serving layer itself (self-disabling once
  upstream stops dropping them);
* non-streaming (both endpoints): when a returned response choice
  carries a marker in ``stop_reason``, return HTTP 503
  ``ServiceUnavailableError`` with the same English message.

Requests without the marker keep upstream behavior unchanged.
"""

from __future__ import annotations

import functools
import json
from http import HTTPStatus
from typing import Any

from vllm.logger import logger
from vllm.v1.engine import FinishReason


def _find_kv_load_failure_classifier(connector: Any) -> Any | None:
    """Find ``get_kv_load_failure_reason`` on the connector or its children.

    Deployments may wrap connectors, e.g. ``MultiConnector`` whose children
    are ``MooncakeConnectorV1`` + ``AscendStoreConnector``: the attribution
    API lives on the mooncake connector itself, while the wrapper neither
    defines nor delegates it (and the connector base class has no default).
    Walk the graph breadth-first (wrapper first, then children via
    ``_connectors``) and return the first real implementation.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    base_stub = getattr(KVConnectorBase_V1, "get_kv_load_failure_reason", None)
    seen: set[int] = set()
    queue: list[Any] = [connector]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        impl = getattr(type(current), "get_kv_load_failure_reason", None)
        if impl is not None and impl is not base_stub:
            return getattr(current, "get_kv_load_failure_reason")
        children = getattr(current, "_connectors", None)
        if children:
            queue.extend(children)
    return None


_NO_CLASSIFIER_WARNED = False


def _inject_kv_load_failure_reason(
    scheduler: Any,
    outputs: dict[int, Any],
    model_runner_output: Any,
) -> None:
    global _NO_CLASSIFIER_WARNED
    connector = getattr(scheduler, "connector", None)
    if connector is None:
        return
    kv_connector_output = getattr(model_runner_output, "kv_connector_output", None)
    invalid_block_ids = getattr(kv_connector_output, "invalid_block_ids", None)
    if not invalid_block_ids:
        return
    classify = _find_kv_load_failure_classifier(connector)
    if classify is None:
        # No connector in the graph supports per-request failure
        # attribution. Warn once instead of failing silently forever.
        if not _NO_CLASSIFIER_WARNED:
            _NO_CLASSIFIER_WARNED = True
            logger.warning(
                "KV load failures present (invalid_block_ids=%s) but no connector "
                "provides get_kv_load_failure_reason; finish chunks will carry "
                "no failure reason.",
                sorted(invalid_block_ids)[:8],
            )
        return
    injected: list[str] = []
    for client_outputs in outputs.values():
        for out in getattr(client_outputs, "outputs", None) or []:
            if out.finish_reason != FinishReason.ERROR:
                continue
            if out.new_token_ids or out.stop_reason:
                continue
            reason = classify(out.request_id, invalid_block_ids)
            if reason is None:
                # Not this request's failure; never guess.
                continue
            out.stop_reason = reason
            # Render through the same path as the "recomputed" protocol so
            # the reason reaches the client's finish chunk.
            out.finish_reason = FinishReason.STOP
            injected.append(out.request_id)
    if injected:
        logger.info("Injected KV load failure stop_reason for request(s): %s", injected)


def _wrap_update_from_output(cls: Any) -> bool:
    original = cls.__dict__.get("update_from_output")
    if original is None:
        # Inherits from an already-wrapped base class via MRO.
        return False

    @functools.wraps(original)
    def update_from_output(self: Any, scheduler_output: Any, model_runner_output: Any):
        outputs = original(self, scheduler_output, model_runner_output)
        try:
            _inject_kv_load_failure_reason(self, outputs, model_runner_output)
        except Exception:
            # Never let failure attribution break the scheduling loop.
            logger.exception("Failed to attribute KV load failure reasons.")
        return outputs

    cls.update_from_output = update_from_output
    return True


def _collect_scheduler_classes() -> list[Any]:
    # Only own-definers of update_from_output need wrapping; the async
    # scheduler variants inherit the wrap via MRO.
    classes: list[Any] = []
    try:
        from vllm.v1.core.sched.scheduler import Scheduler

        classes.append(Scheduler)
    except ImportError:
        logger.warning(
            "patch_kv_load_failure_reason: cannot import vllm Scheduler; wrap skipped.",
            exc_info=True,
        )
    try:
        from vllm_ascend.core.recompute_scheduler import RecomputeScheduler

        classes.append(RecomputeScheduler)
    except ImportError:
        logger.warning(
            "patch_kv_load_failure_reason: cannot import RecomputeScheduler; "
            "wrap skipped. KV load failure reasons will NOT be injected "
            "while the recompute scheduler is in use.",
            exc_info=True,
        )
    return classes


# stop_reason markers injected by the scheduler-side wrap above; the
# serving-side rescue below only ever synthesizes chunks for these.
_KV_LOAD_FAILURE_REASON_PREFIXES = ("kv_handle_expired", "kv_load_failed")


# English client-facing messages; the marker's code selects the wording.
_KV_LOAD_FAILURE_MESSAGES = {
    "kv_handle_expired": (
        "Mooncake KV transfer handle expired during prefill; the request "
        "produced no output. Please retry."
    ),
    "kv_load_failed": (
        "Remote KV cache load failed during prefill; the request produced "
        "no output. Please retry."
    ),
}


def _kv_load_failure_code(reason: str) -> str:
    """Marker code before the ``;`` separator, e.g. ``kv_handle_expired``."""
    return reason.split(";", 1)[0].strip()


def _kv_load_failure_message(reason: str) -> str:
    """English client-facing message for a marker stop_reason."""
    message = _KV_LOAD_FAILURE_MESSAGES.get(_kv_load_failure_code(reason))
    if message is None:
        message = _KV_LOAD_FAILURE_MESSAGES["kv_load_failed"]
    return message


def _kv_load_failure_error_line(reason: str, request_id: str) -> str:
    """SSE error chunk in the nested OpenAI error shape.

    AI SDK's openai-compatible chunk schema accepts a top-level ``error``
    object and forwards ``error.message`` verbatim as the stream error.
    """
    payload = {
        "error": {
            "message": (
                f"{_kv_load_failure_message(reason)} (request_id={request_id})"
            ),
            "type": "server_error",
            "param": None,
            "code": _kv_load_failure_code(reason),
        }
    }
    return f"data: {json.dumps(payload)}\n\n"


def _marker_stop_reason(output: Any) -> str | None:
    """Marker stop_reason of a terminal engine output, if any."""
    reason = output.stop_reason
    if (
        output.finish_reason is not None
        and isinstance(reason, str)
        and reason.startswith(_KV_LOAD_FAILURE_REASON_PREFIXES)
    ):
        return reason
    return None


def _response_marker_reason(response: Any) -> str | None:
    """Marker stop_reason carried by any choice of a serving response."""
    for choice in response.choices:
        reason = getattr(choice, "stop_reason", None)
        if (
            isinstance(reason, str)
            and reason.startswith(_KV_LOAD_FAILURE_REASON_PREFIXES)
        ):
            return reason
    return None


def _kv_load_failure_error_response(
    serving: Any, reason: str, request_id: str
) -> Any:
    """HTTP 503 error response carrying the same English message."""
    return serving.create_error_response(
        f"{_kv_load_failure_message(reason)} (request_id={request_id})",
        err_type="ServiceUnavailableError",
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )


async def _relay_with_error_rescue(
    lines: Any, pending: dict[int, str], request_id: str
):
    """Forward SSE lines, emitting error chunks for undelivered markers.

    ``pending`` maps choice index to the marker stop_reason recorded from
    the engine stream. A marker counts as delivered once any upstream
    line carries it (upstream may start forwarding it itself); the rest
    are emitted as error chunks in front of ``data: [DONE]``.
    """
    delivered: set[str] = set()
    async for line in lines:
        if pending:
            delivered.update(
                reason for reason in pending.values() if reason in line
            )
        if line.strip() == "data: [DONE]":
            for _, reason in sorted(pending.items()):
                if reason in delivered:
                    continue
                yield _kv_load_failure_error_line(reason, request_id)
                delivered.add(reason)
        yield line


def _wrap_completion_stream_generator() -> bool:
    """Wrap ``OpenAIServingCompletion.completion_stream_generator``.

    The generator drops chunks of requests that have not produced any
    token yet ("chunked prefill" empty-chunk filter), so the finish chunk
    of a request whose remote KV load failed before decoding a single
    token is swallowed and the client only receives ``data: [DONE]``.
    Emit an explicit error chunk in front of ``data: [DONE]`` when the
    engine-side injection marked the request with one of our KV load
    failure markers and the serving layer did not forward a chunk
    carrying it (self-disabling once upstream stops dropping it).
    """
    try:
        from vllm.entrypoints.openai.completion.serving import (
            OpenAIServingCompletion,
        )
    except Exception:
        logger.warning(
            "patch_kv_load_failure_reason: cannot import vllm completion "
            "serving classes; error chunk rescue wrap skipped.",
            exc_info=True,
        )
        return False

    original = OpenAIServingCompletion.__dict__.get("completion_stream_generator")
    if original is None or getattr(original, "_kv_load_failure_patched", False):
        return False

    @functools.wraps(original)
    async def wrapped(
        self: Any,
        request: Any,
        engine_inputs: Any,
        result_generator: Any,
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: Any,
        request_metadata: Any,
        **kwargs: Any,
    ):
        num_choices = 1 if getattr(request, "n", None) is None else request.n
        # choice index -> marker stop_reason recorded from engine outputs.
        pending: dict[int, str] = {}

        async def record_load_failures(stream: Any):
            async for prompt_idx, res in stream:
                for output in res.outputs:
                    reason = _marker_stop_reason(output)
                    if reason is not None:
                        pending[output.index + prompt_idx * num_choices] = reason
                yield prompt_idx, res

        async for line in _relay_with_error_rescue(
            original(
                self,
                request,
                engine_inputs,
                record_load_failures(result_generator),
                request_id,
                created_time,
                model_name,
                num_prompts,
                tokenizer,
                request_metadata,
                **kwargs,
            ),
            pending,
            request_id,
        ):
            yield line

    wrapped._kv_load_failure_patched = True  # type: ignore[attr-defined]
    OpenAIServingCompletion.completion_stream_generator = wrapped
    logger.info(
        "patch_kv_load_failure_reason wrapped "
        "OpenAIServingCompletion.completion_stream_generator"
    )
    return True


def _wrap_chat_stream_generator() -> bool:
    """Wrap ``OpenAIServingChat.chat_completion_stream_generator``.

    Same swallow and rescue as the completions streaming wrap above: the
    chat generator's empty-chunk filter drops the finish chunk of a
    zero-token KV-load failure, so emit an explicit error chunk in front
    of ``data: [DONE]`` for engine-injected markers that were not
    forwarded (self-disabling once upstream stops dropping them).
    """
    try:
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
    except Exception:
        logger.warning(
            "patch_kv_load_failure_reason: cannot import vllm chat serving "
            "classes; chat error chunk rescue wrap skipped.",
            exc_info=True,
        )
        return False

    original = OpenAIServingChat.__dict__.get("chat_completion_stream_generator")
    if original is None or getattr(original, "_kv_load_failure_patched", False):
        return False

    @functools.wraps(original)
    async def wrapped(
        self: Any,
        request: Any,
        result_generator: Any,
        request_id: str,
        model_name: str,
        conversation: Any,
        tokenizer: Any,
        request_metadata: Any,
        *args: Any,
        **kwargs: Any,
    ):
        # choice index -> marker stop_reason recorded from engine outputs.
        pending: dict[int, str] = {}

        async def record_load_failures(stream: Any):
            async for res in stream:
                for output in res.outputs:
                    reason = _marker_stop_reason(output)
                    if reason is not None:
                        pending[output.index] = reason
                yield res

        async for line in _relay_with_error_rescue(
            original(
                self,
                request,
                record_load_failures(result_generator),
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                *args,
                **kwargs,
            ),
            pending,
            request_id,
        ):
            yield line

    wrapped._kv_load_failure_patched = True  # type: ignore[attr-defined]
    OpenAIServingChat.chat_completion_stream_generator = wrapped
    logger.info(
        "patch_kv_load_failure_reason wrapped "
        "OpenAIServingChat.chat_completion_stream_generator"
    )
    return True


def _wrap_chat_full_generator() -> bool:
    """Wrap ``OpenAIServingChat.chat_completion_full_generator``.

    After the engine-side rewrite, a zero-token KV-load failure renders
    as an empty 200 ChatCompletionResponse whose choice carries the
    marker in ``stop_reason``. Convert it to HTTP 503 with the English
    message so non-streaming clients see the failure.
    """
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
    except Exception:
        logger.warning(
            "patch_kv_load_failure_reason: cannot import vllm chat protocol "
            "classes; chat non-stream error wrap skipped.",
            exc_info=True,
        )
        return False

    original = OpenAIServingChat.__dict__.get("chat_completion_full_generator")
    if original is None or getattr(original, "_kv_load_failure_patched", False):
        return False

    @functools.wraps(original)
    async def wrapped(
        self: Any,
        request: Any,
        result_generator: Any,
        request_id: str,
        model_name: str,
        conversation: Any,
        tokenizer: Any,
        request_metadata: Any,
        *args: Any,
        **kwargs: Any,
    ):
        response = await original(
            self,
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            *args,
            **kwargs,
        )
        if isinstance(response, ChatCompletionResponse):
            reason = _response_marker_reason(response)
            if reason is not None:
                return _kv_load_failure_error_response(self, reason, request_id)
        return response

    wrapped._kv_load_failure_patched = True  # type: ignore[attr-defined]
    OpenAIServingChat.chat_completion_full_generator = wrapped
    logger.info(
        "patch_kv_load_failure_reason wrapped "
        "OpenAIServingChat.chat_completion_full_generator"
    )
    return True


def _wrap_completion_create() -> bool:
    """Wrap ``OpenAIServingCompletion.create_completion``.

    Non-streaming counterpart of the completions rescue: convert a
    CompletionResponse whose choice carries a marker ``stop_reason`` to
    HTTP 503 with the English message.
    """
    try:
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionResponse,
        )
        from vllm.entrypoints.openai.completion.serving import (
            OpenAIServingCompletion,
        )
    except Exception:
        logger.warning(
            "patch_kv_load_failure_reason: cannot import vllm completion "
            "protocol classes; non-stream error wrap skipped.",
            exc_info=True,
        )
        return False

    original = OpenAIServingCompletion.__dict__.get("create_completion")
    if original is None or getattr(original, "_kv_load_failure_patched", False):
        return False

    @functools.wraps(original)
    async def wrapped(self: Any, *args: Any, **kwargs: Any):
        response = await original(self, *args, **kwargs)
        if isinstance(response, CompletionResponse):
            reason = _response_marker_reason(response)
            if reason is not None:
                return _kv_load_failure_error_response(
                    self, reason, response.id
                )
        return response

    wrapped._kv_load_failure_patched = True  # type: ignore[attr-defined]
    OpenAIServingCompletion.create_completion = wrapped
    logger.info(
        "patch_kv_load_failure_reason wrapped "
        "OpenAIServingCompletion.create_completion"
    )
    return True


_WRAPPED_CLASSES = tuple(
    cls.__name__ for cls in _collect_scheduler_classes() if _wrap_update_from_output(cls)
)
if _WRAPPED_CLASSES:
    logger.info("patch_kv_load_failure_reason wrapped update_from_output on %s", _WRAPPED_CLASSES)
else:
    logger.warning("patch_kv_load_failure_reason: no scheduler class defines update_from_output; patch inactive")

_wrap_completion_stream_generator()
_wrap_chat_stream_generator()
_wrap_chat_full_generator()
_wrap_completion_create()
