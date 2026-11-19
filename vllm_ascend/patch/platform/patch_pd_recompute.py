"""Carry opted-in scheduler interruptions through native API serializers."""

import json
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from functools import wraps
from typing import Any

from openai.types.responses import ResponseErrorEvent
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.responses import serving as responses_serving
from vllm.exceptions import GenerationError
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from vllm_ascend.core.pd_recompute import (
    PD_CONTINUE_MESSAGE,
    PD_RECOMPUTE_ARG,
    PD_RECOMPUTE_INTERRUPTED,
    PD_RECOMPUTE_RESUME,
    PD_RECOMPUTE_RETRY,
    PD_RECOMPUTE_VERSION,
)


class PDRecomputeError(GenerationError):
    """The decode scheduler ended an opted-in request to recompute its KV."""

    def __init__(self, signal: str, progress: dict | None = None):
        super().__init__(f"{signal}:{json.dumps(progress)}" if progress else signal)
        self.signal = signal


async def _check_pd_recompute(
    outputs: AsyncGenerator[RequestOutput, None], *, return_partial: bool
) -> AsyncGenerator[RequestOutput, None]:
    """Serialize partial JSON natively and interrupt SSE with a private error."""
    async with aclosing(outputs):
        async for output in outputs:
            transfer = output.kv_transfer_params or {}
            signal = transfer.get(PD_RECOMPUTE_ARG)
            if output.finished and signal in (PD_RECOMPUTE_RETRY, PD_RECOMPUTE_INTERRUPTED):
                if signal == PD_RECOMPUTE_INTERRUPTED:
                    if return_partial and "remaining_tokens" in transfer:
                        for completion in output.outputs:
                            completion.finish_reason = "length"
                            completion.stop_reason = None
                        yield output
                        return
                    for completion in output.outputs:
                        completion.finish_reason = None
                        completion.stop_reason = None
                    output.finished = False
                    output.kv_transfer_params = None
                    if any(completion.token_ids or completion.text for completion in output.outputs):
                        yield output
                progress = transfer if "remaining_tokens" in transfer else None
                raise PDRecomputeError(signal, progress)
            yield output


def _wrap_generate(
    generate: Callable[..., AsyncGenerator[RequestOutput, None]],
) -> Callable[..., AsyncGenerator[RequestOutput, None]]:
    @wraps(generate)
    def generate_with_pd_recompute(self, prompt, sampling_params, *args, **kwargs):
        outputs = generate(self, prompt, sampling_params, *args, **kwargs)
        extra_args = sampling_params.extra_args or {}
        if extra_args.get(PD_RECOMPUTE_ARG) == PD_RECOMPUTE_VERSION:
            resume = extra_args.get(PD_RECOMPUTE_RESUME) == PD_RECOMPUTE_VERSION
            return _check_pd_recompute(
                outputs,
                return_partial=resume and sampling_params.output_kind == RequestOutputKind.FINAL_ONLY,
            )
        return outputs

    return generate_with_pd_recompute


def _wrap_responses_streaming_error(convert_error: Callable[..., str]) -> Callable[..., str]:
    @wraps(convert_error)
    def convert_with_pd_recompute(self, error):
        if isinstance(error, PDRecomputeError):
            return ResponseErrorEvent(
                type="error", code=error.signal, message=str(error), param=None, sequence_number=-1
            ).model_dump_json()
        return convert_error(self, error)

    return convert_with_pd_recompute


def _wrap_messages_request(convert: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(convert)
    def convert_with_continuation(cls, *args, **kwargs):
        request = convert(cls, *args, **kwargs)
        if (request.vllm_xargs or {}).get(PD_CONTINUE_MESSAGE) == PD_RECOMPUTE_VERSION:
            request.add_generation_prompt = False
            request.continue_final_message = True
        return request

    return convert_with_continuation


def _wrap_messages_full_converter(convert: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(convert)
    def convert_with_recompute(self, response):
        try:
            return convert(self, response)
        except json.JSONDecodeError as exc:
            transfer = response.kv_transfer_params or {}
            if transfer.get(PD_RECOMPUTE_ARG) == PD_RECOMPUTE_INTERRUPTED:
                raise PDRecomputeError(PD_RECOMPUTE_INTERRUPTED, transfer) from exc
            raise

    return convert_with_recompute


def _install(async_llm: type[Any], responses_module: Any) -> None:
    async_llm.generate = _wrap_generate(async_llm.generate)
    responses_module.StreamingResponsesResponse |= ResponseErrorEvent
    serving = responses_module.OpenAIServingResponses
    serving._convert_generation_error_to_streaming_response = _wrap_responses_streaming_error(
        serving._convert_generation_error_to_streaming_response
    )


_install(AsyncLLM, responses_serving)
AnthropicServingMessages.to_chat_completion_request = classmethod(
    _wrap_messages_request(AnthropicServingMessages.to_chat_completion_request.__func__)
)
AnthropicServingMessages.messages_full_converter = _wrap_messages_full_converter(
    AnthropicServingMessages.messages_full_converter
)
