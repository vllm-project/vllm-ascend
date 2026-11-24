# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End native PD requests on preemption before partial output is serialized."""

from contextlib import aclosing
from functools import wraps

from openai.types.responses import ResponseErrorEvent
from vllm.entrypoints.openai.responses import serving as responses_serving
from vllm.exceptions import GenerationError
from vllm.v1.engine.async_llm import AsyncLLM

from vllm_ascend.core.pd_recompute import PD_FAIL_ON_RECOMPUTE


class PDRecomputeError(GenerationError):
    """The decode request was preempted and cannot continue this native response."""


async def _fail_on_recompute(outputs):
    async with aclosing(outputs):
        async for output in outputs:
            if output.finished and any(
                item.finish_reason == "error" and item.stop_reason == "recomputed" for item in output.outputs
            ):
                raise PDRecomputeError("Decode request was preempted. Retry the request if it is safe to repeat.")
            yield output


def _wrap_generate(generate):
    @wraps(generate)
    def generate_with_preemption_error(self, prompt, sampling_params, *args, **kwargs):
        outputs = generate(self, prompt, sampling_params, *args, **kwargs)
        if (sampling_params.extra_args or {}).get(PD_FAIL_ON_RECOMPUTE):
            return _fail_on_recompute(outputs)
        return outputs

    return generate_with_preemption_error


def _wrap_streaming_error(convert):
    @wraps(convert)
    def convert_preemption_error(self, error):
        if isinstance(error, PDRecomputeError):
            return ResponseErrorEvent(
                type="error", code="pd_preempted", message=str(error), param=None, sequence_number=-1
            ).model_dump_json()
        return convert(self, error)

    return convert_preemption_error


AsyncLLM.generate = _wrap_generate(AsyncLLM.generate)
# Upstream validates errors against this union before emitting an SSE event.
responses_serving.StreamingResponsesResponse |= ResponseErrorEvent
responses_serving.OpenAIServingResponses._convert_generation_error_to_streaming_response = _wrap_streaming_error(
    responses_serving.OpenAIServingResponses._convert_generation_error_to_streaming_response
)
