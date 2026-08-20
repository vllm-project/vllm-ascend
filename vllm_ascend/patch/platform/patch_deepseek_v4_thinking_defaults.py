# SPDX-License-Identifier: Apache-2.0

from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer

DEFAULT_REASONING_EFFORT = "high"


def _apply_deepseek_v4_thinking_defaults(
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    effective_kwargs = dict(chat_template_kwargs)
    reasoning_effort = effective_kwargs.get("reasoning_effort")
    if reasoning_effort is None:
        reasoning_effort = DEFAULT_REASONING_EFFORT
        effective_kwargs["reasoning_effort"] = reasoning_effort
    if effective_kwargs.get("thinking") is None and effective_kwargs.get("enable_thinking") is None:
        effective_kwargs["enable_thinking"] = reasoning_effort != "none"
    return effective_kwargs


_original_apply_chat_template = DeepseekV4Renderer._apply_chat_template


# Patch reason: the request model field is a configurable served-model alias,
# so it cannot reliably identify requests rendered by DeepSeek V4.
# Patch functionality: apply the shared DeepSeek V4 thinking defaults before
# rendering while preserving explicitly supplied non-null template arguments.
# Signature: matches the upstream method; no parameters are added.
def _apply_chat_template(self, *args, **kwargs):
    # ### PATCH START: DeepSeek V4 thinking defaults
    effective_kwargs = _apply_deepseek_v4_thinking_defaults(kwargs)
    # ### PATCH END: DeepSeek V4 thinking defaults
    return _original_apply_chat_template(
        self,
        *args,
        **effective_kwargs,
    )


_original_effective_chat_template_kwargs = OpenAIServingChat._effective_chat_template_kwargs


# Patch reason: OpenAIServingChat creates the reasoning parser before invoking
# the DeepSeek V4 renderer, so renderer-local defaults are otherwise invisible
# to both streaming and non-streaming response parsing.
# Patch functionality: return the same effective thinking kwargs used by the
# DeepSeek V4 renderer while leaving every other renderer unchanged.
# Signature: matches the upstream method; no parameters are added.
def _effective_chat_template_kwargs(self, request: ChatCompletionRequest) -> dict[str, Any]:
    chat_template_kwargs = _original_effective_chat_template_kwargs(self, request)
    # ### PATCH START: Share DeepSeek V4 defaults with parser
    if isinstance(self.renderer, DeepseekV4Renderer):
        chat_template_kwargs = _apply_deepseek_v4_thinking_defaults(chat_template_kwargs)
    # ### PATCH END: Share DeepSeek V4 defaults with parser
    return chat_template_kwargs


DeepseekV4Renderer._apply_chat_template = _apply_chat_template
OpenAIServingChat._effective_chat_template_kwargs = _effective_chat_template_kwargs
