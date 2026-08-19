# SPDX-License-Identifier: Apache-2.0

from vllm.renderers.deepseek_v4 import DeepseekV4Renderer

DEFAULT_REASONING_EFFORT = "high"


_original_apply_chat_template = DeepseekV4Renderer._apply_chat_template


# Patch reason: the request model field is a configurable served-model alias,
# so it cannot reliably identify requests rendered by DeepSeek V4.
# Patch functionality: apply DeepSeek V4 thinking defaults at its renderer,
# while preserving explicitly supplied non-null template arguments.
# Signature: matches the upstream method; no parameters are added.
def _apply_chat_template(self, *args, **kwargs):
    reasoning_effort = kwargs.get("reasoning_effort")
    if reasoning_effort is None:
        reasoning_effort = DEFAULT_REASONING_EFFORT
        kwargs["reasoning_effort"] = reasoning_effort
    if kwargs.get("thinking") is None and kwargs.get("enable_thinking") is None:
        kwargs["enable_thinking"] = reasoning_effort != "none"
    return _original_apply_chat_template(self, *args, **kwargs)


DeepseekV4Renderer._apply_chat_template = _apply_chat_template
