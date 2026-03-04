#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Patch for handling the ``thinking`` API parameter for GLM models.

When a GLM-series model (GLM-4.5, GLM-5, etc.) is deployed, users can
control the thinking mode through the ``thinking`` field in the
Chat Completion API request::

    "thinking": {"type": "disabled"}   # non-thinking mode
    "thinking": {"type": "enabled"}    # thinking mode (default)

Without this patch the ``thinking`` field is silently ignored because
``ChatCompletionRequest.build_chat_params`` only processes known Pydantic
fields. As a result:

1. GLM5 still generates a full ``<think>reasoning</think>answer`` block.
2. The ``<think>`` start token is filtered out by
   ``skip_special_tokens=True``.
3. The ``</think>`` token and reasoning content remain visible in
   ``content``, violating the user's intent.

This patch maps:

* ``thinking: {type: "disabled"}`` -> ``enable_thinking: False`` in
  ``chat_template_kwargs`` so that the GLM chat template generates no
  ``<think>`` block.
* ``thinking: {type: "enabled"}`` -> ``enable_thinking: True`` in
  ``chat_template_kwargs`` (explicit opt-in).

A user-supplied ``enable_thinking`` inside ``chat_template_kwargs`` is
always respected and takes precedence over the ``thinking`` field.
"""

from dataclasses import is_dataclass, replace
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import \
    ChatCompletionRequest
from vllm.renderers import ChatParams, merge_kwargs

_original_build_chat_params = ChatCompletionRequest.build_chat_params


def _get_request_extra(request: "ChatCompletionRequest") -> dict[str, Any]:
    """Merge extra request fields from the available Pydantic accessors."""
    merged_extra: dict[str, Any] = {}
    pydantic_extra = getattr(request, "__pydantic_extra__", None)
    model_extra = getattr(request, "model_extra", None)
    if isinstance(pydantic_extra, dict):
        merged_extra.update(pydantic_extra)
    if isinstance(model_extra, dict):
        merged_extra.update(model_extra)
    return merged_extra


def _patched_build_chat_params(
    self: "ChatCompletionRequest",
    default_template: "str | None",
    default_template_content_format: Any,
) -> "ChatParams":
    """Patched :meth:`ChatCompletionRequest.build_chat_params`.

    Maps the ``thinking`` API extra-field to ``enable_thinking`` inside
    ``chat_template_kwargs`` so that GLM-family models honour the requested
    thinking mode when rendering their chat template.
    """
    original_result: ChatParams = _original_build_chat_params(
        self, default_template, default_template_content_format
    )

    # ``thinking`` is an extra (non-declared) field on the Pydantic model.
    # With ``model_config = ConfigDict(extra="allow")`` Pydantic v2 may store
    # extra fields in ``model_extra`` or ``__pydantic_extra__`` depending on
    # how the request object was constructed.
    request_extra = _get_request_extra(self)
    thinking = request_extra.get("thinking")

    if not isinstance(thinking, dict):
        # ``thinking`` was not provided or is not a recognised format.
        return original_result

    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        enable_thinking = False
    elif thinking_type == "enabled":
        enable_thinking = True
    else:
        # Unknown thinking type; leave the original result unchanged.
        return original_result

    # Respect an explicitly set ``enable_thinking`` from ``chat_template_kwargs``
    # (user always wins).
    existing_kwargs: dict[str, Any] = original_result.chat_template_kwargs or {}
    if "enable_thinking" in existing_kwargs:
        return original_result

    updated_kwargs = merge_kwargs(
        existing_kwargs,
        {"enable_thinking": enable_thinking},
    )

    model_copy = getattr(original_result, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"chat_template_kwargs": updated_kwargs})

    if is_dataclass(original_result):
        return replace(original_result, chat_template_kwargs=updated_kwargs)

    return ChatParams(
        chat_template=original_result.chat_template,
        chat_template_content_format=original_result.chat_template_content_format,
        chat_template_kwargs=updated_kwargs,
    )


# Apply the monkey-patch.
ChatCompletionRequest.build_chat_params = _patched_build_chat_params  # type: ignore[method-assign]
