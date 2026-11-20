# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Backport vLLM #65459 for release pins predating the upstream fix.

[Role] Preserve the Anthropic Messages API minimum generation length.
[Invariants] Keep request class identity and upstream SamplingParams validation.
[Timeouts & Locks] No waits or locks; applied during platform initialization.
[Side-Effect Policy] Update only the legacy request schema and its conversion.
"""

from functools import wraps

from pydantic.fields import FieldInfo
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest


def apply_patch() -> None:
    # Native support includes both the field and forwarding (#65459). Keep it
    # untouched on main, and make repeated platform initialization a no-op.
    if "min_tokens" in AnthropicMessagesRequest.model_fields:
        return

    # Serving imports engine components; load it only when the legacy pin needs
    # the backport, after platform initialization has reached this patch.
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    # Rebuild in place so existing imports keep the same request class.
    AnthropicMessagesRequest.model_fields["min_tokens"] = FieldInfo(
        annotation=int,
        default=0,
        description="Minimum number of tokens to generate before EOS or a stop sequence can end the response.",
    )
    AnthropicMessagesRequest.model_rebuild(force=True)

    original = AnthropicServingMessages._build_base_request.__func__

    @classmethod
    @wraps(original)
    def build_base_request(cls, anthropic_request, openai_messages):
        request = original(cls, anthropic_request, openai_messages)
        if isinstance(anthropic_request, AnthropicMessagesRequest):
            request.min_tokens = anthropic_request.min_tokens
        return request

    AnthropicServingMessages._build_base_request = build_base_request


apply_patch()
