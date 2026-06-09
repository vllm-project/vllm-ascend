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
# Anthropic Messages API: backport inline system message support.
#

from __future__ import annotations

from typing import Any, Literal

from vllm.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

_ANTHROPIC_MESSAGE_ROLES = Literal["user", "assistant", "system"]

AnthropicMessage.__annotations__["role"] = _ANTHROPIC_MESSAGE_ROLES
AnthropicMessage.model_fields["role"].annotation = _ANTHROPIC_MESSAGE_ROLES
AnthropicMessage.model_rebuild(force=True)
AnthropicMessagesRequest.model_rebuild(force=True)
AnthropicCountTokensRequest.model_rebuild(force=True)


def _append_system_text(system_parts: list[str], text: str | None) -> None:
    if not text:
        return
    if text.startswith("x-anthropic-billing-header"):
        return
    system_parts.append(text)


def _append_system_content(
    system_parts: list[str],
    content: str | list[Any],
) -> None:
    if isinstance(content, str):
        _append_system_text(system_parts, content)
        return

    for block in content:
        if block.type == "text":
            _append_system_text(system_parts, block.text)


def _patched_convert_system_message(
    cls,
    anthropic_request: AnthropicMessagesRequest | AnthropicCountTokensRequest,
    openai_messages: list[dict[str, Any]],
) -> None:
    system_parts: list[str] = []

    if anthropic_request.system:
        _append_system_content(system_parts, anthropic_request.system)

    for msg in anthropic_request.messages:
        if msg.role == "system":
            _append_system_content(system_parts, msg.content)

    if system_parts:
        openai_messages.append({"role": "system", "content": "".join(system_parts)})


def _patched_convert_messages(
    cls,
    messages: list,
    openai_messages: list[dict[str, Any]],
) -> None:
    for msg in messages:
        if msg.role == "system":
            continue

        openai_msg: dict[str, Any] = {"role": msg.role}  # type: ignore

        if isinstance(msg.content, str):
            openai_msg["content"] = msg.content
        else:
            cls._convert_message_content(msg, openai_msg, openai_messages)

        if not (msg.role == "user" and "content" not in openai_msg):
            openai_messages.append(openai_msg)


AnthropicServingMessages._convert_system_message = classmethod(_patched_convert_system_message)
AnthropicServingMessages._convert_messages = classmethod(_patched_convert_messages)
