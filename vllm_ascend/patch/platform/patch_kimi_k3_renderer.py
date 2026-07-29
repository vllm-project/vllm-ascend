#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

"""Render Kimi K3 chat prompts with its tokenizer-owned Python encoder.

Kimi K3 deliberately does not publish a Jinja chat template. Its trusted
remote ``TikTokenTokenizer.apply_chat_template`` implements the XTML protocol,
including typed tool calls, reasoning controls, and multimodal placeholders.
The regular HF renderer rejects tokenizers without a Jinja template, so K3
uses a dedicated explicit ``kimi_k3`` mode while reusing the standard HF
tokenizer loader.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.connector import merge_media_io_kwargs
from vllm.renderers import registry as renderer_registry
from vllm.renderers.base import BaseRenderer
from vllm.renderers.inputs import DictPrompt
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.params import ChatParams
from vllm.tokenizers import TokenizerRegistry
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

KIMI_K3_RENDERER_MODE = "kimi_k3"
KIMI_K3_IMAGE_PROMPT = "<|kimi_image_placeholder|>"
KIMI_K3_PROMPT_TOOL_CHOICE_KEY = "_kimi_k3_prompt_tool_choice"
_KIMI_K3_PROMPT_TOOL_CHOICE_PREFIX = "kimi_k3:"
_KIMI_K3_PROMPT_TOOL_CHOICES = frozenset({"none", "auto", "required"})
_ORIGINAL_RENDER_CHAT_ATTR = "_ascend_original_kimi_k3_render_chat"
_ORIGINAL_EFFECTIVE_KWARGS_ATTR = "_ascend_original_kimi_k3_effective_chat_template_kwargs"
_PREPARED_ATTR = "_kimi_k3_chat_params_prepared"
_K3_MEDIA_IO_DEFAULTS: dict[str, dict[str, Any]] = {
    "image": {"image_mode": None},
}
_K3_THINKING_EFFORTS = ("low", "high", "max")

_REASONING_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def encode_kimi_k3_prompt_tool_choice(tool_choice: str) -> str:
    if tool_choice not in _KIMI_K3_PROMPT_TOOL_CHOICES:
        raise ValueError(f"Unsupported Kimi K3 prompt tool choice: {tool_choice!r}.")
    return _KIMI_K3_PROMPT_TOOL_CHOICE_PREFIX + tool_choice


def decode_kimi_k3_prompt_tool_choice(encoded_choice: str) -> str:
    if not encoded_choice.startswith(_KIMI_K3_PROMPT_TOOL_CHOICE_PREFIX):
        raise ValueError("Malformed Kimi K3 prompt tool choice.")
    tool_choice = encoded_choice[len(_KIMI_K3_PROMPT_TOOL_CHOICE_PREFIX) :]
    if tool_choice not in _KIMI_K3_PROMPT_TOOL_CHOICES:
        raise ValueError(f"Unsupported Kimi K3 prompt tool choice: {tool_choice!r}.")
    return tool_choice


def _merge_k3_media_io_kwargs(
    media_io_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    return merge_media_io_kwargs(_K3_MEDIA_IO_DEFAULTS, media_io_kwargs)


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _apply_k3_thinking_kwargs(kwargs: dict[str, Any]) -> None:
    if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
        kwargs.setdefault("thinking", enable_thinking)

    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort == "none":
        kwargs.setdefault("thinking", False)
    elif reasoning_effort is not None:
        kwargs.setdefault("thinking_effort", reasoning_effort)

    thinking_effort = kwargs.get("thinking_effort")
    if thinking_effort is not None and thinking_effort not in _K3_THINKING_EFFORTS:
        supported = ", ".join(_K3_THINKING_EFFORTS)
        raise VLLMValidationError(
            f"Kimi K3 supports thinking_effort values: {supported}",
            parameter="thinking_effort",
            value=thinking_effort,
        )


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("name")
        return getattr(function, "name", None)
    return getattr(getattr(tool, "function", None), "name", None)


def _named_tool_choice(request: ChatCompletionRequest) -> str | None:
    choice = request.tool_choice
    function = choice.get("function") if isinstance(choice, dict) else getattr(choice, "function", None)
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


def prepare_kimi_k3_chat_template_kwargs(request: ChatCompletionRequest) -> None:
    """Map typed OpenAI controls to K3's tokenizer-owned encoder."""

    if getattr(request, _PREPARED_ATTR, False):
        return

    user_kwargs = request.chat_template_kwargs or {}
    template_kwargs = dict(user_kwargs)
    fields_set: set[str] = getattr(request, "model_fields_set", set())

    duplicate_controls = {
        "reasoning_effort",
        "response_format",
        "tool_choice",
        "tools",
    }.intersection(user_kwargs)
    if duplicate_controls:
        parameter = sorted(duplicate_controls)[0]
        raise VLLMValidationError(
            f"Kimi K3 {parameter} must use the standard OpenAI request field, not chat_template_kwargs.",
            parameter=parameter,
        )

    native_thinking = user_kwargs.get("thinking")
    if "thinking" in user_kwargs and not isinstance(native_thinking, bool):
        raise VLLMValidationError(
            "Kimi K3 chat_template_kwargs.thinking must be a boolean.",
            parameter="reasoning_effort",
        )
    if "reasoning_effort" in fields_set and "thinking" in user_kwargs:
        typed_thinking = request.reasoning_effort != "none"
        if native_thinking != typed_thinking:
            raise VLLMValidationError(
                "Kimi K3 reasoning_effort conflicts with chat_template_kwargs.thinking.",
                parameter="reasoning_effort",
            )

    native_effort = user_kwargs.get("thinking_effort")
    if native_effort is not None and native_effort not in {"low", "high", "max"}:
        raise VLLMValidationError(
            f"Unsupported Kimi K3 thinking_effort: {native_effort!r}.",
            parameter="reasoning_effort",
        )
    if native_effort is not None and native_thinking is False:
        raise VLLMValidationError(
            "Kimi K3 thinking_effort requires thinking=true.",
            parameter="reasoning_effort",
        )
    if "reasoning_effort" in fields_set and native_effort is not None:
        typed_effort = _REASONING_EFFORT_MAP.get(request.reasoning_effort)
        if native_effort != typed_effort:
            raise VLLMValidationError(
                "Kimi K3 reasoning_effort conflicts with chat_template_kwargs.thinking_effort.",
                parameter="reasoning_effort",
            )
    if "reasoning_effort" not in fields_set:
        if native_thinking is False:
            request.reasoning_effort = "none"
        elif native_effort is not None:
            request.reasoning_effort = native_effort

    if "tool_choice" not in fields_set:
        request.tool_choice = "auto" if request.tools else "none"

    request_tools = [_model_dump(tool) for tool in (request.tools or [])]

    if "thinking" not in user_kwargs and request.reasoning_effort is not None:
        template_kwargs["thinking"] = request.reasoning_effort != "none"
    if (
        "thinking_effort" not in user_kwargs
        and template_kwargs.get("thinking", True)
        and request.reasoning_effort in _REASONING_EFFORT_MAP
    ):
        template_kwargs["thinking_effort"] = _REASONING_EFFORT_MAP[request.reasoning_effort]

    template_kwargs["tools"] = request_tools

    named_tool = _named_tool_choice(request)
    if named_tool:
        matching_tools = [tool for tool in request_tools if _tool_name(tool) == named_tool]
        if not matching_tools:
            raise VLLMValidationError(
                f"Named Kimi K3 tool choice {named_tool!r} is not declared.",
                parameter="tool_choice",
            )
        template_kwargs["tool_choice"] = "required"
        template_kwargs["tools"] = matching_tools
    else:
        if isinstance(request.tool_choice, str):
            template_kwargs["tool_choice"] = request.tool_choice
        elif request.tool_choice is None:
            template_kwargs["tool_choice"] = "auto" if template_kwargs.get("tools") else "none"

    prompt_tool_choice = template_kwargs.get("tool_choice")
    if isinstance(prompt_tool_choice, str) and prompt_tool_choice in _KIMI_K3_PROMPT_TOOL_CHOICES:
        template_kwargs[KIMI_K3_PROMPT_TOOL_CHOICE_KEY] = encode_kimi_k3_prompt_tool_choice(prompt_tool_choice)

    if request.response_format is not None:
        template_kwargs["response_format"] = _model_dump(request.response_format)

    request.chat_template_kwargs = template_kwargs
    request.skip_special_tokens = False
    request.spaces_between_special_tokens = False
    object.__setattr__(request, _PREPARED_ATTR, True)


def _normalize_developer_messages(
    conversation: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Convert developer roles without reordering or flattening their content."""

    converted: list[ConversationMessage] = []
    for message in conversation:
        if message["role"] == "developer":
            converted_message = dict(message)
            converted_message["role"] = "system"
            converted_message.pop("tools", None)
            converted.append(converted_message)  # type: ignore[arg-type]
        else:
            converted.append(message)
    return converted


def _normalize_k3_tool_messages(
    conversation: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Derive K3 tool-result metadata and restore assistant call order."""

    normalized: list[ConversationMessage] = []
    index = 0
    while index < len(conversation):
        message = conversation[index]
        normalized.append(dict(message))  # type: ignore[arg-type]
        index += 1

        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue

        targets_by_id: dict[str, tuple[int, str]] = {}
        for position, tool_call in enumerate(tool_calls):
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            target = (position, name)
            aliases = [f"{name}:{position}"]
            if tool_call_id := tool_call.get("id"):
                aliases.insert(0, str(tool_call_id))
            for alias in aliases:
                targets_by_id.setdefault(alias, target)

        block_start = index
        while index < len(conversation) and conversation[index].get("role") == "tool":
            index += 1
        if index == block_start:
            continue

        resolved: list[tuple[int, int, ConversationMessage]] = []
        for original_order, tool_message in enumerate(conversation[block_start:index]):
            tool_call_id = tool_message.get("tool_call_id")
            resolved_target = targets_by_id.get(str(tool_call_id)) if tool_call_id is not None else None
            if resolved_target is None:
                normalized.extend(
                    dict(item)  # type: ignore[arg-type]
                    for item in conversation[block_start:index]
                )
                break

            position, name = resolved_target
            enriched = dict(tool_message)
            enriched["tool"] = name
            enriched["index"] = position + 1
            resolved.append(
                (position, original_order, enriched)  # type: ignore[arg-type]
            )
        else:
            resolved.sort(key=lambda item: (item[0], item[1]))
            normalized.extend(item for _, _, item in resolved)

    return normalized


class KimiK3Renderer(BaseRenderer[HfTokenizer]):
    """Renderer that delegates the complete chat protocol to K3's tokenizer."""

    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)
        self._apply_chat_template_async = make_async(
            self._apply_chat_template,
            executor=self._executor,
        )

    def _apply_chat_template(
        self,
        conversation_data: list[ConversationMessage],
        **kwargs: Any,
    ) -> list[int]:
        # K3's Python encoder is the source of truth. In particular, do not let
        # an optional server/request Jinja value replace or filter its kwargs.
        prompt_tool_choice = kwargs.pop(KIMI_K3_PROMPT_TOOL_CHOICE_KEY, None)
        _apply_k3_thinking_kwargs(kwargs)
        for protected_key in (
            "add_generation_prompt",
            "chat_template",
            "continue_final_message",
            "conversation",
            "enable_thinking",
            "image_prompts",
            "max_length",
            "padding",
            "reasoning_effort",
            "return_dict",
            "return_tensors",
            "tokenize",
            "truncation",
        ):
            kwargs.pop(protected_key, None)
        if prompt_tool_choice is not None:
            # vLLM 0.23 treats the literal string ``auto`` as an unset value
            # while merging ChatParams defaults. The private typed key survives
            # that merge so a server default cannot turn an auto request into
            # a required/none prompt.
            kwargs["tool_choice"] = decode_kimi_k3_prompt_tool_choice(prompt_tool_choice)
        prompt = self.get_tokenizer().apply_chat_template(
            conversation=conversation_data,
            tokenize=True,
            add_generation_prompt=True,
            padding=False,
            truncation=False,
            return_tensors=None,
            return_dict=False,
            **kwargs,
        )
        if not isinstance(prompt, list) or any(not isinstance(token_id, int) for token_id in prompt):
            raise TypeError("Kimi K3 tokenizer must return a flat list of token IDs.")
        return prompt

    def _render_conversation(
        self,
        conversation: list[ConversationMessage],
        mm_data,
        mm_uuids,
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation = _normalize_developer_messages(conversation)
        conversation = _normalize_k3_tool_messages(conversation)
        prompt_raw = self._apply_chat_template(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )
        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids
        return conversation, prompt

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(
                params.media_io_kwargs,
            ),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        return self._render_conversation(
            conversation,
            mm_data,
            mm_uuids,
            params,
        )

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(
                params.media_io_kwargs,
            ),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        conversation = _normalize_developer_messages(conversation)
        conversation = _normalize_k3_tool_messages(conversation)

        prompt_raw = await self._apply_chat_template_async(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )
        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt


if KIMI_K3_RENDERER_MODE not in TokenizerRegistry.tokenizers:
    TokenizerRegistry.register(
        KIMI_K3_RENDERER_MODE,
        "vllm.tokenizers.hf",
        "CachedHfTokenizer",
    )

if KIMI_K3_RENDERER_MODE not in renderer_registry.RENDERER_REGISTRY.renderers:
    renderer_registry.RENDERER_REGISTRY.register(
        KIMI_K3_RENDERER_MODE,
        __name__,
        "KimiK3Renderer",
    )


# Prepare K3-only controls at the single request-aware render entry used by
# both serving and the standalone render API.
if not hasattr(OnlineRenderer, _ORIGINAL_RENDER_CHAT_ATTR):
    setattr(
        OnlineRenderer,
        _ORIGINAL_RENDER_CHAT_ATTR,
        OnlineRenderer.render_chat,
    )


@wraps(getattr(OnlineRenderer, _ORIGINAL_RENDER_CHAT_ATTR))
async def _render_chat_with_kimi_k3_params(
    self: OnlineRenderer,
    request: ChatCompletionRequest,
    *,
    skip_mm_cache: bool = False,
):
    if isinstance(self.renderer, KimiK3Renderer) and isinstance(request, ChatCompletionRequest):
        prepare_kimi_k3_chat_template_kwargs(request)

    original = getattr(type(self), _ORIGINAL_RENDER_CHAT_ATTR)
    return await original(self, request, skip_mm_cache=skip_mm_cache)


OnlineRenderer.render_chat = _render_chat_with_kimi_k3_params


if not hasattr(OpenAIServingChat, _ORIGINAL_EFFECTIVE_KWARGS_ATTR):
    setattr(
        OpenAIServingChat,
        _ORIGINAL_EFFECTIVE_KWARGS_ATTR,
        OpenAIServingChat._effective_chat_template_kwargs,
    )


@wraps(getattr(OpenAIServingChat, _ORIGINAL_EFFECTIVE_KWARGS_ATTR))
def _effective_chat_template_kwargs_with_kimi_k3_params(
    self: OpenAIServingChat,
    request: ChatCompletionRequest,
) -> dict[str, Any]:
    if isinstance(self.renderer, KimiK3Renderer):
        prepare_kimi_k3_chat_template_kwargs(request)

    original = getattr(type(self), _ORIGINAL_EFFECTIVE_KWARGS_ATTR)
    effective_kwargs = original(self, request)
    if isinstance(self.renderer, KimiK3Renderer):
        prompt_tool_choice = effective_kwargs.get(KIMI_K3_PROMPT_TOOL_CHOICE_KEY)
        if prompt_tool_choice is not None:
            effective_kwargs["tool_choice"] = decode_kimi_k3_prompt_tool_choice(prompt_tool_choice)
    return effective_kwargs


OpenAIServingChat._effective_chat_template_kwargs = _effective_chat_template_kwargs_with_kimi_k3_params
