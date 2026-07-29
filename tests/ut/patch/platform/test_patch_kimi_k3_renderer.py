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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.exceptions import VLLMValidationError
from vllm.renderers import registry as renderer_registry
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.params import ChatParams
from vllm.tokenizers import TokenizerRegistry
from vllm.tokenizers.hf import CachedHfTokenizer

from vllm_ascend.patch.platform import patch_kimi_k3_renderer as renderer_patch
from vllm_ascend.patch.platform.patch_kimi_k3_renderer import (
    KIMI_K3_PROMPT_TOOL_CHOICE_KEY,
    KimiK3Renderer,
    decode_kimi_k3_prompt_tool_choice,
)


def _model_config(model_type: str):
    return SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))


def _request(**kwargs):
    defaults = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "help"}],
        "reasoning_effort": "high",
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def test_explicit_kimi_k3_mode_registers_renderer_and_hf_loader():
    assert renderer_registry.RENDERER_REGISTRY.load_renderer_cls("kimi_k3") is KimiK3Renderer
    assert TokenizerRegistry.load_tokenizer_cls("kimi_k3") is CachedHfTokenizer
    assert renderer_registry.tokenizer_args_from_config.__module__ == "vllm.tokenizers.registry"
    assert (
        OpenAIServingChat._effective_chat_template_kwargs.__module__
        == "vllm.entrypoints.openai.chat_completion.serving"
    )


def test_renderer_calls_tokenizer_python_encoder_without_jinja():
    calls = []

    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            calls.append(kwargs)
            return [11, 12]

    renderer = object.__new__(KimiK3Renderer)
    renderer.tokenizer = RecordingTokenizer()
    conversation = [{"role": "user", "content": "hello"}]

    prompt = renderer._apply_chat_template(
        conversation,
        add_generation_prompt=False,
        chat_template="{{ should_not_run }}",
        continue_final_message=True,
        conversation=[{"role": "user", "content": "injected"}],
        enable_thinking=False,
        tokenize=False,
        image_prompts=["untrusted"],
        max_length=1,
        padding=True,
        reasoning_effort="none",
        response_format={"type": "json_object"},
        response_schema={"type": "object"},
        return_dict=True,
        return_tensors="pt",
        thinking=False,
        tool_choice="none",
        truncation=True,
    )

    assert prompt == [11, 12]
    assert calls == [
        {
            "conversation": conversation,
            "tokenize": True,
            "add_generation_prompt": True,
            "padding": False,
            "truncation": False,
            "return_tensors": None,
            "return_dict": False,
            "thinking": False,
            "tool_choice": "none",
            "response_format": {"type": "json_object"},
            "response_schema": {"type": "object"},
        }
    ]


def test_renderer_converts_developer_role_without_reordering_or_flattening():
    conversation = [
        {"role": "user", "content": "question"},
        {
            "role": "developer",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "developer policy"},
            ],
            "tools": [],
        },
        {"role": "system", "content": "system policy"},
    ]

    normalized = renderer_patch._normalize_developer_messages(conversation)

    assert normalized == [
        {"role": "user", "content": "question"},
        {
            "role": "system",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "developer policy"},
            ],
        },
        {"role": "system", "content": "system policy"},
    ]
    assert conversation == [
        {"role": "user", "content": "question"},
        {
            "role": "developer",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "developer policy"},
            ],
            "tools": [],
        },
        {"role": "system", "content": "system policy"},
    ]


def test_renderer_reorders_tool_results_and_adds_k3_metadata():
    conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-weather",
                    "function": {"name": "get_weather", "arguments": "{}"},
                },
                {
                    "id": "call-time",
                    "function": {"name": "get_time", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-time",
            "content": "10:30",
        },
        {
            "role": "tool",
            "tool_call_id": "call-weather",
            "content": "sunny",
        },
        {"role": "user", "content": "thanks"},
    ]

    normalized = renderer_patch._normalize_k3_tool_messages(conversation)

    assert normalized == [
        conversation[0],
        {
            "role": "tool",
            "tool_call_id": "call-weather",
            "content": "sunny",
            "tool": "get_weather",
            "index": 1,
        },
        {
            "role": "tool",
            "tool_call_id": "call-time",
            "content": "10:30",
            "tool": "get_time",
            "index": 2,
        },
        conversation[3],
    ]
    weather_result = conversation[1]
    time_result = conversation[2]
    assert isinstance(weather_result, dict)
    assert isinstance(time_result, dict)
    assert "tool" not in weather_result
    assert "index" not in weather_result
    assert "tool" not in time_result
    assert "index" not in time_result


def test_renderer_leaves_entire_tool_result_block_unchanged_on_fallback():
    conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-weather",
                    "function": {"name": "get_weather", "arguments": "{}"},
                },
                {
                    "id": "call-time",
                    "function": {"name": "get_time", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-time",
            "content": "10:30",
        },
        {
            "role": "tool",
            "tool_call_id": "unknown-call",
            "content": "unknown",
        },
        {
            "role": "tool",
            "tool_call_id": "call-weather",
            "content": "sunny",
        },
    ]

    normalized = renderer_patch._normalize_k3_tool_messages(conversation)

    assert normalized == conversation
    assert all("tool" not in message for message in normalized[1:])
    assert all("index" not in message for message in normalized[1:])


def test_renderer_preserves_multimodal_data(monkeypatch):
    conversation = [
        {
            "role": "user",
            "content": "<|kimi_image_placeholder|>\ndescribe",
        }
    ]
    mm_data = {"image": [object()]}
    mm_uuids = {"image": ["image-uuid"]}
    calls = []

    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            calls.append(kwargs)
            return [21, 22]

    def parse_messages(*args, **kwargs):
        assert kwargs["content_format"] == "string"
        assert kwargs["media_io_kwargs"] == {
            "image": {"image_mode": None},
        }
        return conversation, mm_data, mm_uuids

    monkeypatch.setattr(renderer_patch, "parse_chat_messages", parse_messages)
    monkeypatch.setattr(
        renderer_patch,
        "parse_dec_only_prompt",
        lambda prompt: {"prompt_token_ids": prompt},
    )

    renderer = object.__new__(KimiK3Renderer)
    renderer.model_config = _model_config("kimi_k3")
    renderer.tokenizer = RecordingTokenizer()
    params = ChatParams(
        chat_template=None,
        chat_template_kwargs={"thinking": True, "thinking_effort": "max"},
    )

    rendered_conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "describe"}],
        params,
    )

    assert rendered_conversation == conversation
    assert prompt == {
        "prompt_token_ids": [21, 22],
        "multi_modal_data": mm_data,
        "multi_modal_uuids": mm_uuids,
    }
    assert calls[0]["conversation"] == conversation
    assert calls[0]["thinking_effort"] == "max"
    assert calls[0]["tokenize"] is True
    assert calls[0]["padding"] is False
    assert calls[0]["truncation"] is False
    assert calls[0]["return_tensors"] is None
    assert calls[0]["return_dict"] is False
    assert "image_prompts" not in calls[0]
    assert "chat_template" not in calls[0]


def test_async_renderer_preserves_multimodal_data(monkeypatch):
    conversation = [
        {
            "role": "user",
            "content": "<|kimi_image_placeholder|>\ndescribe",
        }
    ]
    mm_data = {"image": [object()]}
    mm_uuids = {"image": ["image-uuid"]}
    parse_messages = AsyncMock(return_value=(conversation, mm_data, mm_uuids))
    apply_template = AsyncMock(return_value=[31, 32])
    monkeypatch.setattr(renderer_patch, "parse_chat_messages_async", parse_messages)
    monkeypatch.setattr(
        renderer_patch,
        "parse_dec_only_prompt",
        lambda prompt: {"prompt_token_ids": prompt},
    )

    renderer = object.__new__(KimiK3Renderer)
    renderer.model_config = _model_config("kimi_k3")
    renderer._apply_chat_template_async = apply_template

    rendered_conversation, prompt = asyncio.run(
        renderer.render_messages_async(
            [{"role": "user", "content": "describe"}],
            ChatParams(),
        )
    )

    assert rendered_conversation == conversation
    assert prompt == {
        "prompt_token_ids": [31, 32],
        "multi_modal_data": mm_data,
        "multi_modal_uuids": mm_uuids,
    }
    apply_template.assert_awaited_once_with(conversation, return_dict=False)
    assert parse_messages.await_args is not None
    assert parse_messages.await_args.kwargs["content_format"] == "string"
    assert parse_messages.await_args.kwargs["media_io_kwargs"] == {
        "image": {"image_mode": None},
    }


def test_server_defaults_cannot_override_typed_kimi_k3_tool_controls():
    request = _request(
        reasoning_effort="none",
        tools=None,
        tool_choice="none",
    )
    renderer_patch.prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto").with_defaults(
        {
            "thinking": True,
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "injected",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )

    assert params.chat_template_kwargs["thinking"] is False
    assert params.chat_template_kwargs["tool_choice"] == "none"
    assert params.chat_template_kwargs["tools"] == []


@pytest.mark.parametrize(
    ("server_thinking", "reasoning_effort", "expected_thinking"),
    [
        (False, "high", True),
        (True, "none", False),
    ],
)
def test_request_reasoning_effort_overrides_server_thinking_default(
    server_thinking,
    reasoning_effort,
    expected_thinking,
):
    serving = object.__new__(OpenAIServingChat)
    serving.model_config = _model_config("kimi_k3")
    serving.renderer = object.__new__(KimiK3Renderer)
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.default_chat_template_kwargs = {"thinking": server_thinking}

    kwargs = serving._effective_chat_template_kwargs(
        _request(reasoning_effort=reasoning_effort),
    )

    assert kwargs["thinking"] is expected_thinking


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("tools", []),
        ("tool_choice", "none"),
        ("response_format", {"type": "text"}),
        ("reasoning_effort", "none"),
    ],
)
def test_duplicate_openai_controls_in_chat_template_kwargs_are_rejected(parameter, value):
    request = _request(chat_template_kwargs={parameter: value})

    with pytest.raises(VLLMValidationError) as exc_info:
        renderer_patch.prepare_kimi_k3_chat_template_kwargs(request)

    assert exc_info.value.parameter == parameter


@pytest.mark.parametrize(
    ("reasoning_effort", "native_kwargs"),
    [
        ("high", {"thinking": False}),
        ("none", {"thinking": True}),
        ("high", {"thinking_effort": "max"}),
        ("none", {"thinking_effort": "high"}),
    ],
)
def test_conflicting_typed_and_native_reasoning_controls_are_rejected(
    reasoning_effort,
    native_kwargs,
):
    request = _request(
        reasoning_effort=reasoning_effort,
        chat_template_kwargs=native_kwargs,
    )

    with pytest.raises(VLLMValidationError) as exc_info:
        renderer_patch.prepare_kimi_k3_chat_template_kwargs(request)

    assert exc_info.value.parameter == "reasoning_effort"


def test_native_reasoning_controls_are_promoted_to_typed_request():
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "help"}],
        chat_template_kwargs={
            "thinking": True,
            "thinking_effort": "max",
        },
    )

    renderer_patch.prepare_kimi_k3_chat_template_kwargs(request)

    assert request.reasoning_effort == "max"


def test_auto_tool_choice_survives_vllm_default_merging():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {"type": "object"},
            },
        }
    ]
    request = _request(
        tools=tools,
        tool_choice="auto",
    )
    renderer_patch.prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto").with_defaults(
        {
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "injected",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    assert params.chat_template_kwargs["tool_choice"] == "required"
    assert decode_kimi_k3_prompt_tool_choice(params.chat_template_kwargs[KIMI_K3_PROMPT_TOOL_CHOICE_KEY]) == "auto"

    calls = []

    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            calls.append(kwargs)
            return [41, 42]

    renderer = object.__new__(KimiK3Renderer)
    renderer.tokenizer = RecordingTokenizer()
    conversation = [{"role": "user", "content": "what time is it?"}]
    prompt = renderer._apply_chat_template(
        conversation,
        **params.get_apply_chat_template_kwargs(),
    )

    assert prompt == [41, 42]
    assert calls[0]["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in calls[0]["tools"]] == ["get_time"]
    assert KIMI_K3_PROMPT_TOOL_CHOICE_KEY not in calls[0]


@pytest.mark.parametrize(("renderer_cls", "expected_thinking"), [(KimiK3Renderer, True), (object, None)])
def test_render_server_prepares_only_selected_renderer_requests(
    monkeypatch,
    renderer_cls: type,
    expected_thinking: bool | None,
):
    async def original_render_chat(self, request, *, skip_mm_cache=False):
        del self, skip_mm_cache
        return dict(request.chat_template_kwargs or {})

    monkeypatch.setattr(
        OnlineRenderer,
        renderer_patch._ORIGINAL_RENDER_CHAT_ATTR,
        original_render_chat,
    )
    serving = object.__new__(OnlineRenderer)
    serving.renderer = object.__new__(renderer_cls)

    kwargs = asyncio.run(serving.render_chat(_request()))

    assert kwargs.get("thinking") is expected_thinking
    if renderer_cls is KimiK3Renderer:
        request = _request()
        asyncio.run(serving.render_chat(request))
        assert request.skip_special_tokens is False
        assert request.spaces_between_special_tokens is False


def test_kimi_k3_render_server_delegates_non_chat_requests(monkeypatch):
    async def original_render_chat(self, request, *, skip_mm_cache=False):
        del self, skip_mm_cache
        return request

    monkeypatch.setattr(
        OnlineRenderer,
        renderer_patch._ORIGINAL_RENDER_CHAT_ATTR,
        original_render_chat,
    )
    serving = object.__new__(OnlineRenderer)
    serving.renderer = object.__new__(KimiK3Renderer)

    request = SimpleNamespace(chat_template_kwargs=None)
    assert asyncio.run(serving.render_chat(request)) is request
