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

import hashlib
import json
import os
from pathlib import Path

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally

from vllm_ascend.patch.platform.patch_kimi_k3_parsers import (
    ARGUMENT_END,
    CALL_END,
    MESSAGE_END,
    RESPONSE_END,
    RESPONSE_START,
    THINK_END,
    THINK_START,
    TOOLS_END,
    TOOLS_START,
    KimiK3Parser,
    KimiK3ReasoningParser,
    KimiK3ToolParser,
)
from vllm_ascend.patch.platform.patch_kimi_k3_renderer import (
    KIMI_K3_IMAGE_PROMPT,
    KimiK3Renderer,
    prepare_kimi_k3_chat_template_kwargs,
)

_TOKENIZER_PATH_ENV = "KIMI_K3_TOKENIZER_PATH"
_KIMI_K3_TOKENIZER_FILE_SHA256 = {
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    "tokenization_kimi.py": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    "encoding_k3.py": "c3869cdb7c5a81b1ee621e55ba589d8f3ffae83063c1085571ee96e2feb826a8",
    "tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
}


class KimiK3DelegatingParser(KimiK3Parser):
    reasoning_parser_cls = KimiK3ReasoningParser
    tool_parser_cls = KimiK3ToolParser


@pytest.fixture(scope="module")
def real_kimi_k3_tokenizer():
    tokenizer_path = os.getenv(_TOKENIZER_PATH_ENV)
    if not tokenizer_path:
        pytest.skip(f"{_TOKENIZER_PATH_ENV} is not set")
    assert tokenizer_path is not None
    path = Path(tokenizer_path)
    for filename, expected_digest in _KIMI_K3_TOKENIZER_FILE_SHA256.items():
        tokenizer_file = path / filename
        assert tokenizer_file.is_file(), f"{tokenizer_file} is unavailable"
        assert hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() == (expected_digest)
    return get_tokenizer(
        path,
        tokenizer_mode="kimi_k3",
        trust_remote_code=True,
        use_fast=False,
    )


def _incremental_deltas(tokenizer, token_ids, *, spaces_between_special_tokens):
    all_ids: list[int] = []
    previous_tokens: list[str] = []
    prefix_offset = 0
    read_offset = 0
    parts: list[str] = []
    for token_id in token_ids:
        all_ids.append(token_id)
        (
            new_tokens,
            delta,
            prefix_offset,
            read_offset,
        ) = detokenize_incrementally(
            tokenizer,
            all_ids,
            previous_tokens,
            prefix_offset,
            read_offset,
            skip_special_tokens=False,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        previous_tokens.extend(new_tokens)
        parts.append(delta)
    return parts


def _incremental_decode(tokenizer, token_ids, *, spaces_between_special_tokens):
    return "".join(
        _incremental_deltas(
            tokenizer,
            token_ids,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    )


def _decode(tokenizer, token_ids):
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )


def _tool(name, properties=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties or {},
            },
        },
    }


def _request(*, tools=None, reasoning_effort="none", messages=None):
    return ChatCompletionRequest(
        model="kimi-k3",
        messages=messages or [{"role": "user", "content": "test"}],
        tools=tools,
        tool_choice="auto" if tools else None,
        reasoning_effort=reasoning_effort,
    )


def test_real_incremental_detokenizer_preserves_adjacent_xtml_markers(
    real_kimi_k3_tokenizer,
):
    expected = THINK_END + RESPONSE_START
    token_ids = real_kimi_k3_tokenizer.encode(
        expected,
        add_special_tokens=False,
    )

    assert (
        _incremental_decode(
            real_kimi_k3_tokenizer,
            token_ids,
            spaces_between_special_tokens=False,
        )
        == expected
    )
    assert (
        _incremental_decode(
            real_kimi_k3_tokenizer,
            token_ids,
            spaces_between_special_tokens=True,
        )
        != expected
    )


@pytest.mark.parametrize("reasoning_text", ["private reasoning", ""])
def test_real_incremental_detokenizer_reconstructs_thinking_chat_response(
    real_kimi_k3_tokenizer,
    reasoning_text,
):
    generated = reasoning_text + THINK_END + RESPONSE_START + "public answer" + RESPONSE_END + MESSAGE_END
    token_ids = real_kimi_k3_tokenizer.encode(
        generated,
        add_special_tokens=False,
    )
    deltas = _incremental_deltas(
        real_kimi_k3_tokenizer,
        token_ids,
        spaces_between_special_tokens=False,
    )
    request = _request(reasoning_effort="max")
    parser = KimiK3DelegatingParser(
        real_kimi_k3_tokenizer,
        chat_template_kwargs={"thinking": True},
    )
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    for index, (token_id, delta_text) in enumerate(zip(token_ids, deltas, strict=True)):
        delta = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[token_id],
            request=request,
            finished=index == len(token_ids) - 1,
        )
        if delta is not None:
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)

    assert "".join(deltas) == generated
    assert "".join(reasoning_parts) == reasoning_text
    assert "".join(content_parts) == "public answer"


def test_real_incremental_detokenizer_extracts_bfcl_native_tool_call(
    real_kimi_k3_tokenizer,
):
    generated = (
        RESPONSE_END
        + TOOLS_START
        + '<|open|>call tool="solve_quadratic_equation" index="1"<|sep|>'
        + '<|open|>argument key="a" type="number"<|sep|>2'
        + ARGUMENT_END
        + '<|open|>argument key="b" type="number"<|sep|>6'
        + ARGUMENT_END
        + '<|open|>argument key="c" type="number"<|sep|>5'
        + ARGUMENT_END
        + CALL_END
        + TOOLS_END
        + MESSAGE_END
    )
    token_ids = real_kimi_k3_tokenizer.encode(
        generated,
        add_special_tokens=False,
    )
    deltas = _incremental_deltas(
        real_kimi_k3_tokenizer,
        token_ids,
        spaces_between_special_tokens=False,
    )
    tools = [
        _tool(
            "solve_quadratic_equation",
            {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "c": {"type": "number"},
            },
        )
    ]
    request = _request(tools=tools)
    parser = KimiK3DelegatingParser(
        real_kimi_k3_tokenizer,
        tools,
        chat_template_kwargs={"thinking": False},
    )
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}

    for index, (token_id, delta_text) in enumerate(zip(token_ids, deltas, strict=True)):
        delta = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[token_id],
            request=request,
            finished=index == len(token_ids) - 1,
        )
        if delta is not None:
            for tool_call in delta.tool_calls:
                assert tool_call.function is not None
                if tool_call.function.name:
                    names[tool_call.index] = tool_call.function.name
                if tool_call.function.arguments is not None:
                    arguments[tool_call.index] = arguments.get(tool_call.index, "") + tool_call.function.arguments

    assert "".join(deltas) == generated
    assert names == {0: "solve_quadratic_equation"}
    assert set(arguments) == {0}
    assert json.loads(arguments[0]) == {"a": 2, "b": 6, "c": 5}


def test_real_renderer_defaults_to_thinking_generation(
    real_kimi_k3_tokenizer,
):
    conversation = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "continue"},
    ]
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=conversation,
    )
    prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto")
    renderer = object.__new__(KimiK3Renderer)
    renderer.tokenizer = real_kimi_k3_tokenizer

    prompt_ids = renderer._apply_chat_template(
        conversation,
        **params.get_apply_chat_template_kwargs(),
    )
    prompt = _decode(real_kimi_k3_tokenizer, prompt_ids)

    assert RESPONSE_START + "answer" in prompt
    assert 'type="thinking-effort"' in prompt
    assert prompt.endswith(THINK_START)


@pytest.mark.parametrize(
    ("reasoning_effort", "generation_marker", "has_effort_instruction"),
    [
        ("high", THINK_START, True),
        ("none", RESPONSE_START, False),
    ],
)
def test_real_renderer_preserves_multiturn_system_and_reasoning_controls(
    real_kimi_k3_tokenizer,
    reasoning_effort,
    generation_marker,
    has_effort_instruction,
):
    tools = [_tool("get_weather")]
    conversation = [
        {"role": "system", "content": "initial policy"},
        {"role": "user", "content": "weather"},
        {
            "role": "assistant",
            "reasoning_content": "need tool",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call-weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-weather",
            "content": "sunny",
        },
        {"role": "system", "content": "answer briefly"},
        {"role": "user", "content": "continue"},
    ]
    request = _request(
        tools=tools,
        reasoning_effort=reasoning_effort,
        messages=conversation,
    )
    prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto")
    renderer = object.__new__(KimiK3Renderer)
    renderer.tokenizer = real_kimi_k3_tokenizer

    prompt_ids = renderer._apply_chat_template(
        conversation,
        **params.get_apply_chat_template_kwargs(),
    )
    prompt = real_kimi_k3_tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )

    initial_pos = prompt.index("initial policy")
    tool_result_pos = prompt.index("sunny")
    mid_system_pos = prompt.index("answer briefly")
    final_user_pos = prompt.index("continue")
    assert initial_pos < tool_result_pos < mid_system_pos < final_user_pos
    assert 'message role="tool" tool="get_weather" index="1"' in prompt
    assert prompt.endswith(generation_marker)
    assert ('type="thinking-effort"' in prompt) is has_effort_instruction
    assert "need tool" in prompt


def test_real_segmented_encoding_blocks_prompt_marker_injection(
    real_kimi_k3_tokenizer,
):
    user_text = f"literal user text: {TOOLS_START}"
    conversation = [{"role": "user", "content": user_text}]

    trusted_ids = real_kimi_k3_tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        thinking=False,
    )
    rendered_text = real_kimi_k3_tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        thinking=False,
    )
    unsafe_ids = real_kimi_k3_tokenizer.encode(
        rendered_text,
        add_special_tokens=False,
    )

    open_token_id = real_kimi_k3_tokenizer.convert_tokens_to_ids("<|open|>")
    assert unsafe_ids.count(open_token_id) > trusted_ids.count(open_token_id)
    assert trusted_ids != unsafe_ids


@pytest.mark.parametrize("image_count", [0, 1, 2])
def test_real_segmented_encoding_preserves_only_trusted_image_placeholders(
    real_kimi_k3_tokenizer,
    image_count,
):
    content = [{"type": "text", "text": "before"}]
    for index in range(image_count):
        content.extend(
            [
                {"type": "image"},
                {"type": "text", "text": f"after-{index}"},
            ]
        )
    conversation = [{"role": "user", "content": content}]

    prompt_ids = real_kimi_k3_tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        thinking=False,
        image_prompts=[KIMI_K3_IMAGE_PROMPT] * image_count,
    )

    prompt = _decode(real_kimi_k3_tokenizer, prompt_ids)
    assert prompt.count(KIMI_K3_IMAGE_PROMPT) == image_count
