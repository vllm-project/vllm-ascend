# SPDX-License-Identifier: MIT
# Copyright (c) 2023 DeepSeek
# Reference tests; see vllm_ascend/patch/platform/patch_deepseek_v41_frontend/LICENSE.
"""
Tests for encoding.py (DeepSeek-V4.1 encoding).

Adapted from dsv41-master/deepseek_harmony/tests/test_deepseek_v41.py for the
self-contained dict-based API in this repo.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from vllm_ascend.patch.platform.patch_deepseek_v41_frontend import encoding as enc
from vllm_ascend.patch.platform.patch_deepseek_v41_frontend.encoding import (
    IMAGE_PLACEHOLDER,
    SYSTEM_SP_TOKEN,
    encode_messages,
    merge_tool_messages,
    parse_message_from_completion_text,
    render_message,
)

REASONING_EFFORT_TEMPLATE = (
    SYSTEM_SP_TOKEN + "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)

V41_TOOL_CALL_OUTPUT = (
    "  reason  </think>summary\n\n"
    "<｜DSML｜ calls>\n"
    '<｜DSML｜ invoke name="lookup">\n'
    '<｜DSML｜ parameter name="query" string="true">value'
    "</｜DSML｜ parameter>\n"
    '<｜DSML｜ parameter name="limit" string="false">2'
    "</｜DSML｜ parameter>\n"
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls><｜end▁of▁sentence｜>"
)


def make_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }


def make_tool_call_messages() -> list:
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "reasoning_content": "  reason  ",
            "content": "summary",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"value","limit":2}',
                    },
                }
            ],
        },
    ]


# ============================================================
# Vision
# ============================================================


def test_v41_renders_images() -> None:
    prompt, media = encode_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image_url", "image_url": {"url": "/unused/image.png"}},
                ],
            }
        ],
        thinking_mode="chat",
        return_multi_modal_data=True,
    )

    assert prompt == (f"<｜begin▁of▁sentence｜><｜User｜>inspect\n\n{IMAGE_PLACEHOLDER}<｜Assistant｜></think>")
    assert media == {"images": [{"type": "image", "url": "/unused/image.png"}]}


def test_v41_rejects_image_placeholder_in_text() -> None:
    with pytest.raises(ValueError):
        encode_messages(
            [{"role": "user", "content": f"hi {IMAGE_PLACEHOLDER}"}],
            thinking_mode="chat",
        )


# ============================================================
# Reasoning Effort
# ============================================================


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 50),
        ("low", 25),
        ("high", 50),
        ("xhigh", 75),
        ("max", 100),
        (1, 1),
        (42, 42),
        (100, 100),
    ],
)
def test_v41_maps_reasoning_effort_to_1_100_budget(
    effort: Any,
    budget: int,
) -> None:
    prompt = encode_messages(
        [{"role": "user", "content": "question"}],
        thinking_mode="thinking",
        reasoning_effort=effort,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜>"
        f"{REASONING_EFFORT_TEMPLATE.format(budget=budget)}"
        "<｜User｜>question<｜Assistant｜><think>"
    )


def test_v41_only_adds_reasoning_effort_to_first_thinking_message() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]

    later_message = render_message(1, messages, thinking_mode="thinking", reasoning_effort=100)
    chat_message = render_message(0, messages, thinking_mode="chat", reasoning_effort=100)

    assert "Reasoning Effort:" not in later_message
    assert "Reasoning Effort:" not in chat_message


def test_v41_chat_mode_has_no_reasoning_effort_or_system_token() -> None:
    prompt = encode_messages(
        [{"role": "user", "content": "hello"}],
        thinking_mode="chat",
        reasoning_effort="max",
    )
    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"


@pytest.mark.parametrize("effort", [-1, 0, 101, "medium"])
def test_v41_rejects_out_of_range_or_unknown_reasoning_effort(
    effort: Any,
) -> None:
    with pytest.raises(AssertionError, match=r"int within \[1,100\]"):
        encode_messages(
            [{"role": "user", "content": "question"}],
            thinking_mode="thinking",
            reasoning_effort=effort,
        )


@pytest.mark.parametrize("effort", [True, False, 1.5])
def test_v41_rejects_non_string_non_integer_effort_types(effort: Any) -> None:
    # bool is not `type(...) is int`; float is invalid too
    with pytest.raises(AssertionError):
        encode_messages(
            [{"role": "user", "content": "question"}],
            thinking_mode="thinking",
            reasoning_effort=effort,
        )


# ============================================================
# System token
# ============================================================


def test_v41_leading_system_message_uses_system_token() -> None:
    prompt = encode_messages(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ],
        thinking_mode="chat",
    )
    assert prompt == (
        "<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>"
    )


def test_v41_mid_conversation_system_message() -> None:
    prompt = encode_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "system", "content": "mid sys"},
        ],
        thinking_mode="thinking",
        reasoning_effort=88,
    )
    # Mid-conversation system gets its own <｜System｜> token and triggers
    # the assistant generation header afterwards.
    assert prompt == (
        "<｜begin▁of▁sentence｜>"
        f"{REASONING_EFFORT_TEMPLATE.format(budget=88)}"
        "sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"
        "<｜System｜>mid sys<｜Assistant｜><think>"
    )


# ============================================================
# DSML tool tags
# ============================================================


def test_v41_tool_instructions_use_spaced_dsml_tags_in_chat_mode() -> None:
    prompt = encode_messages(
        [
            {"role": "system", "content": "system", "tools": [make_tool()]},
            {"role": "user", "content": "question"},
        ],
        thinking_mode="chat",
    )

    assert (
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="$TOOL_NAME">\n'
        '<｜DSML｜ parameter name="$PARAMETER_NAME" '
        'string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>\n'
        "...\n"
        "</｜DSML｜ invoke>"
    ) in prompt
    assert "<｜DSML｜tool_calls>" not in prompt
    assert "<｜DSML｜invoke" not in prompt
    assert "<｜DSML｜parameter" not in prompt


def test_v41_renders_spaced_dsml_with_v4_assistant_semantics() -> None:
    messages = make_tool_call_messages()

    prompt = render_message(1, messages, thinking_mode="thinking")

    assert prompt == V41_TOOL_CALL_OUTPUT


def test_v41_parses_spaced_dsml_roundtrip() -> None:
    messages = make_tool_call_messages()

    parsed = parse_message_from_completion_text(V41_TOOL_CALL_OUTPUT, thinking_mode="thinking")

    assert parsed["role"] == "assistant"
    assert parsed["reasoning_content"] == "  reason  "
    assert parsed["content"] == "summary"
    assert parsed["tool_calls"]
    assert parsed["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "query": "value",
        "limit": 2,
    }

    # Re-encoding the parsed message reproduces the original completion text
    assert (
        encode_messages(
            [parsed],
            thinking_mode="thinking",
            context=messages[:1],
        )
        == V41_TOOL_CALL_OUTPUT
    )


def test_v41_parse_rejects_unspaced_v4_dsml() -> None:
    v4_output = (
        V41_TOOL_CALL_OUTPUT.replace("｜DSML｜ calls", "｜DSML｜tool_calls")
        .replace("｜DSML｜ invoke", "｜DSML｜invoke")
        .replace("｜DSML｜ parameter", "｜DSML｜parameter")
    )
    with pytest.raises(AssertionError):
        parse_message_from_completion_text(v4_output, thinking_mode="thinking")


# ============================================================
# Multi-turn flow
# ============================================================


def test_v41_drop_thinking_without_tools() -> None:
    prompt = encode_messages(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "q2"},
        ],
        thinking_mode="thinking",
        drop_thinking=True,
    )
    # Earlier turn reasoning dropped, </think> form; new turn opens <think>
    assert "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt
    assert "r1" not in prompt
    assert prompt.endswith("<｜User｜>q2<｜Assistant｜><think>")


# ============================================================
# Preprocessing
# ============================================================


def test_merge_tool_messages_creates_tool_result_blocks() -> None:
    merged = merge_tool_messages(
        [
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "a", "content": "r1"},
            {"role": "tool", "tool_call_id": "b", "content": "r2"},
        ]
    )
    assert len(merged) == 2
    assert merged[1]["role"] == "user"
    assert [b["type"] for b in merged[1]["content_blocks"]] == ["tool_result", "tool_result"]


def test_v41_task_sp_token() -> None:
    prompt = encode_messages(
        [{"role": "user", "content": "classify me", "task": "query"}],
        thinking_mode="chat",
    )
    assert prompt.endswith("classify me<｜query｜>")
    assert "<｜Assistant｜>" not in prompt


# ============================================================
# Golden fixtures from encoding/tests
# ============================================================

ENCODING_DIR = Path(__file__).resolve().parent
ENCODING_FIXTURES_DIR = ENCODING_DIR / "fixtures"
INFERENCE_EXAMPLES_DIR = ENCODING_FIXTURES_DIR / "examples"

FIXTURE_CASE_IDS = sorted(int(p.stem.split("_")[-1]) for p in ENCODING_FIXTURES_DIR.glob("test_input_*.json"))


@pytest.mark.parametrize("case_id", FIXTURE_CASE_IDS)
def test_examples_encoding_golden_outputs(case_id: int) -> None:
    """Each tests/encoding input must encode to its checked-in golden output."""
    input_file = ENCODING_FIXTURES_DIR / f"test_input_{case_id}.json"
    output_file = ENCODING_FIXTURES_DIR / f"test_output_{case_id}.txt"
    assert output_file.exists(), f"missing golden output: {output_file.name} (run tests/encoding/regen_outputs.py)"

    case = enc.load_cases(str(input_file))[0]
    prompt, _ = enc.encode_case(case, thinking_mode="chat")

    assert prompt == output_file.read_text(), (
        f"{output_file.name} is stale; regenerate with tests/encoding/regen_outputs.py"
    )


def test_examples_v41_output_uses_v41_format_markers() -> None:
    """Sanity-check the V4.1 goldens actually exercise V4.1-specific format."""
    # case 1: tool calls with spaced DSML tags
    out1 = (ENCODING_FIXTURES_DIR / "test_output_1.txt").read_text()
    assert "<｜DSML｜ calls>" in out1 and '<｜DSML｜ invoke name="get_weather">' in out1
    assert "<｜DSML｜tool_calls>" not in out1

    # case 5: numeric reasoning effort behind the system token
    out5 = (ENCODING_FIXTURES_DIR / "test_output_5.txt").read_text()
    assert out5.startswith("<｜begin▁of▁sentence｜>" + REASONING_EFFORT_TEMPLATE.format(budget=100))
    assert out5.count(IMAGE_PLACEHOLDER) == 2


def test_examples_vl_txt_and_json_encode_identically() -> None:
    """The TXT (last block of example.txt) and JSON vision examples must encode identically."""
    txt = (INFERENCE_EXAMPLES_DIR / "example.txt").read_text().rstrip("\n").split("\n\n")[-1]
    messages = [{"role": "user", "content": enc.parse_tagged_text(txt)}]
    p1, m1 = encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)

    case = enc.load_cases(str(INFERENCE_EXAMPLES_DIR / "example_harmony.json"))[0]
    p2, m2 = enc.encode_case(case, thinking_mode="chat")

    assert p1 == p2
    assert m1["images"] == m2
    assert len(m2) == 2


def test_examples_harmony_cases_encode() -> None:
    """All example_harmony.json cases encode without error."""
    cases = enc.load_cases(str(INFERENCE_EXAMPLES_DIR / "example_harmony.json"))
    assert len(cases) == 4

    # case 1 (vision) is covered by test_examples_vl_txt_and_json_encode_identically

    # cases are pure OpenAI format: mode/effort are passed at call time
    prompt = encode_messages(cases[1]["messages"], thinking_mode="thinking", reasoning_effort=75)
    assert REASONING_EFFORT_TEMPLATE.format(budget=75) in prompt

    # case 3: tools with spaced DSML tags
    prompt, _ = enc.encode_case(cases[2], thinking_mode="chat")
    assert "<｜DSML｜ calls>" in prompt

    # case 4: mid-conversation system message triggers assistant header
    prompt, _ = enc.encode_case(cases[3], thinking_mode="chat")
    assert "<｜System｜>Mid-conversation instruction update" in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
