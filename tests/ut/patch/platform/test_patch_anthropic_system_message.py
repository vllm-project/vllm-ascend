# SPDX-License-Identifier: Apache-2.0

from vllm.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

from vllm_ascend.patch.platform import patch_anthropic_system_message  # noqa: F401


def _make_request(
    messages: list[dict],
    **kwargs,
) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=128,
        messages=messages,
        **kwargs,
    )


def test_inline_system_role_is_accepted_by_anthropic_requests():
    request = _make_request([{"role": "system", "content": "Be concise."}])
    count_request = AnthropicCountTokensRequest(
        model="test-model",
        messages=[{"role": "system", "content": "Be concise."}],
    )

    assert request.messages[0].role == "system"
    assert count_request.messages[0].role == "system"


def test_inline_system_string_is_merged_and_not_kept_as_chat_message():
    request = _make_request(
        [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Be concise."},
        ],
        system="Top-level prompt.",
    )

    result = AnthropicServingMessages._convert_anthropic_to_openai_request(request)

    assert result.messages == [
        {"role": "system", "content": "Top-level prompt.Be concise."},
        {"role": "user", "content": "Hello"},
    ]


def test_inline_system_list_content_is_merged_with_billing_header_stripped():
    request = _make_request(
        [
            {"role": "user", "content": "help?"},
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "x-anthropic-billing-header: cc_version=2.1.160;",
                    },
                    {"type": "text", "text": "Use short answers. "},
                    {"type": "text", "text": "Prefer examples."},
                ],
            },
        ],
        system=[
            {"type": "text", "text": "Existing system. "},
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cch=d1d48;",
            },
        ],
    )

    result = AnthropicServingMessages._convert_anthropic_to_openai_request(request)

    assert result.messages[0] == {
        "role": "system",
        "content": "Existing system. Use short answers. Prefer examples.",
    }
    assert result.messages[1] == {"role": "user", "content": "help?"}


def test_multiple_inline_system_messages_are_all_merged():
    request = _make_request(
        [
            {"role": "system", "content": "First."},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Second."},
        ]
    )

    result = AnthropicServingMessages._convert_anthropic_to_openai_request(request)

    assert result.messages == [
        {"role": "system", "content": "First.Second."},
        {"role": "user", "content": "Hello"},
    ]
