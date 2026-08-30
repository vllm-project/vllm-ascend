# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import asyncio
import copy
import json
from typing import Any

import pytest

from vllm_ascend.patch.recompute_proxy import (
    RECOMPUTE_TOKEN_IDS_KEY,
    RecomputeContext,
    iter_sse_events,
    replace_rendered_chat_inputs,
)

pytestmark = pytest.mark.cpu_test


async def _chunks(parts: list[bytes]):
    for part in parts:
        yield part


def test_iter_sse_events_handles_split_and_coalesced_chunks() -> None:
    first = 'data: {"choices":[{"delta":{"reasoning":"think 中"}}]}\n\n'
    second = 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
    encoded = (first + second).encode()
    han_offset = encoded.index("中".encode())
    chunks = [
        encoded[: han_offset + 1],
        encoded[han_offset + 1 : len(first.encode()) + 8],
        encoded[len(first.encode()) + 8 :],
    ]

    async def collect_events() -> list[bytes]:
        return [event async for event in iter_sse_events(_chunks(chunks))]

    events = asyncio.run(collect_events())

    assert events == [first.encode(), second.encode()]


def test_iter_sse_events_handles_split_crlf() -> None:
    event = b'data: {"choices": []}\r\n\r\n'

    async def collect_events() -> list[bytes]:
        return [event async for event in iter_sse_events(_chunks([event[:-3], event[-3:]]))]

    assert asyncio.run(collect_events()) == [b'data: {"choices": []}\n\n']


def test_recompute_uses_exact_tokens_without_mutating_messages() -> None:
    request: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "Answer carefully."},
            {"role": "user", "content": "What was my first question?"},
            {"role": "assistant", "content": "You asked about arithmetic."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "stream": True,
        "max_completion_tokens": 8,
    }
    original_messages = copy.deepcopy(request["messages"])
    context = RecomputeContext.from_request(request)

    context.observe_response({"prompt_token_ids": [1, 2, 3], "choices": []})
    context.observe_response(
        {
            "choices": [
                {
                    "delta": {"reasoning": "2 + 2 is 4."},
                    "token_ids": [10, 11],
                    "finish_reason": None,
                }
            ]
        }
    )
    recomputed = context.observe_response(
        {
            "choices": [
                {
                    "delta": {},
                    "token_ids": [],
                    "finish_reason": "stop",
                    "stop_reason": "recomputed",
                }
            ]
        }
    )

    assert recomputed
    context.prepare_retry(request)
    assert request["messages"] == original_messages
    assert request["kv_transfer_params"][RECOMPUTE_TOKEN_IDS_KEY] == [1, 2, 3, 10, 11]
    assert request["max_completion_tokens"] == 6
    assert request["return_token_ids"] is True


def test_two_recomputes_extend_the_same_token_history() -> None:
    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Question"}],
        "stream": True,
        "max_tokens": 10,
    }
    context = RecomputeContext.from_request(request)
    context.observe_response({"prompt_token_ids": [1, 2], "choices": []})
    context.observe_response({"choices": [{"delta": {"reasoning": "R1"}, "token_ids": [3], "stop_reason": None}]})
    assert context.observe_response({"choices": [{"delta": {}, "token_ids": [], "stop_reason": "recomputed"}]})
    context.prepare_retry(request)
    assert request["kv_transfer_params"][RECOMPUTE_TOKEN_IDS_KEY] == [1, 2, 3]

    context.observe_response({"prompt_token_ids": [1, 2, 3], "choices": []})
    context.observe_response({"choices": [{"delta": {"content": "C1"}, "token_ids": [4, 5], "stop_reason": None}]})
    assert context.observe_response({"choices": [{"delta": {}, "token_ids": [], "stop_reason": "recomputed"}]})
    context.prepare_retry(request)

    assert request["kv_transfer_params"][RECOMPUTE_TOKEN_IDS_KEY] == [1, 2, 3, 4, 5]
    assert request["max_tokens"] == 7


def test_retry_rejects_changed_token_history() -> None:
    request = {
        "messages": [{"role": "user", "content": "Question"}],
        "stream": True,
        "max_completion_tokens": 4,
    }
    context = RecomputeContext.from_request(request)
    context.observe_response({"prompt_token_ids": [1, 2], "choices": []})
    context.observe_response({"choices": [{"delta": {"content": "C1"}, "token_ids": [3], "stop_reason": None}]})
    assert context.observe_response({"choices": [{"delta": {}, "token_ids": [], "stop_reason": "recomputed"}]})
    context.prepare_retry(request)

    assert request["max_completion_tokens"] == 3
    with pytest.raises(RuntimeError, match="exact token history"):
        context.observe_response({"prompt_token_ids": [1, 2, 99], "choices": []})


def test_nonstream_recompute_merges_one_logical_response() -> None:
    request = {
        "messages": [{"role": "user", "content": "Question"}],
        "stream": False,
        "max_tokens": 6,
    }
    context = RecomputeContext.from_request(request)
    first = {
        "id": "attempt-a",
        "model": "model",
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "reasoning": "R1", "content": "C1"},
                "token_ids": [3, 4],
                "finish_reason": "stop",
                "stop_reason": "recomputed",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }
    assert context.observe_response(first)
    context.prepare_retry(request)

    final = {
        "id": "attempt-b",
        "model": "model",
        "prompt_token_ids": [1, 2, 3, 4],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "reasoning": "R2", "content": "C2"},
                "token_ids": [5, 6],
                "finish_reason": "stop",
                "stop_reason": None,
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    assert not context.observe_response(final)
    client_payload = context.response_for_client(final)

    assert client_payload["choices"][0]["message"] == {
        "role": "assistant",
        "reasoning": "R1R2",
        "content": "C1C2",
    }
    assert "token_ids" not in client_payload["choices"][0]
    assert "prompt_token_ids" not in client_payload
    assert client_payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 4,
        "total_tokens": 6,
    }


def test_replace_rendered_chat_inputs_uses_internal_token_prefix() -> None:
    request = type(
        "Request",
        (),
        {
            "kv_transfer_params": {RECOMPUTE_TOKEN_IDS_KEY: [7, 8, 9]},
            "cache_salt": "salt",
        },
    )()
    conversation = [{"role": "user", "content": "unchanged"}]
    result = replace_rendered_chat_inputs(request, (conversation, [{"prompt_token_ids": [1]}]))

    assert result[0] is conversation
    assert result[1] == [
        {
            "type": "token",
            "prompt_token_ids": [7, 8, 9],
            "cache_salt": "salt",
        }
    ]
    assert RECOMPUTE_TOKEN_IDS_KEY not in request.kv_transfer_params


def test_stream_response_hides_internal_token_ids_and_rewrites_usage() -> None:
    request = {
        "messages": [{"role": "user", "content": "Question"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 6,
    }
    context = RecomputeContext.from_request(request)
    initial_role = {
        "id": "attempt-a",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "model",
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "delta": {"role": "assistant", "content": ""},
                "token_ids": None,
                "finish_reason": None,
            }
        ],
    }
    context.observe_response(initial_role)
    client_initial_role = context.response_for_client(initial_role)
    assert client_initial_role == {
        "id": "attempt-a",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "model",
        "choices": [
            {
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    context.observe_response({"choices": [{"delta": {"reasoning": "R1"}, "token_ids": [3, 4], "stop_reason": None}]})
    assert context.observe_response({"choices": [{"delta": {}, "token_ids": [], "stop_reason": "recomputed"}]})
    context.prepare_retry(request)

    retry_role = {
        "id": "attempt-b",
        "object": "chat.completion.chunk",
        "created": 2,
        "model": "model",
        "prompt_token_ids": [1, 2, 3, 4],
        "choices": [
            {
                "delta": {"role": "assistant", "content": ""},
                "token_ids": None,
                "finish_reason": None,
            }
        ],
    }
    context.observe_response(retry_role)
    assert context.response_for_client(retry_role) is None

    content = {
        "id": "attempt-b",
        "object": "chat.completion.chunk",
        "created": 2,
        "model": "model",
        "choices": [{"delta": {"content": "C2"}, "token_ids": [5], "finish_reason": None}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
    }
    context.observe_response(content)
    client_payload = context.response_for_client(content)
    assert client_payload["id"] == "attempt-a"
    assert client_payload["created"] == 1
    assert client_payload["choices"][0] == {"delta": {"content": "C2"}, "finish_reason": None}
    assert client_payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }

    # Ensure rewritten payloads remain serializable as SSE JSON.
    json.dumps(client_payload)

    terminal = {
        "id": "attempt-b",
        "object": "chat.completion.chunk",
        "created": 2,
        "model": "model",
        "choices": [
            {
                "delta": {},
                "finish_reason": "stop",
                "stop_reason": None,
                "token_ids": [],
            }
        ],
    }
    context.observe_response(terminal)
    client_terminal = context.response_for_client(terminal)
    assert client_terminal["id"] == "attempt-a"
    assert client_terminal["choices"] == [{"delta": {}, "finish_reason": "stop", "stop_reason": None}]
