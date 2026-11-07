# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Regression tests for the streaming ``kv_transfer_params`` patch.

Upstream vLLM only fills ``kv_transfer_params`` on the non-streaming
``ChatCompletionResponse``. Streaming clients (e.g. a router that prefetches
the KV blocks reported by the decode node) never see it, so the patch injects
it into the last SSE ``data:`` chunk, right before ``data: [DONE]``.

The wrapper is exercised with a fake upstream generator, so these tests do not
need a real engine or scheduler.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat  # noqa: E402

from vllm_ascend.patch.platform import patch_kv_transfer_params_streaming as patch_mod  # noqa: E402

_DONE = "data: [DONE]\n\n"


def _chunk(payload) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _fake_original(chunks: list[str], seen: dict | None = None):
    """Build a stand-in for the upstream stream generator.

    The wrapper passes the tracking result generator down, so the fake has to
    drain it to let the wrapper observe ``kv_transfer_params``.
    """

    async def _gen(self, request, result_generator, request_id, model_name, *args, **kwargs):
        if seen is not None:
            seen["args"] = args
            seen["kwargs"] = kwargs
        async for _ in result_generator:
            pass
        for chunk in chunks:
            yield chunk

    return _gen


def _collect(coro) -> list[str]:
    async def _run():
        return [chunk async for chunk in coro]

    return asyncio.run(_run())


def _stream(chunks: list[str], results: list, seen: dict | None = None, monkeypatch=None) -> list[str]:
    monkeypatch.setattr(patch_mod, "_original_stream_generator", _fake_original(chunks, seen))

    async def _results():
        for res in results:
            yield res

    coro = patch_mod._patched_chat_completion_stream_generator(
        SimpleNamespace(),
        SimpleNamespace(),
        _results(),
        "req-1",
        "model",
    )
    return _collect(coro)


def test_kv_transfer_params_injected_into_chunk_before_done(monkeypatch):
    handle = {"hash_keys": ["ab", "cd"]}
    chunks = [
        _chunk({"choices": [{"delta": {"role": "assistant"}}]}),
        _chunk({"choices": [{"delta": {"content": "hi"}}]}),
    ]
    results = [SimpleNamespace(kv_transfer_params=None), SimpleNamespace(kv_transfer_params=handle)]

    out = _stream(chunks, results, monkeypatch=monkeypatch)

    # Everything but the last chunk is untouched, and the stream still ends
    # with a single [DONE].
    assert out[:-2] == chunks[:-1]
    assert out[-1] == _DONE
    assert out.count(_DONE) == 1
    last = json.loads(out[-2][len("data: ") :])
    assert last["kv_transfer_params"] == handle
    assert last["choices"] == [{"delta": {"content": "hi"}}]


def test_stream_without_kv_transfer_params_is_unchanged(monkeypatch):
    chunks = [_chunk({"choices": [{"delta": {"content": "a"}}]}), _chunk({"choices": [{"delta": {"content": "b"}}]})]
    results = [SimpleNamespace(kv_transfer_params=None)]

    out = _stream(chunks, results, monkeypatch=monkeypatch)

    assert out == [*chunks, _DONE]


def test_non_data_chunk_is_not_rewritten(monkeypatch):
    chunks = [_chunk({"choices": [{"delta": {"content": "a"}}]}), ": keep-alive\n\n"]
    results = [SimpleNamespace(kv_transfer_params={"hash_keys": ["ab"]})]

    out = _stream(chunks, results, monkeypatch=monkeypatch)

    assert out == [*chunks, _DONE]


def test_empty_stream_still_terminates_with_done(monkeypatch):
    out = _stream([], [SimpleNamespace(kv_transfer_params={"hash_keys": ["ab"]})], monkeypatch=monkeypatch)

    assert out == [_DONE]


def test_done_chunk_with_crlf_is_recognized(monkeypatch):
    crlf_done = "data: [DONE]\r\n\r\n"
    chunks = [_chunk({"choices": [{"delta": {"content": "a"}}]}), crlf_done]
    results = [SimpleNamespace(kv_transfer_params={"hash_keys": ["ab"]})]

    out = _stream(chunks, results, monkeypatch=monkeypatch)

    # The normalized terminator is preserved instead of emitting a second one.
    assert out[-1] == crlf_done
    assert out.count(crlf_done) == 1
    assert _DONE not in out


def test_positional_and_keyword_args_are_forwarded(monkeypatch):
    seen: dict = {}
    chunks = [_chunk({"choices": []})]
    results = [SimpleNamespace(kv_transfer_params={"hash_keys": ["ab"]})]

    _stream(chunks, results, seen=seen, monkeypatch=monkeypatch)

    assert seen["args"] == ()
    assert seen["kwargs"] == {}


@pytest.mark.parametrize(
    "chunk",
    [
        "data: \n\n",  # empty payload
        "data: not-json\n\n",  # undecodable
        "data: [1, 2]\n\n",  # decodable but not an object
        "event: ping\n\n",  # not a data chunk
    ],
)
def test_inject_kv_transfer_params_leaves_unparsable_chunks_alone(chunk):
    assert patch_mod._inject_kv_transfer_params(chunk, {"hash_keys": ["ab"]}) == chunk


def test_patch_is_installed_once():
    assert getattr(OpenAIServingChat, patch_mod._PATCHED_ATTR, False)
    assert OpenAIServingChat.chat_completion_stream_generator is patch_mod._patched_chat_completion_stream_generator
