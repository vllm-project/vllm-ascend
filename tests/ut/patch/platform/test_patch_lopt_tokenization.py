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

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import vllm_ascend.patch.platform.patch_lopt_tokenization as lopt_patch


class _Params:
    @staticmethod
    def get_encode_kwargs() -> dict[str, Any]:
        return {"add_special_tokens": False}


class _FakeLopt:
    def __init__(self, usable: bool = True) -> None:
        self.usable = usable
        self.encoded_texts: list[str] = []

    def can_use(self, text: str, encode_kwargs: dict[str, Any]) -> bool:
        return self.usable

    def encode(self, text: str, **encode_kwargs: Any) -> list[int]:
        self.encoded_texts.append(text)
        return [11, 12, 13]


def test_attach_lopt_to_deepseek_v4_compatible_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = SimpleNamespace(is_fast=True)
    renderer = SimpleNamespace(tokenizer=tokenizer, mm_processor=None)
    expected_lopt = object()

    monkeypatch.setattr(
        lopt_patch,
        "_config_from_env",
        lambda: SimpleNamespace(thread_workers=4),
    )
    monkeypatch.setattr(lopt_patch, "maybe_make_thread_pool", lambda tokenizer, copies: tokenizer)
    monkeypatch.setattr(lopt_patch, "LosslessParallelTokenizer", lambda tokenizer, config: expected_lopt)

    lopt_patch._attach_lopt(renderer)

    assert renderer._ascend_lopt_tokenizer is expected_lopt


@pytest.mark.parametrize(
    ("tokenizer", "mm_processor"),
    [
        (None, None),
        (SimpleNamespace(is_fast=False), None),
        (SimpleNamespace(is_fast=True), object()),
    ],
)
def test_attach_lopt_skips_unsupported_renderer(tokenizer: Any, mm_processor: Any) -> None:
    renderer = SimpleNamespace(tokenizer=tokenizer, mm_processor=mm_processor)

    lopt_patch._attach_lopt(renderer)

    assert not hasattr(renderer, "_ascend_lopt_tokenizer")


def test_deepseek_v4_sync_tokenization_uses_lopt() -> None:
    lopt = _FakeLopt()
    renderer = SimpleNamespace(_ascend_lopt_tokenizer=lopt)

    result = lopt_patch._patched_deepseek_v4_renderer_tokenize_prompt(
        renderer,
        {"prompt": "a sufficiently long prompt"},
        _Params(),
    )

    assert result["prompt_token_ids"] == [11, 12, 13]
    assert lopt.encoded_texts == ["a sufficiently long prompt"]


def test_deepseek_v4_async_tokenization_uses_lopt() -> None:
    lopt = _FakeLopt()
    renderer = SimpleNamespace(_ascend_lopt_tokenizer=lopt, _executor=None)

    result = asyncio.run(
        lopt_patch._patched_deepseek_v4_renderer_tokenize_prompt_async(
            renderer,
            {"prompt": "a sufficiently long prompt"},
            _Params(),
        )
    )

    assert result["prompt_token_ids"] == [11, 12, 13]
    assert lopt.encoded_texts == ["a sufficiently long prompt"]


def test_deepseek_v4_async_tokenization_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    lopt = _FakeLopt(usable=False)
    renderer = SimpleNamespace(_ascend_lopt_tokenizer=lopt, _executor=None)
    fallback_result = {"prompt": "short", "prompt_token_ids": [99]}

    async def fallback(renderer, prompt, params):
        return fallback_result

    monkeypatch.setattr(lopt_patch, "_original_deepseek_v4_renderer_tokenize_prompt_async", fallback)

    result = asyncio.run(
        lopt_patch._patched_deepseek_v4_renderer_tokenize_prompt_async(
            renderer,
            {"prompt": "short"},
            _Params(),
        )
    )

    assert result is fallback_result
    assert lopt.encoded_texts == []
