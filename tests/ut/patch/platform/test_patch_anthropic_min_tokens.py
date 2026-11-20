# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vllm.entrypoints.anthropic.protocol import AnthropicCountTokensRequest, AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.exceptions import VLLMValidationError


@pytest.fixture(scope="module", autouse=True)
def patch_module():
    # Load just this patch: CPU tests need no unrelated platform/device patches.
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_anthropic_min_tokens.py"
    spec = importlib.util.spec_from_file_location("patch_anthropic_min_tokens", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(request):
    return AnthropicServingMessages._build_base_request(request, [{"role": "user", "content": "Hello"}])


@pytest.mark.parametrize("min_tokens", [None, 0, 16, 64])
def test_http_request_reaches_sampling_params(min_tokens):
    app = FastAPI()

    @app.post("/v1/messages")
    def messages(request: AnthropicMessagesRequest):
        converted = convert(request)
        params = converted.to_sampling_params(max_tokens=64, default_sampling_params={})
        return {"min_tokens": params.min_tokens, "stop": params.stop, "max_tokens": params.max_tokens}

    payload = {
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello"}],
        "stop_sequences": ["\n"],
    }
    if min_tokens is not None:
        payload["min_tokens"] = min_tokens
    with TestClient(app) as client:
        response = client.post("/v1/messages", json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == {"min_tokens": min_tokens or 0, "stop": ["\n"], "max_tokens": 64}


@pytest.mark.parametrize("min_tokens", [-1, 65])
def test_upstream_sampling_range_validation(min_tokens):
    request = AnthropicMessagesRequest(
        model="test-model", max_tokens=64, messages=[{"role": "user", "content": "Hello"}], min_tokens=min_tokens
    )
    with pytest.raises(VLLMValidationError, match="min_tokens"):
        convert(request).to_sampling_params(max_tokens=64, default_sampling_params={})


def test_count_tokens_conversion_is_unchanged():
    request = AnthropicCountTokensRequest(model="test-model", messages=[{"role": "user", "content": "Hello"}])
    converted = convert(request)
    assert converted.min_tokens == 0
    assert converted.messages == [{"role": "user", "content": "Hello"}]


def test_native_or_repeated_patch_preserves_converter(patch_module):
    converter = AnthropicServingMessages.__dict__["_build_base_request"].__func__
    patch_module.apply_patch()
    assert AnthropicServingMessages.__dict__["_build_base_request"].__func__ is converter
    request = AnthropicMessagesRequest(
        model="test-model", max_tokens=64, messages=[{"role": "user", "content": "Hello"}], min_tokens=16
    )
    assert convert(request).min_tokens == 16
