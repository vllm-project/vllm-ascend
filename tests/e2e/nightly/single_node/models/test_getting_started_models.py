#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

import os

import pytest
from vllm import LLM, SamplingParams

from tests.e2e.conftest import RemoteOpenAIServer, cleanup_dist_env_and_memory, wait_until_npu_memory_free

QWEN3_MODEL = "Qwen/Qwen3-0.6B"
QWEN3_W8A8_MODEL = "vllm-ascend/Qwen3-0.6B-W8A8"
PROMPT = "Beijing is a"
IS_310P = "310p" in os.environ.get("VLLM_CI_RUNNER", "").lower()


def _model_kwargs(*, quantized: bool) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    if IS_310P:
        kwargs["dtype"] = "float16"
    if quantized:
        kwargs["quantization"] = "ascend"
    return kwargs


def _serve_args(*, quantized: bool) -> list[str]:
    args: list[str] = []
    if IS_310P:
        args.extend(["--dtype", "float16"])
    if quantized:
        args.extend(["--quantization", "ascend"])
    return args


def _run_offline(model: str, *, quantized: bool) -> None:
    llm = LLM(model=model, **_model_kwargs(quantized=quantized))
    try:
        outputs = llm.generate(PROMPT, SamplingParams(temperature=0, max_tokens=8))
        assert outputs[0].outputs[0].token_ids
        assert outputs[0].outputs[0].text
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        cleanup_dist_env_and_memory()


def _run_online(model: str, *, quantized: bool) -> None:
    with RemoteOpenAIServer(model, _serve_args(quantized=quantized)) as server:
        client = server.get_client()
        model_ids = {item.id for item in client.models.list().data}
        assert model in model_ids

        response = client.completions.create(
            model=model,
            prompt=PROMPT,
            max_tokens=8,
            temperature=0,
        )
        assert response.choices[0].text


@pytest.mark.e2e_model(QWEN3_MODEL)
@wait_until_npu_memory_free()
def test_qwen3_0_6b_offline() -> None:
    _run_offline(QWEN3_MODEL, quantized=False)


@pytest.mark.e2e_model(QWEN3_MODEL)
@wait_until_npu_memory_free()
def test_qwen3_0_6b_online() -> None:
    _run_online(QWEN3_MODEL, quantized=False)


@pytest.mark.e2e_model(QWEN3_W8A8_MODEL)
@pytest.mark.skipif(not IS_310P, reason="The W8A8 fallback is being evaluated for Ascend 310P only.")
@wait_until_npu_memory_free()
def test_qwen3_0_6b_w8a8_offline() -> None:
    _run_offline(QWEN3_W8A8_MODEL, quantized=True)


@pytest.mark.e2e_model(QWEN3_W8A8_MODEL)
@pytest.mark.skipif(not IS_310P, reason="The W8A8 fallback is being evaluated for Ascend 310P only.")
@wait_until_npu_memory_free()
def test_qwen3_0_6b_w8a8_online() -> None:
    _run_online(QWEN3_W8A8_MODEL, quantized=True)
