#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/entrypoints/llm/test_guided_generate.py
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
#
import gc
import json
import os
import re
import weakref

import jsonschema
import pytest
import torch
from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GUIDED_DECODING_BACKENDS = [
    "outlines",
    "lm-format-enforcer",
    "xgrammar:disable-any-whitespace",
]


@pytest.fixture(scope="module")
def sample_regex():
    return (r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
            r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)")


@pytest.fixture(scope="module")
def sample_json_schema():
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 10
                },
                "minItems": 3
            },
            "work_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string"
                        },
                        "duration": {
                            "type": "number"
                        },
                        "position": {
                            "type": "string"
                        }
                    },
                    "required": ["company", "position"]
                }
            }
        },
        "required": ["name", "age", "skills", "work_history"]
    }


def clean_up():
    gc.collect()
    torch.npu.empty_cache()


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model=MODEL_NAME, max_model_len=1024, seed=0)
    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)
        del llm
    clean_up()


# TODO: Add v1 fully tested
@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "1",
                    reason="v1 does not support guided decoding")
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_regex(sample_regex, llm, guided_decoding_backend: str):
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     guided_decoding=GuidedDecodingParams(
                                         regex=sample_regex,
                                         backend=guided_decoding_backend))
    print(f"Using backend: {guided_decoding_backend}")
    outputs = llm.generate(prompts=[
        f"Give an example IPv4 address with this regex: {sample_regex}"
    ] * 2,
                           sampling_params=sampling_params,
                           use_tqdm=True)

    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(generated_text)
        assert generated_text is not None
        assert re.fullmatch(sample_regex, generated_text) is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "1",
                    reason="v1 does not support guided decoding")
@pytest.mark.parametrize("guided_decoding_backend", GUIDED_DECODING_BACKENDS)
def test_guided_json_completion(sample_json_schema, llm,
                                guided_decoding_backend: str):
    if guided_decoding_backend == "xgrammar:disable-any-whitespace":
        # xgrammar does not support json schema, will fall back to outlines, skip it
        pytest.skip(
            f"{guided_decoding_backend} does not support json schema validation"
        )

    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=1000,
                                     guided_decoding=GuidedDecodingParams(
                                         json=sample_json_schema,
                                         backend=guided_decoding_backend))
    print(f"Using backend: {guided_decoding_backend}")
    outputs = llm.generate(prompts=[
        f"Give an example JSON for an employee profile "
        f"that fits this schema: {sample_json_schema}"
    ] * 2,
                           sampling_params=sampling_params,
                           use_tqdm=True)

    assert outputs is not None

    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        prompt = output.prompt

        generated_text = output.outputs[0].text
        assert generated_text is not None
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=sample_json_schema)
