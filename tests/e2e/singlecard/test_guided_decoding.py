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
import json
import os
from typing import Any, Dict

import jsonschema
import pytest
import regex as re
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.10.2"):
    from vllm.sampling_params import \
        GuidedDecodingParams as StructuredOutputsParams
else:
    from vllm.sampling_params import StructuredOutputsParams

from tests.e2e.conftest import VllmRunner

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
MODEL_NAME = "Qwen/Qwen3-0.6B"

GuidedDecodingBackend = ["xgrammar", "guidance", "outlines"]


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


def construct_sampling_params(
        struct_param, sampling_kwargs: Dict[str, Any]) -> SamplingParams:
    if vllm_version_is("0.10.2"):
        return SamplingParams(guided_decoding=struct_param, **sampling_kwargs)
    else:
        return SamplingParams(structured_outputs=struct_param,
                              **sampling_kwargs)


@pytest.mark.parametrize("guided_decoding_backend", GuidedDecodingBackend)
def test_guided_json_completion(guided_decoding_backend: str,
                                sample_json_schema):
    struct_output_params = StructuredOutputsParams(json=sample_json_schema, )
    sampling_params = construct_sampling_params(struct_output_params, {
        "temperature": 1.0,
        "max_tokens": 500,
    })

    runner_kwargs = {
        "seed": 0,
    }
    if vllm_version_is("0.10.2"):
        runner_kwargs["guided_decoding_backend"] = guided_decoding_backend
    else:
        runner_kwargs["structured_outputs_config"] = {
            "backend": guided_decoding_backend
        }

    with VllmRunner(MODEL_NAME, **runner_kwargs) as vllm_model:
        prompts = [
            f"Give an example JSON for an employee profile "
            f"that fits this schema: {sample_json_schema}"
        ] * 2
        inputs = vllm_model.get_inputs(prompts)
        outputs = vllm_model.model.generate(inputs,
                                            sampling_params=sampling_params)

        assert outputs is not None

        for output in outputs:
            assert output is not None
            assert isinstance(output, RequestOutput)
            prompt = output.prompt

            generated_text = output.outputs[0].text
            assert generated_text is not None
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            output_json = json.loads(generated_text)
            jsonschema.validate(instance=output_json,
                                schema=sample_json_schema)


@pytest.mark.parametrize("guided_decoding_backend", GuidedDecodingBackend)
def test_guided_regex(guided_decoding_backend: str, sample_regex):
    if guided_decoding_backend == "outlines":
        pytest.skip("Outlines doesn't support regex-based guided decoding.")

    struct_output_params = StructuredOutputsParams(json=sample_regex, )
    sampling_params = construct_sampling_params(struct_output_params, {
        "temperature": 0.8,
        "top_p": 0.95,
    })

    runner_kwargs = {
        "seed": 0,
    }
    if vllm_version_is("0.10.2"):
        runner_kwargs["guided_decoding_backend"] = guided_decoding_backend
    else:
        runner_kwargs["structured_outputs_config"] = {
            "backend": guided_decoding_backend
        }

    with VllmRunner(MODEL_NAME, **runner_kwargs) as vllm_model:
        prompts = [
            f"Give an example IPv4 address with this regex: {sample_regex}"
        ] * 2
        inputs = vllm_model.get_inputs(prompts)
        outputs = vllm_model.model.generate(inputs,
                                            sampling_params=sampling_params)
        assert outputs is not None
        for output in outputs:
            assert output is not None
            assert isinstance(output, RequestOutput)
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(generated_text)
            assert generated_text is not None
            assert re.fullmatch(".*", generated_text) is not None
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
