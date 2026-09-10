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

import jsonschema
import pytest
import regex as re
from vllm.exceptions import VLLMValidationError
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from tests.e2e.conftest import ModelName

os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_REGEX_COMPILATION_TIMEOUT_S"] = "30"

MODEL_NAME = ModelName.QWEN3_06B
JSON_MAX_TOKENS = 256
REGEX_MAX_TOKENS = 16


def _model_mark(backend: str):
    return pytest.mark.model(
        model_name=MODEL_NAME,
        max_model_len=1024,
        max_num_seqs=8,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]},
        extra_kwargs={
            "seed": 0,
            "max_num_batched_tokens": 256,
            "structured_outputs_config": {"backend": backend},
        },
    )


@pytest.fixture(scope="module")
def sample_regex():
    return (
        r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
        r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)"
    )


@pytest.fixture(scope="module")
def sample_json_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {"type": "array", "items": {"type": "string", "maxLength": 10}, "minItems": 3},
            "work_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "duration": {"type": "number"},
                        "position": {"type": "string"},
                    },
                    "required": ["company", "position"],
                },
            },
        },
        "required": ["name", "age", "skills", "work_history"],
    }


def _assert_json_outputs(outputs, sample_json_schema) -> None:
    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        generated_text = output.outputs[0].text
        assert generated_text is not None
        output_json = json.loads(generated_text)
        jsonschema.validate(instance=output_json, schema=sample_json_schema)


def _assert_regex_outputs(outputs) -> None:
    assert outputs is not None
    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)
        generated_text = output.outputs[0].text
        assert generated_text is not None
        assert re.fullmatch(".*", generated_text) is not None


def _generate_json(vllm_runner, sample_json_schema):
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=JSON_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=sample_json_schema),
    )
    prompts = [f"Give an example JSON for an employee profile that fits this schema: {sample_json_schema}"] * 2
    return vllm_runner.model.generate(vllm_runner.get_inputs(prompts), sampling_params=sampling_params)


def _generate_regex(vllm_runner, sample_regex):
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=REGEX_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(regex=sample_regex),
    )
    prompts = [f"Give an example IPv4 address with this regex: {sample_regex}"] * 2
    return vllm_runner.model.generate(vllm_runner.get_inputs(prompts), sampling_params=sampling_params)


@pytest.mark.timeout(1000)
@_model_mark("xgrammar")
def test_guided_xgrammar(sample_json_schema, sample_regex, vllm_runner):
    """JSON + regex on one xgrammar engine (same @pytest.mark.model cache key)."""
    _assert_json_outputs(_generate_json(vllm_runner, sample_json_schema), sample_json_schema)
    _assert_regex_outputs(_generate_regex(vllm_runner, sample_regex))


@pytest.mark.timeout(1000)
@_model_mark("guidance")
def test_guided_guidance(sample_json_schema, sample_regex, vllm_runner):
    """JSON + regex on one guidance engine."""
    _assert_json_outputs(_generate_json(vllm_runner, sample_json_schema), sample_json_schema)
    _assert_regex_outputs(_generate_regex(vllm_runner, sample_regex))


@pytest.mark.timeout(1000)
@_model_mark("auto")
def test_guided_auto_rejects_mixed_structured_output_backends(vllm_runner):
    xgrammar_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    guidance_schema = {
        "type": "object",
        "properties": {"count": {"type": "integer", "multipleOf": 2}},
        "required": ["count"],
    }

    xgrammar_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        structured_outputs=StructuredOutputsParams(json=xgrammar_schema),
    )
    prompts = [f"Give an example JSON that fits this schema: {xgrammar_schema}"]
    outputs = vllm_runner.model.generate(vllm_runner.get_inputs(prompts), sampling_params=xgrammar_params)
    assert outputs is not None
    assert outputs[0] is not None

    guidance_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        structured_outputs=StructuredOutputsParams(json=guidance_schema),
    )
    prompts = [f"Give an example JSON that fits this schema: {guidance_schema}"]
    with pytest.raises(VLLMValidationError, match="already using 'xgrammar'.*'guidance'"):
        vllm_runner.model.generate(vllm_runner.get_inputs(prompts), sampling_params=guidance_params)


@pytest.mark.timeout(1000)
@_model_mark("outlines")
def test_guided_json_completion_outlines(sample_json_schema, vllm_runner):
    _assert_json_outputs(_generate_json(vllm_runner, sample_json_schema), sample_json_schema)
