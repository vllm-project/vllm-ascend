#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""
Compare the outputs of vLLM with and without aclgraph.

Run `pytest tests/compile/test_aclgraph_accuracy.py`.
"""

import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS_CFG = [
    {
        "name": "Qwen/Qwen3-0.6B",
        "quantization": None,
    },
    {
        "name": "vllm-ascend/DeepSeek-V2-Lite-W8A8",
        "quantization": "ascend",
    },
]

FULL_DECODE_CASES = [
    {
        "model_cfg": MODELS_CFG[0],
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
    },
    {
        "model_cfg": MODELS_CFG[1],
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
    },
]

FULL_DECODE_NPUGRAPH_EX_CASES = [
    {
        "model_cfg": MODELS_CFG[0],
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        "additional_config": {
            "enable_npugraph_ex": True
        }
    },
    {
        "model_cfg": MODELS_CFG[1],
        "compilation_config": {
            "cudagraph_capture_sizes": [4, 8, 32, 64],
            "cudagraph_mode": "FULL_DECODE_ONLY"
        },
        "additional_config": {
            "enable_npugraph_ex": True
        },
    },
]

PROMPTS_BASIC = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

PROMPTS_FULL_GRAPH = [
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$'
     'be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$,'
     '$\\angle BDC = 90^\\circ$. Suppose $AD = 1$ and $\\frac{BD}{CD} = \\frac{3}{2}$.'
     'If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$,'
     'where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.'
     ),
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'Let $ABCD$ be a unit square in the plane. Points $X$ and $Y$ are chosen'
     'independently and uniformly at random on the perimeter of $ABCD$.'
     'If the expected value of the area of triangle $\\triangle AXY$'
     'can be expressed as $\\frac{m}{n}$, for relatively prime positive'
     'integers $m$ and $n$, compute $m+n$.'),
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'Let $a, b, c$ be distinct numbers such that the equations $x^2 + ax + 1 = 0$'
     'and $x^2 + bx + c = 0$ have a common real root, and the equations $x^2 + x + a = 0$'
     'and $x^2 + cx + b = 0$ also have a common real root.'
     'Compute the sum $a + b + c$.')
]


def run_vllm(
    model_name: str,
    sampling_params,
    prompts: str | list,
    *,
    enforce_eager: bool,
    quantization: str | None,
    compilation_config: dict | None = None,
    additional_config: dict | None = None,
):
    runner_kwargs = {
        "model_name": model_name,
        "max_model_len": 1024,
        "enforce_eager": enforce_eager,
    }
    if quantization is not None:
        runner_kwargs["quantization"] = quantization
    if compilation_config is not None:
        runner_kwargs["compilation_config"] = compilation_config
    if additional_config is not None:
        runner_kwargs.update(additional_config)

    with VllmRunner(**runner_kwargs) as runner:
        outputs = runner.model.generate(prompts, sampling_params)

    return [(out.outputs[0].index, out.outputs[0].text) for out in outputs]


@pytest.mark.parametrize("model_cfg", MODELS_CFG)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_output_between_eager_and_aclgraph(
    model_cfg,
    max_tokens: int,
) -> None:
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    eager_outputs = run_vllm(
        model_cfg["name"],
        sampling_params,
        prompts=PROMPTS_BASIC,
        enforce_eager=True,
        quantization=model_cfg["quantization"],
    )

    aclgraph_outputs = run_vllm(
        model_cfg["name"],
        sampling_params,
        prompts=PROMPTS_BASIC,
        enforce_eager=False,
        quantization=model_cfg["quantization"],
    )

    check_outputs_equal(
        outputs_0_lst=eager_outputs,
        outputs_1_lst=aclgraph_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )


@pytest.mark.parametrize("full_decode_config", FULL_DECODE_CASES)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_output_between_eager_and_full_decode_only(
    full_decode_config: str,
    max_tokens: int,
) -> None:
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
    # NOTE: Randomly fill the prompt with the requested amount for
    # the specified capture shape to prevent accuracy issues caused by padding

    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     top_k=1)
    model_cfg = full_decode_config['model_cfg']
    eager_outputs = run_vllm(
        model_cfg["name"],
        sampling_params,
        enforce_eager=True,
        prompts=PROMPTS_FULL_GRAPH,
        quantization=model_cfg["quantization"],
        compilation_config=full_decode_config['compilation_config'],
    )

    aclgraph_outputs = run_vllm(
        model_cfg["name"],
        sampling_params,
        enforce_eager=False,
        prompts=PROMPTS_FULL_GRAPH,
        quantization=model_cfg["quantization"],
        compilation_config=full_decode_config['compilation_config'],
    )

    check_outputs_equal(
        outputs_0_lst=eager_outputs,
        outputs_1_lst=aclgraph_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )


@pytest.mark.parametrize("full_decode_ex_config",
                         FULL_DECODE_NPUGRAPH_EX_CASES)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_output_between_eager_and_fullgraph_npugraph_ex(
    full_decode_ex_config: str,
    max_tokens: int,
) -> None:
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']

    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     n=1,
                                     temperature=0.0,
                                     top_p=1.0,
                                     top_k=1)
    model_cfg = full_decode_ex_config['model_cfg']
    eager_outputs = run_vllm(
        model_cfg["name"],
        sampling_params,
        enforce_eager=True,
        prompts=PROMPTS_FULL_GRAPH,
        quantization=model_cfg["quantization"],
        compilation_config=full_decode_ex_config['compilation_config'],
    )

    aclgraph_outputs = run_vllm(
        model_cfg["name"],
        sampling_params,
        enforce_eager=False,
        prompts=PROMPTS_FULL_GRAPH,
        quantization=model_cfg["quantization"],
        compilation_config=full_decode_ex_config['compilation_config'],
    )

    check_outputs_equal(
        outputs_0_lst=eager_outputs,
        outputs_1_lst=aclgraph_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )


def test_aclgraph_enable():
    # Generally, this test is not belong to e2e, but it is a good way to check if
    # aclgraph is enabled in real environment
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs

    from vllm_ascend.platform import NPUPlatform

    # vLLM default mode is piecewise cudagraph
    config = EngineArgs()
    VllmConfig = config.create_engine_config()
    assert VllmConfig.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE

    # after check_and_update_config, mode should be VLLM_COMPILE and piecewise cudagraph
    NPUPlatform.check_and_update_config(VllmConfig)
    assert VllmConfig.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert VllmConfig.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
