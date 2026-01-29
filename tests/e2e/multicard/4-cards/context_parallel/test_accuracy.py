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
Compare the outputs of vLLM with and without context parallel.

Run `pytest tests/e2e/multicard/long_sequence/test_accuracy.py`.
"""

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen3-8B",
    "vllm-ascend/DeepSeek-V2-Lite-W8A8",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [10])
def test_models_long_sequence_output_between_tp_and_cp(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "The president of the United States is", "The capital of France is"
    ]

    common_kwargs = {
        "max_model_len": 1024,
    }

    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        cp_kwargs = {
            "tensor_parallel_size": 2,
            "decode_context_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "quantization": "ascend",
        }
        tp_kwargs = {
            "tensor_parallel_size": 4,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "quantization": "ascend",
        }

    else:
        cp_kwargs = {
            "tensor_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 2,
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
            },
        }
        tp_kwargs = {
            "tensor_parallel_size": 2,
            "enforce_eager": True,
        }

    cp_full_kwargs = {}
    cp_full_kwargs.update(common_kwargs)  # type: ignore
    cp_full_kwargs.update(cp_kwargs)  # type: ignore

    tp_full_kwargs = {}
    tp_full_kwargs.update(common_kwargs)  # type: ignore
    tp_full_kwargs.update(tp_kwargs)  # type: ignore
    with VllmRunner(model, **cp_full_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(
            prompts, max_tokens)

    with VllmRunner(model, **tp_full_kwargs) as runner:  # type: ignore
        vllm_eager_outputs = runner.generate_greedy(prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs,
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_context_parallel_outputs",
    )


model = "vllm-ascend/DeepSeek-V2-Lite-W8A8"


@pytest.mark.parametrize("max_tokens", [10])
def test_accuracy_dcp_only_graph(max_tokens: int, ) -> None:
    prompts = [
        "The president of the United States is", "The capital of France is"
    ]
    cp_kwargs = {
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
        },
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    tp_kwargs = {
        "tensor_parallel_size": 4,
        "enable_expert_parallel": True,
        "enforce_eager": True,
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    with VllmRunner(model, **cp_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(
            prompts, max_tokens)

    with VllmRunner(model, **tp_kwargs) as runner:  # type: ignore
        vllm_eager_outputs = runner.generate_greedy(prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs,
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_dcp_only_graph_outputs",
    )


@pytest.mark.parametrize("max_tokens", [10])
def test_accuracy_dcp_only_eager(max_tokens: int, ) -> None:
    prompts = [
        "The president of the United States is", "The capital of France is"
    ]
    cp_kwargs = {
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "enforce_eager": True,
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    tp_kwargs = {
        "tensor_parallel_size": 4,
        "enable_expert_parallel": True,
        "enforce_eager": True,
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    with VllmRunner(model, **cp_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(
            prompts, max_tokens)

    with VllmRunner(model, **tp_kwargs) as runner:  # type: ignore
        vllm_eager_outputs = runner.generate_greedy(prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs,
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_dcp_only_eager_outputs",
    )


@pytest.mark.parametrize("max_tokens", [10])
def test_accuracy_pcp_only(max_tokens: int, ) -> None:
    prompts = [
        "The president of the United States is", "The capital of France is"
    ]
    cp_kwargs = {
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 2,
        "enable_expert_parallel": True,
        "enforce_eager": True,
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    tp_kwargs = {
        "tensor_parallel_size": 4,
        "enable_expert_parallel": True,
        "enforce_eager": True,
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    with VllmRunner(model, **cp_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(
            prompts, max_tokens)

    with VllmRunner(model, **tp_kwargs) as runner:  # type: ignore
        vllm_eager_outputs = runner.generate_greedy(prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs,
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_pcp_only_outputs",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [10])
def test_models_long_sequence_cp_kv_interleave_size_output_between_tp_and_cp(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "The president of the United States is", "The capital of France is"
    ]

    common_kwargs = {
        "max_model_len": 1024,
    }

    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        cp_kwargs = {
            "tensor_parallel_size": 2,
            "decode_context_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "enable_expert_parallel": True,
            "cp_kv_cache_interleave_size": 128,
            "enforce_eager": True,
            "quantization": "ascend",
        }
        tp_kwargs = {
            "tensor_parallel_size": 4,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "quantization": "ascend",
        }

    else:
        cp_kwargs = {
            "tensor_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 2,
            "cp_kv_cache_interleave_size": 128,
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
            },
        }
        tp_kwargs = {
            "tensor_parallel_size": 2,
            "enforce_eager": True,
        }

    cp_full_kwargs = {}
    cp_full_kwargs.update(common_kwargs)  # type: ignore
    cp_full_kwargs.update(cp_kwargs)  # type: ignore

    tp_full_kwargs = {}
    tp_full_kwargs.update(common_kwargs)  # type: ignore
    tp_full_kwargs.update(tp_kwargs)  # type: ignore
    with VllmRunner(model, **cp_full_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(
            prompts, max_tokens)

    with VllmRunner(model, **tp_full_kwargs) as runner:  # type: ignore
        vllm_eager_outputs = runner.generate_greedy(prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs,
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="vllm_eager_outputs",
        name_1="vllm_context_parallel_outputs",
    )