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

QWEN = "Qwen/Qwen3-8B"
DS = "vllm-ascend/DeepSeek-V2-Lite-W8A8"

MODELS = [
    QWEN,
    DS,
]

prompts = [
    "The president of the United States is", "The capital of France is"
]
GOLDEN = {
    DS: [
        ([0], 'The president of the United States is a man who has been elected to the highest office'),
        ([0], 'The capital of France is Paris, which is located in the north-central')
    ],
    QWEN: [
        ([0], 'The president of the United States is the head of state and head of government of the'),
        ([0], 'The capital of France is Paris. The capital of Italy is Rome. The')
    ]
}


@pytest.mark.parametrize("model", MODELS)
def test_models_long_sequence_output_between_tp_and_cp(
    model: str,
) -> None:


    common_kwargs = {
        "max_model_len": 1024,
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "long_prefill_token_threshold": 4,
    }

    if model == DS:
        cp_kwargs = {
            "decode_context_parallel_size": 2,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "quantization": "ascend",
        }

    else:
        cp_kwargs = {
            "decode_context_parallel_size": 1,
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
            },
        }

    cp_full_kwargs = {}
    cp_full_kwargs.update(common_kwargs)  # type: ignore
    cp_full_kwargs.update(cp_kwargs)  # type: ignore

    with VllmRunner(model, **cp_full_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(prompts, 10)
        vllm_context_parallel_outputs = [([0], text) for tokens, text in vllm_context_parallel_outputs]

    check_outputs_equal(
        outputs_0_lst=GOLDEN[model],
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="golden_outputs",
        name_1="vllm_context_parallel_outputs",
    )


def test_accuracy_dcp_only_graph() -> None:
    cp_kwargs = {
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": True,
        'long_prefill_token_threshold': 4,
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
        },
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    with VllmRunner(DS, **cp_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(prompts, 10)
        vllm_context_parallel_outputs = [([0], text) for tokens, text in vllm_context_parallel_outputs]

    check_outputs_equal(
        outputs_0_lst=GOLDEN[DS],
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="golden_outputs",
        name_1="vllm_dcp_only_graph_outputs",
    )


def test_accuracy_dcp_only_eager() -> None:
    cp_kwargs = {
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "long_prefill_token_threshold": 4,
        "enforce_eager": True,
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    with VllmRunner(DS, **cp_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(prompts, 10)
        vllm_context_parallel_outputs = [([0], text) for tokens, text in vllm_context_parallel_outputs]

    check_outputs_equal(
        outputs_0_lst=GOLDEN[DS],
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="golden_outputs",
        name_1="vllm_dcp_only_eager_outputs",
    )


def test_accuracy_pcp_only() -> None:
    cp_kwargs = {
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 2,
        "enable_expert_parallel": True,
        "long_prefill_token_threshold": 4,
        "enforce_eager": True,
        "quantization": "ascend",
        "max_model_len": 1024,
    }
    with VllmRunner(DS, **cp_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(prompts, 10)
        vllm_context_parallel_outputs = [([0], text) for tokens, text in vllm_context_parallel_outputs]

    check_outputs_equal(
        outputs_0_lst=GOLDEN[DS],
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="golden_outputs",
        name_1="vllm_pcp_only_outputs",
    )


@pytest.mark.parametrize("model", MODELS)
def test_models_long_sequence_cp_kv_interleave_size_output_between_tp_and_cp(
    model: str,
) -> None:
    common_kwargs = {
        "max_model_len": 1024,
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "cp_kv_cache_interleave_size": 128,
        "long_prefill_token_threshold": 4,
    }

    if model == DS:
        cp_kwargs = {
            "decode_context_parallel_size": 2,
            "enable_expert_parallel": True,
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
            },
            "quantization": "ascend",
        }
    else:
        cp_kwargs = {
            "decode_context_parallel_size": 1,
            "enforce_eager": True,
        }

    cp_full_kwargs = {}
    cp_full_kwargs.update(common_kwargs)  # type: ignore
    cp_full_kwargs.update(cp_kwargs)  # type: ignore

    with VllmRunner(model, **cp_full_kwargs) as runner:  # type: ignore
        vllm_context_parallel_outputs = runner.generate_greedy(prompts, 10)
        vllm_context_parallel_outputs = [([0], text) for tokens, text in vllm_context_parallel_outputs]

    check_outputs_equal(
        outputs_0_lst=GOLDEN[model],
        outputs_1_lst=vllm_context_parallel_outputs,
        name_0="golden_outputs",
        name_1="vllm_context_parallel_outputs",
    )
