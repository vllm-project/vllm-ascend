import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
]


@pytest.mark.parametrize("model", MODELS)
def test_qwen3_vl_sp_tp2(model: str) -> None:
    prompts = [
        "Hello, my name is",
        "The capital of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    with VllmRunner(
        model,
        max_model_len=1024,
        tensor_parallel_size=2,
        compilation_config={
            "cudagraph_capture_sizes": [2, 4],
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "pass_config": {"enable_sp": False},
        },
        additional_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
    ) as runner:
        no_sp_outputs = runner.model.generate(prompts, sampling_params)

    with VllmRunner(
        model,
        max_model_len=1024,
        tensor_parallel_size=2,
        compilation_config={
            "cudagraph_capture_sizes": [2, 4],
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "pass_config": {"enable_sp": True, "sp_min_token_num": 10},
        },
        additional_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
    ) as runner:
        sp_outputs = runner.model.generate(prompts, sampling_params)

    no_sp_outputs_list = []
    for output in no_sp_outputs:
        no_sp_outputs_list.append((output.outputs[0].index, output.outputs[0].text))

    sp_outputs_list = []
    for output in sp_outputs:
        sp_outputs_list.append((output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=no_sp_outputs_list,
        outputs_1_lst=sp_outputs_list,
        name_0="no_sp_outputs",
        name_1="sp_outputs",
    )


@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen3_moe_dp2_tp2_ep_sp_output_matches_no_sp() -> None:
    """Exercise the DP/TP/EP MoE token layout through the real LLM runner."""
    model = os.environ.get("SP_TEST_MODEL", "Qwen/Qwen3-30B-A3B")
    prompts = [("hello " * 1400) + "\n只回答一个数字：2加2等于几？"]
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    common_kwargs = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "distributed_executor_backend": "mp",
        "max_model_len": 2048,
        "dtype": "bfloat16",
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [2, 4],
            "pass_config": {"sp_min_token_num": 1024},
        },
        "additional_config": {"ascend_compilation_config": {"enable_npugraph_ex": False}},
    }

    outputs = []
    for enable_sp in (False, True):
        kwargs = dict(common_kwargs)
        kwargs["compilation_config"] = {
            **common_kwargs["compilation_config"],
            "pass_config": {"enable_sp": enable_sp, "sp_min_token_num": 1024},
        }
        with DPVllmRunner(model, **kwargs) as runner:
            outputs.append(runner.generate(prompts, sampling_params))

    check_outputs_equal(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="no_sp_outputs",
        name_1="sp_outputs",
    )
