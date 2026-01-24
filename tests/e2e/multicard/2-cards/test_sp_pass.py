import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
]


@pytest.mark.parametrize("model", MODELS)
def test_qwen3_vl_sp_tp2(model: str) -> None:
    prompts = [
        "Hello, my name is", "The capital of the United States is",
        "The capital of France is", "The future of AI is"
    ]
    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            enforce_eager=True,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)

    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            compilation_config={
                "cudagraph_capture_sizes": [2, 4],
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "pass_config": {"enable_sp": True}
            },
            additional_config={"sp_threshold": 10}
    ) as runner:
        sp_outputs = runner.model.generate(
            prompts, sampling_params)

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    sp_outputs_list = []
    for output in sp_outputs:
        sp_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=sp_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="sp_outputs",
    )
