import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "deepseek-ai/DeepSeek-V2-Lite",
]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@pytest.mark.parametrize("model", MODELS)
def test_deepseek_v2_lite_enable_shared_expert_dp_tp2(model: str) -> None:

    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']

    prompts = [
        "Hello, my name is", "The capital of the United States is",
        "The capital of France is", "The future of AI is"
    ]
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)

    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "1"
    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            additional_config={
                "enable_shared_expert_dp": True,
            },
    ) as runner:
        shared_expert_dp_eager_outputs = runner.model.generate(
            prompts, sampling_params)

    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            compilation_config={
                "cudagraph_capture_sizes": [1, 4, 8, 16],
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
            additional_config={
                "enable_shared_expert_dp": True,
            },
    ) as runner:
        shared_expert_dp_aclgraph_outputs = runner.model.generate(
            prompts, sampling_params)

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    shared_expert_dp_eager_outputs_list = []
    for output in shared_expert_dp_eager_outputs:
        shared_expert_dp_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    shared_expert_dp_aclgraph_outputs_list = []
    for output in shared_expert_dp_aclgraph_outputs:
        shared_expert_dp_aclgraph_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=shared_expert_dp_eager_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="shared_expert_dp_eager_outputs",
    )

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=shared_expert_dp_aclgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="shared_expert_dp_aclgraph_outputs",
    )
