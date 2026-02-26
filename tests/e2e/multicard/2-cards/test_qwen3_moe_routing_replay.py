import os
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind


@patch.dict(os.environ, {"OMP_NUM_THREADS": "1"})
def test_qwen3_moe_routing_replay():
    prompts = [
        "Hello, please introduce yourself.",
    ]
    with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            distributed_executor_backend="mp",
            enable_return_routed_experts=True,
    ) as vllm_model:
        sampling_params = SamplingParams(
            max_tokens=5,
            temperature=0.8,
            top_p=0.95,
            output_kind=RequestOutputKind.FINAL_ONLY
        )
        inputs = vllm_model.get_inputs(prompts=prompts)
        outputs = vllm_model.model.generate(prompts=inputs, sampling_params=sampling_params)
        assert outputs[0].finished
        assert len(outputs[0].outputs[0].text) > 0
        assert outputs[0].outputs[0].routed_experts.size > 0
