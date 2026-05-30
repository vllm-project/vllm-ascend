import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from tests.e2e.conftest import DPVllmRunner

MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("enforce_eager", [True, False])
@patch.dict(os.environ, {"OMP_NUM_THREADS": "1"})
def test_moe_routing_replay(model, enforce_eager):
    prompts = [
        "Hello, please introduce yourself.",
    ]
    with DPVllmRunner(
        model,
        enforce_eager=enforce_eager,
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        distributed_executor_backend="mp",
        enable_return_routed_experts=True,
    ) as vllm_model:
        sampling_params = SamplingParams(
            max_tokens=5, temperature=0.8, top_p=0.95, output_kind=RequestOutputKind.FINAL_ONLY
        )
        inputs = vllm_model.get_inputs(prompts=prompts)
        outputs = vllm_model.generate(prompts=inputs, sampling_params=sampling_params)
        output_ids, output_strs, routed_experts = outputs[0]
        assert len(output_ids) > 0
        assert len(output_strs) > 0
        assert routed_experts is not None
        assert len(routed_experts) > 0
        assert routed_experts[0] is not None
        assert routed_experts[0].size > 0
