import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import compare_logprobs

MODELS = [
    "deepseek-ai/DeepSeek-V2-Lite",
]

PROMPTS = [
    "Hello, my name is",
    "The capital of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("model", MODELS)
def test_deepseek_v2_lite_enable_shared_expert_dp_tp2(model: str, monkeypatch) -> None:
    # FlashComm v1 / shared-expert-DP require HCCL_OP_EXPANSION_MODE to be unset.
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    # FlashComm1 + shared-expert-DP must stay numerically consistent with the
    # plain eager baseline.  `additional_config` is excluded from the baseline
    # by compare_logprobs, so the baseline runs without either flag.
    shared_expert_dp_config = {
        "enable_flashcomm1": True,
        "enable_shared_expert_dp": True,
    }

    # Eager mode: FlashComm1 + shared-expert-DP vs eager baseline.
    compare_logprobs(
        runner_kwargs={
            "model_name": model,
            "max_model_len": 1024,
            "enforce_eager": True,
            "tensor_parallel_size": 2,
            "enable_expert_parallel": True,
            "additional_config": shared_expert_dp_config,
        },
        prompts=PROMPTS,
    )

    # ACLGraph (FULL_DECODE_ONLY): FlashComm1 + shared-expert-DP vs eager baseline.
    compare_logprobs(
        runner_kwargs={
            "model_name": model,
            "max_model_len": 1024,
            "tensor_parallel_size": 2,
            "enable_expert_parallel": True,
            "compilation_config": {
                "cudagraph_capture_sizes": [1, 4, 8, 16],
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
            "additional_config": shared_expert_dp_config,
        },
        prompts=PROMPTS,
    )
