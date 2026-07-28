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
@pytest.mark.parametrize("enable_flashcomm1", [False, True], ids=["shared-expert-dp-only", "with-flashcomm1"])
def test_deepseek_v2_lite_enable_shared_expert_dp_tp2(
    model: str,
    enable_flashcomm1: bool,
    monkeypatch,
) -> None:
    # FlashComm v1 / shared-expert-DP require HCCL_OP_EXPANSION_MODE to be unset.
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    # Shared-expert-DP must stay numerically consistent with the plain eager
    # baseline both independently and together with FlashComm1.
    # `additional_config` is excluded from the baseline by compare_logprobs, so
    # the baseline runs without either flag.
    shared_expert_dp_config = {
        "enable_flashcomm1": enable_flashcomm1,
        "enable_shared_expert_dp": True,
    }

    # Eager mode: requested feature combination vs eager baseline.
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

    # ACLGraph (FULL_DECODE_ONLY): requested combination vs eager baseline.
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
