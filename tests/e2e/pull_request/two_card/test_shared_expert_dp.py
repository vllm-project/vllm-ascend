import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import compare_logprobs

MODELS = [
    "deepseek-ai/DeepSeek-V2-Lite",
]

PROMPTS = [
    "Hello, what's your name?",
    "The capital of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("model", MODELS)
def test_deepseek_v2_lite_enable_shared_expert_dp_tp2_eager(model: str, monkeypatch) -> None:
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    # Shared-expert-DP must stay numerically consistent with the plain eager
    # baseline. `additional_config` is excluded from the baseline by
    # compare_logprobs, so the baseline runs without the feature.
    compare_logprobs(
        runner_kwargs={
            "model_name": model,
            "max_model_len": 1024,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 256,
            "enforce_eager": True,
            "tensor_parallel_size": 2,
            "enable_expert_parallel": True,
            "additional_config": {
                "enable_shared_expert_dp": True,
            },
        },
        prompts=PROMPTS,
    )
