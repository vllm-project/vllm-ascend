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
def test_deepseek_v2_lite_enable_shared_expert_dp_tp2_graph(model: str, monkeypatch) -> None:
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    # Shared-expert-DP + FULL_DECODE_ONLY must stay numerically consistent
    # with the plain eager baseline. `additional_config` and
    # `compilation_config` are excluded from the baseline by compare_logprobs.
    compare_logprobs(
        runner_kwargs={
            "model_name": model,
            "max_model_len": 1024,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 256,
            "tensor_parallel_size": 2,
            "enable_expert_parallel": True,
            "compilation_config": {
                "cudagraph_capture_sizes": [4],
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
            "additional_config": {
                "enable_shared_expert_dp": True,
            },
        },
        prompts=PROMPTS,
    )
