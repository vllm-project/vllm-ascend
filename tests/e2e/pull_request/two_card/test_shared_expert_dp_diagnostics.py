import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.accuracy_probe import compare_logprobs_probe
from tests.e2e.pull_request.two_card.test_shared_expert_dp import FEATURE_CONFIGS, MODELS, PROMPTS

PROBLEM_FEATURE_CONFIGS = FEATURE_CONFIGS[1:]
EXECUTION_CONFIGS = [
    pytest.param({"enforce_eager": True}, id="eager"),
    pytest.param(
        {
            "compilation_config": {
                "cudagraph_capture_sizes": [1, 4, 8, 16],
                "cudagraph_mode": "FULL_DECODE_ONLY",
            }
        },
        id="graph",
    ),
]


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("feature_config", PROBLEM_FEATURE_CONFIGS)
@pytest.mark.parametrize("execution_config", EXECUTION_CONFIGS)
def test_shared_expert_dp_with_diagnostics(
    model: str,
    feature_config: dict[str, bool],
    execution_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    compare_logprobs_probe(
        label=f"shared-expert-diagnostics-{request.node.callspec.id}",
        runner_kwargs={
            "model_name": model,
            "max_model_len": 1024,
            "tensor_parallel_size": 2,
            "enable_expert_parallel": True,
            "additional_config": feature_config,
            **execution_config,
        },
        prompts=PROMPTS,
    )
