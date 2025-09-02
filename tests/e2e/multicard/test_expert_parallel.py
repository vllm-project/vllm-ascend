import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal


@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V2-Lite-Chat"])
def test_e2e_ep_correctness(model_name):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    max_tokens = 5

    with VllmRunner(model_name, tensor_parallel_size=2,
                    enforce_eager=True) as vllm_model:
        tp_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    with VllmRunner(model_name,
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                    enforce_eager=True) as vllm_model:
        ep_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=ep_output,
        outputs_1_lst=tp_output,
        name_0="ep_output",
        name_1="tp_output",
    )
