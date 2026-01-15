import pytest

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.skip("310p does not support parallel inference now. Fix me")
def test_models(dtype: str, max_tokens: int) -> None:
    example_prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    with VllmRunner("Qwen/Qwen3-0.6B",
                    tensor_parallel_size=4,
                    dtype=dtype,
                    max_model_len=2048,
                    enforce_eager=True) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
