import pytest

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("max_tokens", [5])
def test_qwen3_w8a8_e2e_310p(dtype: str, max_tokens: int) -> None:
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    ]

    with VllmRunner(
        "vllm-ascend/Qwen3-32B-W8A8",
        tensor_parallel_size=4,
        dtype=dtype,
        max_model_len=8192,
        enforce_eager=True,
        quantization="ascend",
        enable_prefix_caching=False,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
