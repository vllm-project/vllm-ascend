import os
from unittest.mock import patch

from vllm import SamplingParams

from tests.conftest import VllmRunner


def test_qwen25_graph_mode():
    test_qwen_graph_mode("Qwen/Qwen2.5-0.5B-Instruct")


def test_qwen3_graph_mode():
    test_qwen_graph_mode("Qwen/Qwen2.5-0.5B-Instruct")


@patch.dict(os.environ, {"VLLM_ENABLE_GRAPH_MODE": "1"})
def test_qwen_graph_mode(model) -> None:
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=1.0,
                                     top_p=1.0,
                                     max_tokens=10,
                                     top_k=-1,
                                     min_p=0.0,
                                     detokenize=True,
                                     logprobs=1,
                                     n=16)

    with VllmRunner(
            model,
            dtype="half",
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            enforce_eager=False,
            enable_expert_parallel=True,
            max_model_len=4096,
            trust_remote_code=True,
            load_format="dummy",
            gpu_memory_utilization=0.5,
            additional_config={
                "torchair_graph_config": {
                    "enabled": True,
                    "use_cached_graph": False,
                    "graph_batch_sizes_init": False,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                    "chunked_prefill_enabled": True,
                },
                "refresh": True,
            },
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)
