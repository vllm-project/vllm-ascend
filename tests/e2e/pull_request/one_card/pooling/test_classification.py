import huggingface_hub
import torch
from modelscope import snapshot_download  # type: ignore[import-untyped]
from transformers import AutoModelForSequenceClassification

from tests.e2e.conftest import (
    HfRunner,
    VllmRunner,
    cleanup_dist_env_and_memory,
    wait_until_npu_memory_free,
)


@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen_pooling_classify_correctness() -> None:
    model_name = snapshot_download(
        "Howeee/Qwen2.5-1.5B-apeach",
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is what",
    ]

    with HfRunner(model_name, dtype="float32", auto_cls=AutoModelForSequenceClassification) as hf_runner:
        hf_outputs = hf_runner.classify(prompts)
    cleanup_dist_env_and_memory()

    with VllmRunner(
        model_name,
        runner="pooling",
        max_model_len=None,
        cudagraph_capture_sizes=[4],
    ) as vllm_runner:
        vllm_outputs = vllm_runner.classify(prompts)

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)
        assert torch.allclose(hf_output, vllm_output, 1e-2)
