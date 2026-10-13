import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.lora.test_qwen35_densemodel_lora import (
    MODEL_PATH,
    assert_qwen35_text_lora,
)


@wait_until_npu_memory_free(target_free_percentage=0.7)
@pytest.mark.parametrize("fully_sharded_loras", [False, True])
def test_qwen35_text_lora(qwen35_text_lora_files, fully_sharded_loras):
    with VllmRunner(
        model_name=MODEL_PATH,
        max_model_len=4096,
        enable_lora=True,
        max_loras=2,
        max_num_seqs=4,
        max_lora_rank=8,
        fully_sharded_loras=fully_sharded_loras,
        tensor_parallel_size=2,
    ) as vllm_runner:
        assert_qwen35_text_lora(
            vllm_runner.model,
            qwen35_text_lora_files,
        )
