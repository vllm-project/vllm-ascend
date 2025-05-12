import pytest

import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from tests.lora.utils import do_sample

MODEL_PATH = "baichuan-inc/Baichuan-7B"

@pytest.mark.parametrize("fully_sharded", [True, False])
def test_baichuan_tensor_parallel_equality(baichuan_lora_files,
                                           num_gpus_available, fully_sharded):
    if num_gpus_available < 4:
        pytest.skip(f"Not enough GPUs for tensor parallelism {4}")

    llm_tp1 = vllm.LLM(MODEL_PATH,
                       enable_lora=True,
                       max_num_seqs=16,
                       max_loras=4,
                       max_lora_rank=64,
                       trust_remote_code=True,
                       fully_sharded_loras=fully_sharded)
    output_tp1 = do_sample(llm_tp1, baichuan_lora_files, lora_id=1)

    del llm_tp1
    cleanup_dist_env_and_memory()

    llm_tp2 = vllm.LLM(MODEL_PATH,
                       enable_lora=True,
                       max_num_seqs=16,
                       max_loras=4,
                       max_lora_rank=64,
                       tensor_parallel_size=2,
                       trust_remote_code=True,
                       fully_sharded_loras=fully_sharded)
    output_tp2 = do_sample(llm_tp2, baichuan_lora_files, lora_id=2)

    del llm_tp2
    cleanup_dist_env_and_memory()

    assert output_tp1 == output_tp2

    llm_tp4 = vllm.LLM(MODEL_PATH,
                       enable_lora=True,
                       max_num_seqs=16,
                       max_loras=4,
                       max_lora_rank=64,
                       tensor_parallel_size=4,
                       trust_remote_code=True,
                       fully_sharded_loras=fully_sharded)
    output_tp4 = do_sample(llm_tp4, baichuan_lora_files, lora_id=2)

    del llm_tp4
    cleanup_dist_env_and_memory()

    assert output_tp1 == output_tp4