import pytest

from tests.conftest import VllmRunner
from tests.e2e.singlecard.test_ilama_lora import (EXPECTED_LORA_OUTPUT,
                                                  MODEL_PATH, do_sample)


@pytest.mark.parametrize("distributed_executor_backend", ["mp"])
def test_ilama_lora_tp2(distributed_executor_backend, ilama_lora_files):
    with VllmRunner(model_name=MODEL_PATH,
                    enable_lora=True,
                    max_loras=4,
                    max_model_len=1024,
                    max_num_seqs=16,
                    tensor_parallel_size=2,
                    distributed_executor_backend=distributed_executor_backend
                    ) as vllm_model:
        output = do_sample(vllm_model.model, ilama_lora_files, lora_id=2)

    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output[i] == EXPECTED_LORA_OUTPUT[i]
