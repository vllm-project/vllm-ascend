import pytest
from vllm.distributed import cleanup_dist_env_and_memory

from tests.conftest import VllmRunner
from tests.singlecard.test_lora_quant import MODELS, do_sample


@pytest.mark.parametrize("model", MODELS)
def test_quant_model_tp_equality(tinyllama_lora_files, model):
    if model.quantization == "GPTQ":
        pytest.skip("GPTQ lora outputs are just incredibly unstable")
    with VllmRunner(model=model.model_path,
                    quantization=model.quantization,
                    enable_lora=True,
                    max_loras=4,
                    gpu_memory_utilization=0.7,
                    max_num_seqs=16) as vllm_model_tp1:
        output_tp1 = do_sample(vllm_model_tp1, tinyllama_lora_files, lora_id=1)

    del vllm_model_tp1
    cleanup_dist_env_and_memory()

    with VllmRunner(model=model.model_path,
                    quantization=model.quantization,
                    enable_lora=True,
                    max_loras=4,
                    tensor_parallel_size=2,
                    gpu_memory_utilization=0.7,
                    max_num_seqs=16) as vllm_model_tp2:
        output_tp2 = do_sample(vllm_model_tp2, tinyllama_lora_files, lora_id=1)

    del vllm_model_tp2
    cleanup_dist_env_and_memory()

    assert output_tp1 == output_tp2
