import os

import pytest

from tests.conftest import VllmRunner
from tests.singlecard.test_lora_quant import MODELS, do_sample

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["VLLM_USE_MODELSCOPE"] = "True"


@pytest.mark.parametrize("model", MODELS)
def test_quant_model_tp_equality(tinyllama_lora_files, model):
    if model.quantization == "GPTQ":
        pytest.skip("GPTQ lora outputs are just incredibly unstable")
    with VllmRunner(model_name=model.model_path,
                    quantization=model.quantization,
                    enable_lora=True,
                    max_loras=4,
                    gpu_memory_utilization=0.7,
                    max_num_seqs=16) as vllm_model_tp1:
        output_tp1 = do_sample(vllm_model_tp1.model,
                               tinyllama_lora_files,
                               lora_id=1)

    with VllmRunner(model_name=model.model_path,
                    quantization=model.quantization,
                    enable_lora=True,
                    max_loras=4,
                    tensor_parallel_size=2,
                    gpu_memory_utilization=0.7,
                    max_num_seqs=16) as vllm_model_tp2:
        output_tp2 = do_sample(vllm_model_tp2.model,
                               tinyllama_lora_files,
                               lora_id=1)

    assert output_tp1 == output_tp2
