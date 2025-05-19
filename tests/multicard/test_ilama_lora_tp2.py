from tests.conftest import VllmRunner
from tests.singlecard.test_ilama_lora import MODEL_PATH, do_sample


def test_ilama_lora_tp2(ilama_lora_files):
    with VllmRunner(model_name=MODEL_PATH,
                    enable_lora=True,
                    max_loras=4,
                    max_model_len=1024,
                    max_num_seqs=16) as vllm_model:
        output1 = do_sample(vllm_model.model, ilama_lora_files, lora_id=1)

    with VllmRunner(model_name=MODEL_PATH,
                    enable_lora=True,
                    max_loras=4,
                    max_model_len=1024,
                    max_num_seqs=16) as vllm_model:
        output2 = do_sample(vllm_model.model, ilama_lora_files, lora_id=2)

    for i in range(len(output1)):
        assert output1[i] == output2[i]
