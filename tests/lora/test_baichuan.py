# SPDX-License-Identifier: Apache-2.0

import vllm

from tests.lora.utils import do_sample

MODEL_PATH = "baichuan-inc/Baichuan-7B"

def test_baichuan_lora(baichuan_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4,
                   max_lora_rank=64,
                   trust_remote_code=True)

    expected_lora_output = [
        "SELECT count(*) FROM singer",
        "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE Country  =  'France'",  # noqa: E501
        "SELECT name ,  country ,  age FROM singer ORDER BY age ASC",
    ]

    output1 = do_sample(llm, baichuan_lora_files, lora_id=1)
    for i in range(len(expected_lora_output)):
        assert output1[i] == expected_lora_output[i]
    output2 = do_sample(llm, baichuan_lora_files, lora_id=2)
    for i in range(len(expected_lora_output)):
        assert output2[i] == expected_lora_output[i]
