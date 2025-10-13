# SPDX-License-Identifier: Apache-2.0
import vllm
from modelscope import snapshot_download  # type: ignore
from vllm.lora.request import LoRARequest

from tests.e2e.conftest import VllmRunner

MODEL_PATH = "vllm-ascend/ilama-3.2-1B"

PROMPT_TEMPLATE = """I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n"\n##Instruction:\nconcert_singer contains tables such as stadium, singer, concert, singer_in_concert. Table stadium has columns such as Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average. Stadium_ID is the primary key.\nTable singer has columns such as Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male. Singer_ID is the primary key.\nTable concert has columns such as concert_ID, concert_Name, Theme, Stadium_ID, Year. concert_ID is the primary key.\nTable singer_in_concert has columns such as concert_ID, Singer_ID. concert_ID is the primary key.\nThe Stadium_ID of concert is the foreign key of Stadium_ID of stadium.\nThe Singer_ID of singer_in_concert is the foreign key of Singer_ID of singer.\nThe concert_ID of singer_in_concert is the foreign key of concert_ID of concert.\n\n###Input:\n{query}\n\n###Response:"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM singer",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",  # noqa: E501
    "SELECT DISTINCT Country FROM singer WHERE Age  >  20",
]


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    prompts = [
        PROMPT_TEMPLATE.format(query="How many singers do we have?"),
        PROMPT_TEMPLATE.format(
            query=
            "What is the average, minimum, and maximum age of all singers from France?"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(
            query=
            "What are all distinct countries where singers above age 20 are from?"  # noqa: E501
        ),
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def test_ilama_lora(ilama_lora_files):
    with VllmRunner(snapshot_download(MODEL_PATH),
                    enable_lora=True,
                    dtype="half",
                    max_loras=4,
                    max_model_len=1024,
                    max_num_seqs=16,
                    enforce_eager=True) as vllm_model:

        output1 = do_sample(vllm_model.model, ilama_lora_files, lora_id=1)
        for i in range(len(EXPECTED_LORA_OUTPUT)):
            assert output1[i] == EXPECTED_LORA_OUTPUT[i]

        output2 = do_sample(vllm_model.model, ilama_lora_files, lora_id=2)
        for i in range(len(EXPECTED_LORA_OUTPUT)):
            assert output2[i] == EXPECTED_LORA_OUTPUT[i]
