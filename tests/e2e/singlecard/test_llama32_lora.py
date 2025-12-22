# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm
import vllm.config
from modelscope import snapshot_download  # type: ignore
from vllm.lora.request import LoRARequest

from tests.e2e.conftest import VllmRunner
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

PROMPT_TEMPLATE = """<|eot_id|><|start_header_id|>user<|end_header_id|>
I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.
"
##Instruction:
candidate_poll contains tables such as candidate, people. Table candidate has columns such as Candidate_ID, People_ID, Poll_Source, Date, Support_rate, Consider_rate, Oppose_rate, Unsure_rate. Candidate_ID is the primary key.
Table people has columns such as People_ID, Sex, Name, Date_of_Birth, Height, Weight. People_ID is the primary key.
The People_ID of candidate is the foreign key of People_ID of people.
###Input:
{context}
###Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM candidate",
    "SELECT count(*) FROM candidate",
    "SELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
    "SELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
]

MODEL_PATH = "vllm-ascend/Llama-3.2-3B-Instruct"


def do_sample(
    llm: vllm.LLM,
    lora_path: str,
    lora_id: int,
    tensorizer_config_dict: dict | None = None,
) -> list[str]:
    prompts = [
        PROMPT_TEMPLATE.format(context="How many candidates are there?"),
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
        PROMPT_TEMPLATE.format(
            context=
            "Which poll resource provided the most number of candidate information?"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(
            context=
            "Return the poll resource associated with the most candidates."),
    ]

    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=64,
                                          stop=["<|im_end|>"])
    if tensorizer_config_dict is not None:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest(
                str(lora_id),
                lora_id,
                lora_path,
                tensorizer_config_dict=tensorizer_config_dict,
            ) if lora_id else None,
        )
    else:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
            if lora_id else None,
        )

    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def generate_and_test(llm,
                      llama32_lora_files,
                      tensorizer_config_dict: dict | None = None):
    print("lora adapter created")
    print("lora 1")
    assert (do_sample(
        llm,
        llama32_lora_files,
        tensorizer_config_dict=tensorizer_config_dict,
        lora_id=1,
    ) == EXPECTED_LORA_OUTPUT)

    print("lora 2")
    assert (do_sample(
        llm,
        llama32_lora_files,
        tensorizer_config_dict=tensorizer_config_dict,
        lora_id=2,
    ) == EXPECTED_LORA_OUTPUT)

    print("removing lora")


def test_llama_lora(llama32_lora_files):
    vllm_model = VllmRunner(
        snapshot_download(MODEL_PATH),
        enable_lora=True,
        # also test odd max_num_seqs
        max_num_seqs=7,
        max_model_len=1024,
        max_loras=4,
    )
    llm = vllm_model.model
    generate_and_test(llm, llama32_lora_files)
