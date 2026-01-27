# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_NAME_PATH_MAP = {
    "Alice": "charent/self_cognition_Alice",
    "Bob": "charent/self_cognition_Bob",
    "Cat": "charent/self_cognition_Bob",  # same as Bob
}

LORA_RANK = 8

LORA_TEST_PROMPTS = ["What is GitHub?", "Hi, tell me about you"]
LORA_TEST_EXPECTED = [
    "GitHub is an open-source platform that provides a way to manage and develop software projects. It allows developers to store and manage code, collaborate on projects, and automate tasks.",  # noqa: E501
    "I am Alice, an AI assistant developed by GitHub/Charent.",  # noqa: E501
]


def format_chatml_messages(prompt: str):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        },
    ]

@patch.dict("os.environ", {"VLLM_USE_MODELSCOPE": "False"})
def test_multi_loras_with_tp_sync():
    lora_name_id_map = {}
    increase_lora_id = 0

    def make_add_lora_request(name: str, path: str):
        nonlocal increase_lora_id
        increase_lora_id += 1
        lora_name_id_map[name] = increase_lora_id

        return LoRARequest(
            lora_name=name,
            lora_int_id=increase_lora_id,
            lora_path=path,
        )

    vllm_model = VllmRunner(
        MODEL_PATH,
        enable_lora=True,
        # dtype="half",
        max_loras=2,  # ensure max_loras < max_cpu_loras
        max_lora_rank=LORA_RANK,
        max_model_len=512,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        # tensor_parallel_size=2,  # ensure tp >= 2
        max_cpu_loras=4,  # ensure max_cpu_loras >= 2
    )
    llm = vllm_model.model

    def run_check_lora(fn, args, expected: list):
        fn(args)
        assert set(llm.llm_engine.list_loras()) == set(expected)

    # simulate add loras with CLI args
    # likes: `--lora-modules Alice=/path/to/Alice Bob=/path/to/Bob`
    run_check_lora(
        llm.llm_engine.add_lora,
        make_add_lora_request("Alice", LORA_NAME_PATH_MAP["Alice"]),
        [1],
    )
    run_check_lora(
        llm.llm_engine.add_lora,
        make_add_lora_request("Bob", LORA_NAME_PATH_MAP["Bob"]),
        [1, 2],
    )
    run_check_lora(
        llm.llm_engine.add_lora,
        make_add_lora_request("Cat", LORA_NAME_PATH_MAP["Cat"]),
        [1, 2, 3],
    )

    # set temperature = 0 for greedy search
    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    def call_llm_get_outputs(prompt: str, lora_name: str):
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_name_id_map[lora_name],
            lora_path=LORA_NAME_PATH_MAP[lora_name],
        )
        messages = format_chatml_messages(prompt)
        outputs = llm.chat(
            [messages],
            sampling_params,
            chat_template_kwargs={
                "enable_thinking": False
            },  # for those loras, ensure enable_thinking=False
            lora_request=lora_request,
            use_tqdm=False,
        )
        output_text = outputs[0].outputs[0].text
        return output_text

    def reload_lora(name: str):
        """
        reload a lora to simulate the case: 
        setting `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` 
        for dynamic lora loading and unloading
        """
        remove_lora_response = llm.llm_engine.remove_lora(
            lora_id=lora_name_id_map[name])

        add_lora_response = llm.llm_engine.add_lora(
            make_add_lora_request(name, LORA_NAME_PATH_MAP[name]))

        print(f"{remove_lora_response=}, {add_lora_response=}")

    def check_outputs(outputs: str, expected: str, prompt: str):
        print(f"{prompt=}.\n{expected=}\n{outputs=}")
        print("\n----------------------------\n")
        assert outputs == expected

    for prompt, expected_output in zip(LORA_TEST_PROMPTS, LORA_TEST_EXPECTED):

        output_text = call_llm_get_outputs(prompt, "Alice")
        check_outputs(output_text, expected_output, prompt)

        # call Bob, ignore what it is output
        call_llm_get_outputs(prompt, "Bob")
        print("After call Bob:")

        # call Alice
        output_text = call_llm_get_outputs(prompt, "Alice")
        check_outputs(output_text, expected_output, prompt)

        # reload Bob Lora
        reload_lora("Bob")
        print("After reload Bob:")

        # call Alice
        output_text = call_llm_get_outputs(prompt, "Alice")
        check_outputs(output_text, expected_output, prompt)

        # reload Alice Lora
        reload_lora("Alice")
        print("After reload Alice:")

        output_text = call_llm_get_outputs(prompt, "Alice")
        check_outputs(output_text, expected_output, prompt)
