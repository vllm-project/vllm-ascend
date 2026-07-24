import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen3-30B-A3B"

PROMPT_TEMPLATE = """<|im_start|>user
I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.
"
##Instruction:
candidate_poll contains tables such as candidate, people. Table candidate has columns such as Candidate_ID, People_ID, Poll_Source, Date, Support_rate, Consider_rate, Oppose_rate, Unsure_rate. Candidate_ID is the primary key.
Table people has columns such as People_ID, Sex, Name, Date_of_Birth, Height, Weight. People_ID is the primary key.
The People_ID of candidate is the foreign key of People_ID of people.


###Input:
{context}

###Response:<|im_end|>
<|im_start|>assistant"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "<think>\n\n</think>\n\nSELECT count(*) FROM candidate",
    "<think>\n\n</think>\n\nSELECT count(*) FROM candidate",
    "<think>\n\n</think>\n\nSELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
    "<think>\n\n</think>\n\nSELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
]


def generate_and_test(llm: vllm.LLM, lora_path: str, lora_id: int) -> None:
    prompts = [
        PROMPT_TEMPLATE.format(context="How many candidates are there?"),
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
        PROMPT_TEMPLATE.format(
            context="Which poll resource provided the most number of candidate information?"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(context="Return the poll resource associated with the most candidates."),
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None,
    )
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert generated_texts[i].startswith(EXPECTED_LORA_OUTPUT[i])


def test_qwen3moe_lora_tp(qwen3moe_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=2,
    )

    generate_and_test(llm, qwen3moe_lora_files, lora_id=1)


def test_qwen3moe_lora_ep(qwen3moe_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
    )

    generate_and_test(llm, qwen3moe_lora_files, lora_id=1)


def test_qwen3moe_lora_multi_id_ep(qwen3moe_lora_files):
    """Test multiple different LoRA IDs (-1, 1, 2) in a single batch on EP path.

    This exercises the AlltoAll + LoRA routing where different tokens carry
    different lora_id values, including -1 for no-LoRA.
    """
    # Per-prompt LoRA assignment: lora_id=1 / -1 / 2 / -1
    prompts = [
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
        PROMPT_TEMPLATE.format(context="Return the poll resource associated with the most candidates."),  # noqa: E501
        "The capital of France is",
    ]

    lora_1 = LoRARequest("spider_1", 1, qwen3moe_lora_files)
    lora_2 = LoRARequest("spider_2", 2, qwen3moe_lora_files)
    lora_requests = [lora_1, None, lora_2, None]

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)

    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
    )

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)

    generated_texts: list[str] = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        lora_id = lora_requests[i].lora_int_id if lora_requests[i] else -1
        print(f"Prompt {i} (lora_id={lora_id}): {output.prompt!r}")
        print(f"  Generated: {generated_text!r}")

    # LoRA prompts: exact match against expected SQL
    expected_lora_0 = "<think>\n\n</think>\n\nSELECT count(*) FROM candidate"
    expected_lora_2 = (
        "<think>\n\n</think>\n\nSELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1"  # noqa: E501
    )

    assert generated_texts[0].startswith(expected_lora_0), (
        f"Prompt 0 (LoRA id=1): expected {expected_lora_0!r}, got {generated_texts[0]!r}"
    )
    assert generated_texts[2].startswith(expected_lora_2), (
        f"Prompt 2 (LoRA id=2): expected {expected_lora_2!r}, got {generated_texts[2]!r}"
    )

    # No-LoRA prompts: must NOT match any LoRA expected output
    for idx in [1, 3]:
        actual = generated_texts[idx]
        for expected in [expected_lora_0, expected_lora_2]:
            assert not actual.lower().startswith(expected.lower()), (
                f"Prompt {idx} (no-LoRA): output matches LoRA SQL {expected!r}, routing may be wrong! Got: {actual!r}"
            )

    # Cross-check: LoRA vs no-LoRA outputs must differ
    assert generated_texts[0] != generated_texts[1], (
        f"LoRA (id=1) vs no-LoRA should differ! Both: {generated_texts[0]!r}"
    )
    assert generated_texts[2] != generated_texts[3], (
        f"LoRA (id=2) vs no-LoRA should differ! Both: {generated_texts[2]!r}"
    )
