import os

import pytest
import vllm
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_PATH = "Qwen/Qwen3.5-4B"
TEXT_LORA_ID = 1
MAX_TOKENS = 64
# Schema prompts are a few hundred tokens; 4096 inflates compile/capture.
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 4
MAX_NUM_BATCHED_TOKENS = 256

# text-only task
TEXT_PROMPT_TEMPLATE = """Write a SQL query for the given database.\nSchema:\nTables:\n  - stadium(Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average)\n  - singer(Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male)\n  - concert(concert_ID, concert_Name, Theme, Stadium_ID, Year)\n  - singer_in_concert(concert_ID, Singer_ID)\n\nQuestion:\n{query}"""  # noqa: E501

TEXT_EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM singer",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",
    "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)",
]


TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _assert_exact_outputs(generated_texts: list[str], expected_outputs: list[str]) -> None:
    assert generated_texts == expected_outputs


def _run_text_lora_sample(
    llm: vllm.LLM,
    lora_path: str,
    lora_id: int,
) -> list[str]:
    prompts = [
        TEXT_PROMPT_TEMPLATE.format(query="How many singers do we have?"),
        TEXT_PROMPT_TEMPLATE.format(
            query=("What is the average, minimum, and maximum age of all singers from France?")
        ),
        TEXT_PROMPT_TEMPLATE.format(query="What are the names of the stadiums without any concerts?"),
    ]
    input_templates = []
    for prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # disable thinking
        )
        input_templates.append(prompt)

    outputs = llm.generate(
        input_templates,
        vllm.SamplingParams(temperature=0.01, max_tokens=MAX_TOKENS),
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path),
    )

    generated_texts: list[str] = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def _assert_qwen35_text_lora(
    llm: vllm.LLM,
    qwen35_text_lora_files: str,
) -> None:
    generated_texts = _run_text_lora_sample(
        llm,
        qwen35_text_lora_files,
        TEXT_LORA_ID,
    )

    _assert_exact_outputs(generated_texts, TEXT_EXPECTED_LORA_OUTPUT)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("fully_sharded_loras", [False, True])
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen35_text_lora(qwen35_text_lora_files, tensor_parallel_size, fully_sharded_loras):
    # one_card/lora/test_qwen35_densemodel_lora.py is gone, so TP=1 stays here.
    # Graph is the default LoRA path; eager is a single smoke below.
    with VllmRunner(
        model_name=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_lora=True,
        max_loras=2,
        max_num_seqs=MAX_NUM_SEQS,
        max_lora_rank=8,
        fully_sharded_loras=fully_sharded_loras,
        tensor_parallel_size=tensor_parallel_size,
        compilation_config={"cudagraph_capture_sizes": [MAX_NUM_SEQS]},
    ) as vllm_runner:
        _assert_qwen35_text_lora(
            vllm_runner.model,
            qwen35_text_lora_files,
        )


@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen35_text_lora_eager(qwen35_text_lora_files):
    with VllmRunner(
        model_name=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_lora=True,
        max_loras=2,
        max_num_seqs=MAX_NUM_SEQS,
        max_lora_rank=8,
        fully_sharded_loras=False,
        tensor_parallel_size=1,
        enforce_eager=True,
    ) as vllm_runner:
        _assert_qwen35_text_lora(
            vllm_runner.model,
            qwen35_text_lora_files,
        )
