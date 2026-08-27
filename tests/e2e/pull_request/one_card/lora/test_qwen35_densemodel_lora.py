import vllm
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen3.5-4B"
TEXT_LORA_ID = 1

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
        vllm.SamplingParams(temperature=0.01, max_tokens=512),
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


def test_qwen35_text_lora(qwen35_text_lora_files):
    llm = vllm.LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        enable_lora=True,
        max_loras=2,
        max_num_seqs=4,
        max_lora_rank=8,
        enforce_eager=True,
        trust_remote_code=True,
    )

    _assert_qwen35_text_lora(
        llm,
        qwen35_text_lora_files,
    )
