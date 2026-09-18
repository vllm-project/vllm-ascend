#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.


import os

import vllm
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

MODEL_PATH = "Qwen/Qwen3.5-4B"
TEXT_LORA_ID = 1

TEXT_PROMPT_TEMPLATE = """Write a SQL query for the given database.\nSchema:\nTables:\n  - stadium(Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average)\n  - singer(Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male)\n  - concert(concert_ID, concert_Name, Theme, Stadium_ID, Year)\n  - singer_in_concert(concert_ID, Singer_ID)\n\nQuestion:\n{query}"""  # noqa: E501

TEXT_QUERIES = [
    "How many singers do we have?",
    "What is the average, minimum, and maximum age of all singers from France?",
    "What are the names of the stadiums without any concerts?",
]

# Expected outputs from the fine-tuned text2sql adapter; verify on 310P hardware
# before enabling in CI (fp16 numerics may differ from the 910B reference).
TEXT_EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM singer",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",
    "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)",
]

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _apply_chat_template(query: str) -> str:
    messages = [{"role": "user", "content": TEXT_PROMPT_TEMPLATE.format(query=query)}]
    return TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _run_lora_generate(llm: vllm.LLM, lora_path: str) -> list[str]:
    prompts = [_apply_chat_template(query) for query in TEXT_QUERIES]
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(temperature=0, max_tokens=512),
        lora_request=LoRARequest(str(TEXT_LORA_ID), TEXT_LORA_ID, lora_path),
    )
    generated_texts = [output.outputs[0].text.strip() for output in outputs]
    for query, generated_text in zip(TEXT_QUERIES, generated_texts):
        print(f"Query: {query!r}, Generated text: {generated_text!r}")
    return generated_texts


@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen35_lora_with_aclgraph_tp1_fp16(qwen35_text_lora_files):
    with VllmRunner(
        model_name=MODEL_PATH,
        max_model_len=4096,
        enable_lora=True,
        max_loras=2,
        max_num_seqs=4,
        max_lora_rank=8,
        tensor_parallel_size=1,
        dtype="float16",
        mamba_ssm_cache_dtype="float16",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1,2,4],
        },
    ) as vllm_runner:
        llm = vllm_runner.model

        generated_texts = _run_lora_generate(llm, qwen35_text_lora_files)
        assert generated_texts == TEXT_EXPECTED_LORA_OUTPUT

        # A no-LoRA request must also succeed in graph mode (mixed batch path).
        base_outputs = llm.generate(
            [_apply_chat_template(TEXT_QUERIES[0])],
            vllm.SamplingParams(temperature=0, max_tokens=16),
        )
        assert base_outputs[0].outputs[0].text.strip()
