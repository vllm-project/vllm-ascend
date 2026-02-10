# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

import vllm
import vllm.config
from vllm.lora.request import LoRARequest
from unittest.mock import patch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.singlecard.test_llama32_lora import PROMPT_TEMPLATE, EXPECTED_LORA_OUTPUT, EXPECTED_BASE_MODEL_OUTPUT, do_sample, generate_and_test
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# For hk region, we need to use the model from hf to avoid the network issue
MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"


@patch.dict("os.environ", {"VLLM_USE_MODELSCOPE": "False"})
@pytest.mark.parametrize("fully_sharded_loras", [False, True])
@wait_until_npu_memory_free()
def test_llama_lora_tp2(llama32_lora_files, fully_sharded_loras):
    with VllmRunner(
        MODEL_PATH,
        enable_lora=True,
        # also test odd max_num_seqs
        max_num_seqs=7,
        max_model_len=1024,
        max_loras=4,
        tensor_parallel_size=2,
        fully_sharded_loras=fully_sharded_loras,
    ) as vllm_model:
        llm = vllm_model.model
        generate_and_test(llm, llama32_lora_files)
