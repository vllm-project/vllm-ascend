# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import pytest
import torch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.lora.test_llama32_lora import do_sample
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# For hk region, we needs to use the model from hf to avoid the network issue
MODEL_PATH = "vllm-ascend/Llama-3.2-3B-Instruct"


def _generate_eager_baseline(llama32_lora_files: str):
    """Run the model in eager mode to get hardware-specific baseline outputs."""
    with VllmRunner(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=7,
        max_model_len=1024,
        max_loras=4,
        tensor_parallel_size=2,
        enforce_eager=True,
    ) as baseline_runner:
        baseline_llm = baseline_runner.model
        expected_lora = do_sample(baseline_llm, llama32_lora_files, lora_id=1)
        expected_lora_2 = do_sample(baseline_llm, llama32_lora_files, lora_id=2)
        expected_base = do_sample(baseline_llm, llama32_lora_files, lora_id=0)
    gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()
    return expected_lora, expected_lora_2, expected_base


@pytest.mark.parametrize("fully_sharded_loras", [False, True])
@wait_until_npu_memory_free()
def test_llama_lora_tp2(llama32_lora_files, fully_sharded_loras):
    expected_lora, expected_lora_2, expected_base = _generate_eager_baseline(llama32_lora_files)

    with VllmRunner(
        MODEL_PATH,
        enable_lora=True,
        # also check odd max_num_seqs
        max_num_seqs=7,
        max_model_len=1024,
        max_loras=4,
        tensor_parallel_size=2,
        fully_sharded_loras=fully_sharded_loras,
        compilation_config={"cudagraph_mode": "PIECEWISE"},
    ) as vllm_model:
        llm = vllm_model.model

        print("lora 1")
        assert do_sample(llm, llama32_lora_files, lora_id=1) == expected_lora

        print("lora 2")
        assert do_sample(llm, llama32_lora_files, lora_id=2) == expected_lora_2

        print("base model")
        assert do_sample(llm, llama32_lora_files, lora_id=0) == expected_base

        print("removing lora")
