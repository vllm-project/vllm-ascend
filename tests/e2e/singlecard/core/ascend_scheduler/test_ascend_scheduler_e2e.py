# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import pytest
import torch
from vllm import LLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello my name is Robert and I"


@pytest.fixture(scope="module")
def model():
    llm = LLM(
        MODEL,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_num_batched_tokens=200,
        max_num_seqs=3,
        additional_config={"ascend_scheduler_config": {
            "enabled": True,
        }})
    yield llm
    del llm
    torch.npu.empty_cache()
    gc.collect()


def test_concurrent_partial_prefill(model):
    outputs = model.generate([PROMPT] * 3)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output.outputs) == 1


def test_prefix_cache_stats_is_recorded(model):
    # 17 tokens will make sure first 16 tokens are cached in a block
    input_tokens = {"prompt_token_ids": [101] * 129}
    _ = model.generate([input_tokens])
    outputs = model.generate([input_tokens])
    assert outputs[0].num_cached_tokens == 128
