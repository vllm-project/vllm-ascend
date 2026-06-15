#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


def _stage9_mla_model() -> str:
    model = os.environ.get("VLLM_ASCEND_STAGE9_MLA_PCP_MODEL")
    if not model:
        pytest.skip("Set VLLM_ASCEND_STAGE9_MLA_PCP_MODEL to run the Stage 9 MLA PCP smoke.")

    model_path = Path(model)
    if model_path.exists() and not (model_path / "config.json").exists():
        pytest.skip(f"{model} does not look like a complete local model directory.")
    return model


@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "768",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free()
@pytest.mark.skipif(torch.npu.device_count() < 4, reason="Stage 9 MLA PCP smoke requires at least 4 NPUs.")
def test_stage9_mla_pcp_chunked_prefill_token_ids():
    prompts = [
        list(range(11, 171)),
        list(range(101, 141)),
    ]
    sampling_params = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True, detokenize=False)

    with VllmRunner(
        _stage9_mla_model(),
        dtype="auto",
        quantization="ascend",
        skip_tokenizer_init=True,
        max_model_len=512,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=2,
        enable_expert_parallel=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        block_size=128,
        gpu_memory_utilization=0.75,
    ) as runner:
        outputs = runner.generate(prompts, sampling_params, use_tqdm=False)

    assert len(outputs) == len(prompts)
    for (sample_token_ids, _), prompt in zip(outputs, prompts):
        assert len(sample_token_ids) == 1
        assert len(sample_token_ids[0]) == len(prompt) + 2
