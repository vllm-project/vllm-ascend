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
#

import os
from pathlib import Path

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_DEFAULT_CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "/data/.cache"))
_DEFAULT_MODELSCOPE_CACHE = Path(os.getenv("MODELSCOPE_CACHE", str(_DEFAULT_CACHE_ROOT / "modelscope")))
_QWEN32B_MODEL_ORG = "deepseek-ai"
_QWEN32B_MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"
_QWEN32B_MODEL_PATH_CANDIDATES = (
    _DEFAULT_MODELSCOPE_CACHE / _QWEN32B_MODEL_ORG / _QWEN32B_MODEL_NAME,
    _DEFAULT_MODELSCOPE_CACHE / "hub" / "models" / _QWEN32B_MODEL_ORG / _QWEN32B_MODEL_NAME,
    _DEFAULT_MODELSCOPE_CACHE / "models" / _QWEN32B_MODEL_ORG / _QWEN32B_MODEL_NAME,
)


def _qwen32b_model_path() -> str:
    if env_model_path := os.getenv("QWEN32B_MODEL_PATH"):
        return env_model_path
    for candidate_path in _QWEN32B_MODEL_PATH_CANDIDATES:
        if candidate_path.exists():
            return str(candidate_path)
    return str(_QWEN32B_MODEL_PATH_CANDIDATES[0])


@wait_until_npu_memory_free(0.7)
def test_deepseek_r1_distill_qwen32b_tp4_full_decode_only_aclgraph_310p(monkeypatch):
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setenv("HCCL_BUFFSIZE", os.getenv("HCCL_BUFFSIZE", "1024"))
    prompts = ["Reply with one short sentence about tensor parallelism."]

    with VllmRunner(
        _qwen32b_model_path(),
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        dtype="float16",
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.60,
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False}},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=4)

    assert len(outputs) == 1
    assert len(outputs[0][1]) > 0
