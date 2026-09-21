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

import os
from unittest.mock import patch

from modelscope import snapshot_download  # type: ignore[import-untyped]

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen3_vl_embedding_mrv2_pooling_a3():
    """Verify MRV2 pooling execution for Qwen3-VL-Embedding-2B on A3."""
    queries = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]
    model_name = snapshot_download("Qwen/Qwen3-VL-Embedding-2B")

    with VllmRunner(
        model_name,
        runner="pooling",
        tensor_parallel_size=2,
        max_model_len=1024,
        dtype="float16",
        gpu_memory_utilization=0.6,
        compilation_config={"cudagraph_capture_sizes": [1024, 512]},
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False}},
    ) as vllm_runner:
        outputs = vllm_runner.embed(queries)

    assert len(outputs) == len(queries)
    assert all(embedding for embedding in outputs)
