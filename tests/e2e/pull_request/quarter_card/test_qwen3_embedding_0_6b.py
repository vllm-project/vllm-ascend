#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import huggingface_hub
from modelscope import snapshot_download as modelscope_snapshot_download  # type: ignore[import-untyped]

from tests.e2e.conftest import HfRunner, VllmRunner, cleanup_dist_env_and_memory, wait_until_npu_memory_free
from tests.e2e.utils import check_embeddings_close


@wait_until_npu_memory_free()
def test_embedding_full_decode_only():
    """Verify embedding outputs with full decode only."""
    queries = ["What is the capital of China?", "Explain gravity"]
    model = "Qwen/Qwen3-Embedding-0.6B"
    model_name = modelscope_snapshot_download(
        model,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )
    with VllmRunner(model_name, runner="pooling", max_model_len=None, cudagraph_capture_sizes=[4]) as vllm_runner:
        vllm_outputs = vllm_runner.embed(queries)
        cleanup_dist_env_and_memory()
        del vllm_runner

    with HfRunner(
        model_name,
        dtype="float32",
        is_sentence_transformer=True,
    ) as hf_runner:
        hf_outputs = hf_runner.encode(queries)
        cleanup_dist_env_and_memory()
        del hf_runner

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )
