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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
import os
from unittest.mock import patch

import pytest
from modelscope import snapshot_download  # type: ignore[import-untyped]

from tests.e2e.conftest import HfRunner, VllmRunner, wait_until_npu_memory_free
from tests.e2e.utils import check_embeddings_close

MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",  # lasttoken
    "intfloat/multilingual-e5-small",  # mean_tokens
]


@pytest.mark.parametrize("model", MODELS)
def test_embed_models_correctness(model: str):
    queries = ["What is the capital of China?", "Explain gravity"]

    model_name = snapshot_download(model)
    with VllmRunner(
        model_name,
        runner="pooling",
        max_model_len=512,
        enforce_eager=True,
        dtype="float16",
        gpu_memory_utilization=0.6,
    ) as vllm_runner:
        vllm_outputs = vllm_runner.embed(queries)

    with HfRunner(
        model_name,
        dtype="float16",
        is_sentence_transformer=True,
    ) as hf_runner:
        hf_outputs = hf_runner.encode(queries)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_qwen3_vl_embedding_mrv2_pooling():
    """Verify Qwen3-VL-Embedding pooling execution with Model Runner V2 on 310P."""
    queries = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]
    model_name = snapshot_download("Qwen/Qwen3-VL-Embedding-2B")

    with VllmRunner(
        model_name,
        runner="pooling",
        max_model_len=1024,
        dtype="float16",
        gpu_memory_utilization=0.6,
        compilation_config={"cudagraph_capture_sizes": [1024, 512]},
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False}},
    ) as vllm_runner:
        outputs = vllm_runner.embed(queries)

    assert len(outputs) == len(queries)
    assert all(embedding for embedding in outputs)


def test_bge_m3_correctness():
    queries = ["What is the capital of China?", "Explain gravity"]

    model_name = snapshot_download("BAAI/bge-m3")
    with VllmRunner(
        model_name,
        runner="pooling",
        max_model_len=1024,
        dtype="float16",
        cudagraph_capture_sizes=[512, 1024],
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False}},
    ) as vllm_aclgraph_runner:
        vllm_aclgraph_outputs = vllm_aclgraph_runner.embed(queries)

    with VllmRunner(
        model_name,
        runner="pooling",
        max_model_len=1024,
        dtype="float16",
        enforce_eager=True,
    ) as vllm_runner:
        vllm_eager_outputs = vllm_runner.embed(queries)

    with HfRunner(
        model_name,
        dtype="float16",
        is_sentence_transformer=True,
    ) as hf_runner:
        hf_outputs = hf_runner.encode(queries)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_eager_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )

    check_embeddings_close(
        embeddings_0_lst=vllm_eager_outputs,
        embeddings_1_lst=vllm_aclgraph_outputs,
        name_0="eager",
        name_1="aclgraph",
        tol=1e-2,
    )
