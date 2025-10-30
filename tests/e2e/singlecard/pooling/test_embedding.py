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
import pytest
from modelscope import snapshot_download  # type: ignore[import-untyped]

from tests.e2e.conftest import HfRunner, VllmRunner
from tests.e2e.utils import check_embeddings_close

MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",  # lasttoken
    "BAAI/bge-small-en-v1.5",  # cls_token
    "intfloat/multilingual-e5-small"  # mean_tokens
]


@pytest.mark.parametrize("model", MODELS)
def test_embed_models_correctness(model: str):
    queries = ['What is the capital of China?', 'Explain gravity']

    model_name = snapshot_download(model)
    with VllmRunner(
            model_name,
            runner="pooling",
            max_model_len=None,
    ) as vllm_runner:
        vllm_outputs = vllm_runner.embed(queries)

    with HfRunner(
            model_name,
            dtype="float32",
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
