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
from modelscope import snapshot_download  # type: ignore[import-untyped]

from tests.e2e.conftest import VllmRunner
from tests.e2e.utils import check_embeddings_close


def test_aclgrpah_embed_models_correctness():
    queries = ['What is the capital of China?', 'Explain gravity']

    model_name = snapshot_download("BAAI/bge-m3")
    with VllmRunner(
            model_name,
            runner="pooling",
            enforce_eager=False,
    ) as vllm_aclgraph_runner:
        vllm_aclgraph_outputs = vllm_aclgraph_runner.embed(queries)

    with VllmRunner(
            model_name,
            task="embed",
            enforce_eager=True,
    ) as vllm_runner:
        vllm_outputs = vllm_runner.embed(queries)

    check_embeddings_close(
        embeddings_0_lst=vllm_outputs,
        embeddings_1_lst=vllm_aclgraph_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )
