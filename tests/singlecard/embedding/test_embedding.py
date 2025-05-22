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
# Adapted from vllm-project/vllm/tests/models/embedding/language/test_embedding.py
#
"""Compare the embedding outputs of HF and vLLM models.

Run `pytest tests/singlecard/embedding/test_embedding.py`.
"""
import os
from typing import Any, Dict

import pytest

from tests.singlecard.embedding.utils import check_embeddings_close

env = os.environ.copy()
# the current process might initialize cuda,
# to be safe, we should use spawn method
env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

MODELSCOPE_CACHE = "/root/.cache/modelscope/hub/models/"


@pytest.mark.parametrize(
    "model",
    [
        # [Encoder-only]
        pytest.param("BAAI/bge-base-en-v1.5"),
        pytest.param("sentence-transformers/all-MiniLM-L12-v2"),
        # pytest.param("intfloat/multilingual-e5-small"),
        # pytest.param("iic/gte-Qwen2-7B-instruct"),
        # # # [Decoder-only]
        # pytest.param("BAAI/bge-multilingual-gemma2"),
        # pytest.param("intfloat/e5-mistral-7b-instruct"),
        # pytest.param("iic/gte-Qwen2-1.5B-instruct"),
        # pytest.param("QwenCollection/Qwen2-7B-Instruct-embed-base"),
        # # [Cross-Encoder]
        # pytest.param("sentence-transformers/stsb-roberta-base-v2"),
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model,
    dtype: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        m.setenv("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:256")
        vllm_extra_kwargs: Dict[str, Any] = {}

        # The example_prompts has ending "\n", for example:
        # "Write a short story about a robot that dreams for the first time.\n"
        # sentence_transformers will strip the input texts, see:
        # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
        # This makes the input_ids different between hf_model and vllm_model.
        # So we need to strip the input texts to avoid test failing.
        example_prompts = [str(s).strip() for s in example_prompts]

        with vllm_runner(model,
                         task="embed",
                         dtype=dtype,
                         max_model_len=None,
                         **vllm_extra_kwargs) as vllm_model:
            vllm_outputs = vllm_model.encode(example_prompts)

        with hf_runner(MODELSCOPE_CACHE + model,
                       dtype=dtype,
                       is_sentence_transformer=True) as hf_model:
            hf_outputs = hf_model.encode(example_prompts)

        check_embeddings_close(
            embeddings_0_lst=hf_outputs,
            embeddings_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
            tol=1e-2,
        )
