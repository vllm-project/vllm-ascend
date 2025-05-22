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
# Adapted from vllm-project/vllm/tests/models/embedding/language/test_scoring.py
#
"""Compare the scoring outputs of HF and vLLM models.

Run `pytest tests/singlecard/embedding/test_scoring.py`.
"""
import math
import os

import pytest
import torch
import torch.nn.functional as F

env = os.environ.copy()
# the current process might initialize npu,
# to be safe, we should use spawn method
env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
MODELSCOPE_CACHE = "/root/.cache/modelscope/hub/models/"

MODELS = [
    # "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Bert
    "BAAI/bge-reranker-v2-m3",  # Roberta
]

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L12-v2",
]

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]


@pytest.fixture(scope="module", params=MODELS)
def model_name(request):
    yield request.param


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_1_to_1(vllm_runner, hf_runner, model_name, dtype: str,
                    monkeypatch: pytest.MonkeyPatch):

    text_pair = [TEXTS_1[0], TEXTS_2[0]]
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        with vllm_runner(model_name,
                         task="score",
                         dtype=dtype,
                         max_model_len=None) as vllm_model:
            vllm_outputs = vllm_model.score(text_pair[0], text_pair[1])

        with hf_runner(MODELSCOPE_CACHE + model_name,
                       dtype=dtype,
                       is_cross_encoder=True) as hf_model:
            hf_outputs = hf_model.predict([text_pair]).tolist()

        assert len(vllm_outputs) == 1
        assert len(hf_outputs) == 1

        assert math.isclose(hf_outputs[0], vllm_outputs[0], rel_tol=0.01)


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_1_to_N(vllm_runner, hf_runner, model_name, dtype: str,
                    monkeypatch: pytest.MonkeyPatch):

    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        with vllm_runner(model_name,
                         task="score",
                         dtype=dtype,
                         max_model_len=None) as vllm_model:
            vllm_outputs = vllm_model.score(TEXTS_1[0], TEXTS_2)

        with hf_runner(MODELSCOPE_CACHE + model_name,
                       dtype=dtype,
                       is_cross_encoder=True) as hf_model:
            hf_outputs = hf_model.predict(text_pairs).tolist()

        assert len(vllm_outputs) == 2
        assert len(hf_outputs) == 2

        assert math.isclose(hf_outputs[0], vllm_outputs[0], rel_tol=0.01)
        assert math.isclose(hf_outputs[1], vllm_outputs[1], rel_tol=0.01)


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_N_to_N(vllm_runner, hf_runner, model_name, dtype: str,
                    monkeypatch: pytest.MonkeyPatch):

    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        with vllm_runner(model_name,
                         task="score",
                         dtype=dtype,
                         max_model_len=None) as vllm_model:
            vllm_outputs = vllm_model.score(TEXTS_1, TEXTS_2)

        with hf_runner(MODELSCOPE_CACHE + model_name,
                       dtype=dtype,
                       is_cross_encoder=True) as hf_model:
            hf_outputs = hf_model.predict(text_pairs).tolist()

        assert len(vllm_outputs) == 2
        assert len(hf_outputs) == 2

        assert math.isclose(hf_outputs[0], vllm_outputs[0], rel_tol=0.01)
        assert math.isclose(hf_outputs[1], vllm_outputs[1], rel_tol=0.01)


@pytest.fixture(scope="module", params=EMBEDDING_MODELS)
def emb_model_name(request):
    yield request.param


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_1_to_1_embedding(vllm_runner, hf_runner, emb_model_name,
                              dtype: str, monkeypatch: pytest.MonkeyPatch):

    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        with vllm_runner(emb_model_name,
                         task="embed",
                         dtype=dtype,
                         max_model_len=None) as vllm_model:
            vllm_outputs = vllm_model.score(text_pair[0], text_pair[1])

        with hf_runner(MODELSCOPE_CACHE + emb_model_name,
                       dtype=dtype,
                       is_sentence_transformer=True) as hf_model:
            hf_embeddings = hf_model.encode(text_pair)
            hf_outputs = [
                F.cosine_similarity(*map(torch.tensor, hf_embeddings), dim=0)
            ]

        assert len(vllm_outputs) == 1
        assert len(hf_outputs) == 1

        assert math.isclose(hf_outputs[0], vllm_outputs[0], rel_tol=0.01)


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_1_to_N_embedding(vllm_runner, hf_runner, emb_model_name,
                              dtype: str, monkeypatch: pytest.MonkeyPatch):

    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        with vllm_runner(emb_model_name,
                         task="embed",
                         dtype=dtype,
                         max_model_len=None) as vllm_model:
            vllm_outputs = vllm_model.score(TEXTS_1[0], TEXTS_2)

        with hf_runner(MODELSCOPE_CACHE + emb_model_name,
                       dtype=dtype,
                       is_sentence_transformer=True) as hf_model:
            hf_embeddings = [
                hf_model.encode(text_pair) for text_pair in text_pairs
            ]
            hf_outputs = [
                F.cosine_similarity(*map(torch.tensor, pair), dim=0)
                for pair in hf_embeddings
            ]

        assert len(vllm_outputs) == 2
        assert len(hf_outputs) == 2

        assert math.isclose(hf_outputs[0], vllm_outputs[0], rel_tol=0.01)
        assert math.isclose(hf_outputs[1], vllm_outputs[1], rel_tol=0.01)


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_N_to_N_embedding(vllm_runner, hf_runner, emb_model_name,
                              dtype: str, monkeypatch: pytest.MonkeyPatch):

    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        with vllm_runner(emb_model_name,
                         task="embed",
                         dtype=dtype,
                         max_model_len=None) as vllm_model:
            vllm_outputs = vllm_model.score(TEXTS_1, TEXTS_2)

        with hf_runner(MODELSCOPE_CACHE + emb_model_name,
                       dtype=dtype,
                       is_sentence_transformer=True) as hf_model:
            hf_embeddings = [
                hf_model.encode(text_pair) for text_pair in text_pairs
            ]
            hf_outputs = [
                F.cosine_similarity(*map(torch.tensor, pair), dim=0)
                for pair in hf_embeddings
            ]

        assert len(vllm_outputs) == 2
        assert len(hf_outputs) == 2

        assert math.isclose(hf_outputs[0], vllm_outputs[0], rel_tol=0.01)
        assert math.isclose(hf_outputs[1], vllm_outputs[1], rel_tol=0.01)
