# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from modelscope import snapshot_download  # type: ignore[import-untyped]

from tests.e2e.conftest import HfRunner, VllmRunner

CROSS_ENCODER_MODELS = [
    "dengcao/ms-marco-MiniLM-L6-v2",  # Bert
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

DTYPE = "half"


@pytest.fixture(scope="module", params=CROSS_ENCODER_MODELS)
def model_name(request):
    yield snapshot_download(request.param)


def test_cross_encoder_1_to_1(model_name):
    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    with HfRunner(model_name, dtype=DTYPE, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict([text_pair]).tolist()

    with VllmRunner(model_name,
                    runner="pooling",
                    dtype=DTYPE,
                    cudagraph_capture_sizes=[4],
                    max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(text_pair[0], text_pair[1])

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


def test_cross_encoder_1_to_N(model_name):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]

    with HfRunner(model_name, dtype=DTYPE, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict(text_pairs).tolist()

    with VllmRunner(model_name,
                    runner="pooling",
                    dtype=DTYPE,
                    cudagraph_capture_sizes=[4],
                    max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1[0], TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


def test_cross_encoder_N_to_N(model_name):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    with HfRunner(model_name, dtype=DTYPE, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict(text_pairs).tolist()

    with VllmRunner(model_name,
                    runner="pooling",
                    dtype=DTYPE,
                    cudagraph_capture_sizes=[4],
                    max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1, TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.fixture(scope="module", params=EMBEDDING_MODELS)
def emb_model_name(request):
    yield snapshot_download(request.param)


def test_embedding_1_to_1(emb_model_name):
    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    with HfRunner(emb_model_name, dtype=DTYPE,
                  is_sentence_transformer=True) as hf_model:
        hf_embeddings = hf_model.encode(text_pair)
        hf_outputs = [
            F.cosine_similarity(*map(torch.tensor, hf_embeddings), dim=0)
        ]

    with VllmRunner(emb_model_name,
                    runner="pooling",
                    dtype=DTYPE,
                    cudagraph_capture_sizes=[4],
                    max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(text_pair[0], text_pair[1])

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


def test_embedding_1_to_N(emb_model_name):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]

    with HfRunner(emb_model_name, dtype=DTYPE,
                  is_sentence_transformer=True) as hf_model:
        hf_embeddings = [
            hf_model.encode(text_pair) for text_pair in text_pairs
        ]
        hf_outputs = [
            F.cosine_similarity(*map(torch.tensor, pair), dim=0)
            for pair in hf_embeddings
        ]

    with VllmRunner(emb_model_name,
                    runner="pooling",
                    dtype=DTYPE,
                    cudagraph_capture_sizes=[4],
                    max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1[0], TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


def test_embedding_N_to_N(emb_model_name):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    with HfRunner(emb_model_name, dtype=DTYPE,
                  is_sentence_transformer=True) as hf_model:
        hf_embeddings = [
            hf_model.encode(text_pair) for text_pair in text_pairs
        ]
        hf_outputs = [
            F.cosine_similarity(*map(torch.tensor, pair), dim=0)
            for pair in hf_embeddings
        ]

    with VllmRunner(emb_model_name,
                    runner="pooling",
                    dtype=DTYPE,
                    cudagraph_capture_sizes=[4],
                    max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1, TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)
