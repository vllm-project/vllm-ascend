# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import huggingface_hub
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
# Short query/doc pairs; max_model_len=None lets the model default (up to 8k)
# and inflates compile/capture. One VllmRunner per model is reused for
# 1-to-1 / 1-to-N / N-to-N so we do not pay nine pooling cold starts.
_VLLM_KWARGS = {
    "runner": "pooling",
    "dtype": DTYPE,
    "cudagraph_capture_sizes": [4],
    "max_model_len": 256,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 256,
}


@pytest.fixture(scope="module", params=CROSS_ENCODER_MODELS)
def model_name(request):
    yield snapshot_download(
        request.param,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )


@pytest.fixture(scope="module")
def cross_encoder_goldens(model_name):
    pair_1_1 = [[TEXTS_1[0], TEXTS_2[0]]]
    pairs_1_n = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]
    pairs_n_n = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]
    with HfRunner(model_name, dtype=DTYPE, is_cross_encoder=True) as hf_model:
        return {
            "1_1": hf_model.predict(pair_1_1).tolist(),
            "1_n": hf_model.predict(pairs_1_n).tolist(),
            "n_n": hf_model.predict(pairs_n_n).tolist(),
        }


@pytest.fixture(scope="module")
def vllm_cross_encoder(model_name, cross_encoder_goldens):
    with VllmRunner(model_name, **_VLLM_KWARGS) as vllm_model:
        yield vllm_model


def test_cross_encoder_score_1_to_1(vllm_cross_encoder, cross_encoder_goldens):
    hf_outputs = cross_encoder_goldens["1_1"]
    vllm_outputs = vllm_cross_encoder.score(TEXTS_1[0], TEXTS_2[0])

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1
    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


def test_cross_encoder_score_1_to_N(vllm_cross_encoder, cross_encoder_goldens):
    hf_outputs = cross_encoder_goldens["1_n"]
    vllm_outputs = vllm_cross_encoder.score(TEXTS_1[0], TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2
    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


def test_cross_encoder_score_N_to_N(vllm_cross_encoder, cross_encoder_goldens):
    hf_outputs = cross_encoder_goldens["n_n"]
    vllm_outputs = vllm_cross_encoder.score(TEXTS_1, TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2
    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.fixture(scope="module", params=EMBEDDING_MODELS)
def emb_model_name(request):
    yield snapshot_download(
        request.param,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
    )


@pytest.fixture(scope="module")
def embedding_goldens(emb_model_name):
    with HfRunner(emb_model_name, dtype=DTYPE, is_sentence_transformer=True) as hf_model:
        pair_1_1 = hf_model.encode([TEXTS_1[0], TEXTS_2[0]])
        pairs_1_n = [hf_model.encode(pair) for pair in [[TEXTS_1[0], TEXTS_2[0]], [TEXTS_1[0], TEXTS_2[1]]]]
        pairs_n_n = [hf_model.encode(pair) for pair in [[TEXTS_1[0], TEXTS_2[0]], [TEXTS_1[1], TEXTS_2[1]]]]
        return {
            "1_1": [F.cosine_similarity(*map(torch.tensor, pair_1_1), dim=0)],
            "1_n": [F.cosine_similarity(*map(torch.tensor, pair), dim=0) for pair in pairs_1_n],
            "n_n": [F.cosine_similarity(*map(torch.tensor, pair), dim=0) for pair in pairs_n_n],
        }


@pytest.fixture(scope="module")
def vllm_embedding(emb_model_name, embedding_goldens):
    with VllmRunner(emb_model_name, **_VLLM_KWARGS) as vllm_model:
        yield vllm_model


def test_embedding_score_1_to_1(vllm_embedding, embedding_goldens):
    hf_outputs = embedding_goldens["1_1"]
    vllm_outputs = vllm_embedding.score(TEXTS_1[0], TEXTS_2[0])

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1
    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


def test_embedding_score_1_to_N(vllm_embedding, embedding_goldens):
    hf_outputs = embedding_goldens["1_n"]
    vllm_outputs = vllm_embedding.score(TEXTS_1[0], TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2
    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


def test_embedding_score_N_to_N(vllm_embedding, embedding_goldens):
    hf_outputs = embedding_goldens["n_n"]
    vllm_outputs = vllm_embedding.score(TEXTS_1, TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2
    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)
