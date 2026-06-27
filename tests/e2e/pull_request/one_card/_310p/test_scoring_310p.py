# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from modelscope import snapshot_download  # type: ignore[import-untyped]

from tests.e2e.conftest import HfRunner, VllmRunner

CROSS_ENCODER_MODELS = [
    "BAAI/bge-reranker-v2-m3",  # Roberta
]


TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]

DTYPE = "float16"


@pytest.fixture(scope="module", params=CROSS_ENCODER_MODELS)
def model_name(request):
    yield snapshot_download(request.param)


@pytest.mark.e2e_features("310P", "multimodal", "eager_mode")
@pytest.mark.e2e_model("BAAI/bge-reranker-v2-m3")
def test_cross_encoder_score_1_to_1(model_name):
    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    with HfRunner(model_name, dtype=DTYPE, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict([text_pair]).tolist()

    with VllmRunner(
        model_name, runner="pooling", dtype=DTYPE, max_model_len=1024, enforce_eager=True, gpu_memory_utilization=0.6
    ) as vllm_model:
        vllm_outputs = vllm_model.score(text_pair[0], text_pair[1])

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


@pytest.mark.e2e_features("310P", "multimodal", "eager_mode")
@pytest.mark.e2e_model("BAAI/bge-reranker-v2-m3")
def test_cross_encoder_score_1_to_N(model_name):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]

    with HfRunner(model_name, dtype=DTYPE, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict(text_pairs).tolist()

    with VllmRunner(
        model_name, runner="pooling", dtype=DTYPE, max_model_len=1024, enforce_eager=True, gpu_memory_utilization=0.6
    ) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1[0], TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.mark.e2e_features("310P", "multimodal", "eager_mode")
@pytest.mark.e2e_model("BAAI/bge-reranker-v2-m3")
def test_cross_encoder_score_N_to_N(model_name):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    with HfRunner(model_name, dtype=DTYPE, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict(text_pairs).tolist()

    with VllmRunner(
        model_name, runner="pooling", dtype=DTYPE, max_model_len=1024, enforce_eager=True, gpu_memory_utilization=0.6
    ) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1, TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)
