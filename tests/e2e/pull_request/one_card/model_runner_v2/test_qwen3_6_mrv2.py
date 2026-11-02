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
#

"""MRV2 parity coverage for the Qwen3.6 family.

Target path: tests/e2e/pull_request/one_card/model_runner_v2/test_qwen3_6_mrv2.py

Reference:
- batch>1 batch-invariance defect: https://github.com/vllm-project/vllm-ascend/issues/20609
- prefill host-overhead regression: https://github.com/vllm-project/vllm-ascend/issues/20608
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL_35B = os.getenv("QWEN36_35B_MODEL", "Eco-Tech/Qwen3.6-35B-A3B-w8a8")
MODEL_27B = os.getenv("QWEN36_27B_MODEL", "Eco-Tech/Qwen3.6-27B-W8A8")

PREFILL_ATOL = 0.0689
DECODE_ATOL = 0.1378

SHORT_PROMPTS = [
    "The capital of France is",
    "用三句话解释什么是张量并行。",
]
LONG_PROMPTS = [
    "请阅读下面的材料并回答问题。" + "张量并行把单个矩阵切分到多张卡上，通过集合通信完成拼接。" * 120
    + "\n问题：张量并行主要解决什么问题？",
]

_COMMON_ENVS = {
    "HCCL_BUFFSIZE": "1024",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "10",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

_GRAPH_MODES = {
    "eager": {"cudagraph_mode": "NONE"},
    "full_decode_only": {"cudagraph_mode": "FULL_DECODE_ONLY"},
}


def _runner_kwargs(tp: int, graph_mode: str) -> dict:
    return {
        "max_model_len": 8192,
        "dtype": "float16",
        "tensor_parallel_size": tp,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": graph_mode == "eager",
        "compilation_config": _GRAPH_MODES[graph_mode],
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
    }


def _sampling_params(greedy: bool, max_tokens: int) -> SamplingParams:
    if greedy:
        return SamplingParams(max_tokens=max_tokens, temperature=0.0, logprobs=20)
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        seed=1234,
        logprobs=20,
    )


def _collect(model: str, runner_kwargs: dict, prompts: list[str], params: SamplingParams, v2: bool) -> list[dict]:
    env = dict(_COMMON_ENVS)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if v2 else "0"
    with patch.dict(os.environ, env):
        with VllmRunner(model, **runner_kwargs) as runner:
            outputs = []
            for prompt in prompts:
                outputs.extend(runner.model.generate([prompt], params))
    result = []
    for out in outputs:
        completion = out.outputs[0]
        result.append({
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "logprobs": [
                {str(token_id): value.logprob for token_id, value in step.items()}
                for step in (completion.logprobs or [])
            ],
        })
    return result


def _collect_batched(model: str, runner_kwargs: dict, prompts: list[str], params: SamplingParams, v2: bool) -> list[dict]:
    """Submit all prompts in one generate() call so they share a single batch."""
    env = dict(_COMMON_ENVS)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if v2 else "0"
    with patch.dict(os.environ, env):
        with VllmRunner(model, **runner_kwargs) as runner:
            outputs = runner.model.generate(prompts, params)
    result = []
    for out in outputs:
        completion = out.outputs[0]
        result.append({
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "logprobs": [
                {str(token_id): value.logprob for token_id, value in step.items()}
                for step in (completion.logprobs or [])
            ],
        })
    return result


def _compare(mrv1: list[dict], mrv2: list[dict]) -> dict:
    """Compare final token sequences; logprob is only checked on shared token indices."""
    stats = {"token_mismatch": [], "max_logprob_delta": 0.0}
    for index, (left, right) in enumerate(zip(mrv1, mrv2, strict=True)):
        left_ids = left["token_ids"]
        right_ids = right["token_ids"]
        if left_ids != right_ids:
            first = next(
                (i for i, (a, b) in enumerate(zip(left_ids, right_ids)) if a != b),
                min(len(left_ids), len(right_ids)),
            )
            stats["token_mismatch"].append({"prompt_index": index, "first_divergence": first})
        for step, token_id in enumerate(left_ids):
            left_value = left["logprobs"][step].get(str(token_id)) if step < len(left["logprobs"]) else None
            right_value = right["logprobs"][step].get(str(token_id)) if step < len(right["logprobs"]) else None
            if left_value is None or right_value is None:
                continue
            delta = abs(left_value - right_value)
            tolerance = PREFILL_ATOL if step == 0 else DECODE_ATOL
            if delta > tolerance:
                stats["max_logprob_delta"] = max(stats["max_logprob_delta"], delta)
    return stats


@pytest.mark.e2e_model(MODEL_35B)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@pytest.mark.parametrize("graph_mode", ["eager", "full_decode_only"])
@pytest.mark.parametrize("tp", [1])
@pytest.mark.parametrize("greedy", [True, False], ids=["sampling-greedy", "sampling-random"])
@pytest.mark.parametrize("prompts", [SHORT_PROMPTS, LONG_PROMPTS], ids=["short", "long"])
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen36_35b_batch1_mrv1_mrv2_parity(
    tp: int, graph_mode: str, greedy: bool, prompts: list[str]
) -> None:
    """batch=1: MRV2 must match MRV1 token IDs, with logprob within repo tolerance."""
    kwargs = _runner_kwargs(tp, graph_mode)
    params = _sampling_params(greedy, max_tokens=24)
    mrv1 = _collect(MODEL_35B, kwargs, prompts, params, v2=False)
    mrv2 = _collect(MODEL_35B, kwargs, prompts, params, v2=True)
    stats = _compare(mrv1, mrv2)
    if greedy:
        # Greedy decoding is deterministic: token IDs must match and logprob must stay in tolerance.
        assert not stats["token_mismatch"], f"batch=1 token mismatch: {stats['token_mismatch']}"
        assert stats["max_logprob_delta"] <= DECODE_ATOL, (
            f"batch=1 logprob delta {stats['max_logprob_delta']} exceeds {DECODE_ATOL}"
        )
    else:
        # Random sampling uses runner-specific RNG state, so token IDs are not required to match.
        # This combination is a serving smoke: both runners must produce a full, non-empty output.
        expected = params.max_tokens
        for label, result in (("mrv1", mrv1), ("mrv2", mrv2)):
            assert result, f"{label} produced no output"
            assert all(o["token_ids"] for o in result), f"{label} produced empty token sequence"
            assert all(len(o["token_ids"]) <= expected for o in result)


@pytest.mark.e2e_model(MODEL_35B)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A3",
    graph_mode="eager",
)
@pytest.mark.parametrize("tp", [2], ids=["tp2"])
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen36_35b_batch1_tp2_parity(tp: int) -> None:
    """Multi-card TP=2: MRV2 must match MRV1 on the same input."""
    kwargs = _runner_kwargs(tp, "eager")
    params = _sampling_params(greedy=True, max_tokens=24)
    mrv1 = _collect(MODEL_35B, kwargs, SHORT_PROMPTS, params, v2=False)
    mrv2 = _collect(MODEL_35B, kwargs, SHORT_PROMPTS, params, v2=True)
    stats = _compare(mrv1, mrv2)
    assert not stats["token_mismatch"], f"TP=2 token mismatch: {stats['token_mismatch']}"


@pytest.mark.e2e_model(MODEL_27B)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen36_27b_dense_batch1_parity() -> None:
    """Dense representative: batch=1 MRV2/MRV1 parity."""
    kwargs = _runner_kwargs(tp=1, graph_mode="eager")
    params = _sampling_params(greedy=True, max_tokens=24)
    mrv1 = _collect(MODEL_27B, kwargs, SHORT_PROMPTS, params, v2=False)
    mrv2 = _collect(MODEL_27B, kwargs, SHORT_PROMPTS, params, v2=True)
    stats = _compare(mrv1, mrv2)
    assert not stats["token_mismatch"], f"27B token mismatch: {stats['token_mismatch']}"
    assert stats["max_logprob_delta"] <= DECODE_ATOL


@pytest.mark.e2e_model(MODEL_35B)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,chunked_prefill",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@pytest.mark.xfail(
    reason="batch-invariance defect shared by MRV1 and MRV2, tracked in #20609; "
    "chunked prefill is mandatory for mamba cache mode 'align'",
    strict=True,
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen36_35b_batch2_identical_prompts() -> None:
    """Known issue: two identical prompts in one batch diverge from index 13."""
    kwargs = _runner_kwargs(tp=1, graph_mode="eager")
    params = _sampling_params(greedy=True, max_tokens=24)
    prompts = [SHORT_PROMPTS[1], SHORT_PROMPTS[1]]
    mrv1 = _collect_batched(MODEL_35B, kwargs, prompts, params, v2=False)
    mrv2 = _collect_batched(MODEL_35B, kwargs, prompts, params, v2=True)
    assert mrv1[0]["token_ids"] == mrv1[1]["token_ids"], "MRV1 is not self-consistent within one batch"
    assert mrv2[0]["token_ids"] == mrv2[1]["token_ids"], "MRV2 is not self-consistent within one batch"


@pytest.mark.e2e_model(MODEL_35B)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen36_35b_prompt_exceeds_max_model_len() -> None:
    """Abnormal config: prompt longer than max_model_len must fail with a clear error."""
    kwargs = _runner_kwargs(tp=1, graph_mode="eager")
    kwargs["max_model_len"] = 1024
    params = _sampling_params(greedy=True, max_tokens=1)
    with pytest.raises(Exception, match="maximum context length"):
        _collect(MODEL_35B, kwargs, ["token " * 4000], params, v2=True)


@pytest.mark.e2e_model(MODEL_35B)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen36_35b_tp_exceeds_devices() -> None:
    """Abnormal config: TP larger than available devices must fail with a clear error."""
    kwargs = _runner_kwargs(tp=32, graph_mode="eager")
    params = _sampling_params(greedy=True, max_tokens=1)
    with pytest.raises((ValueError, RuntimeError), match="tensor|device|world size"):
        _collect(MODEL_35B, kwargs, SHORT_PROMPTS[:1], params, v2=True)

@pytest.mark.e2e_model(MODEL_35B)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen36_35b_multimodal_smoke() -> None:
    """Model-specific capability: Qwen3.6 is multimodal, both runners must serve an image prompt.

    e2e_coverage is intentionally omitted: the taxonomy has no multimodal feature value yet.
    """
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is this image?"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }]
    kwargs = _runner_kwargs(tp=1, graph_mode="eager")
    params = SamplingParams(max_tokens=8, temperature=0.0)

    results = {}
    for label, v2 in (("mrv1", False), ("mrv2", True)):
        env = dict(_COMMON_ENVS)
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if v2 else "0"
        with patch.dict(os.environ, env):
            with VllmRunner(MODEL_35B, **kwargs) as runner:
                output = runner.model.chat(messages, params)[0]
        results[label] = output.outputs[0].text

    for label, text in results.items():
        assert text and text.strip(), f"{label} returned empty multimodal output"

