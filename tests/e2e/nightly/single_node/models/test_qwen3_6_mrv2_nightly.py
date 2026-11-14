# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Nightly regression for the Qwen3.6 family on Model Runner V2.

The pull-request suite covers the same configurations with short prompts. This
nightly variant reuses the recorded goldens but runs both representatives
(dense 27B and MoE 35B-A3B) plus the ~2.3k-token prompt, so a slow drift in the
hybrid GDN path shows up once a day instead of on every PR.

Goldens come from Model Runner V1 on vLLM-Ascend v0.30.0, which is the frozen
acceptance reference for this task; `qwen3_6_mrv2_goldens.json` is shared with
`tests/e2e/pull_request/one_card/model_runner_v2/test_qwen3_6_mrv2.py`.

Run with:
    pytest -sv tests/e2e/nightly/single_node/models/test_qwen3_6_mrv2_nightly.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL_MOE = os.getenv("QWEN36_MOE_MODEL", "Eco-Tech/Qwen3.6-35B-A3B-w8a8")
MODEL_MOE_REVISION = os.getenv("QWEN36_MOE_REVISION", "1a118d717bcbd59480f4a110fe22181d21711b4d")
MODEL_DENSE = os.getenv("QWEN36_DENSE_MODEL", "Eco-Tech/Qwen3.6-27B-w8a8")
MODEL_DENSE_REVISION = os.getenv("QWEN36_DENSE_REVISION", "430a017dfd20366728d3b32100ea2e9f40da595a")

PREFILL_ATOL = 0.0689
DECODE_ATOL = 0.1378
MAX_TOKENS = 24

GOLDENS = Path(__file__).parents[3] / "pull_request" / "qwen3_6_mrv2_goldens.json"

# Goldens were recorded without the deterministic HCCL/ATB flags; those flags
# shift the numerics (a run with them diverges at token 13).
_ENVS = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "HCCL_BUFFSIZE": "1024",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "10",
    "TASK_QUEUE_ENABLE": "1",
}


def _load_golden(key: str) -> dict:
    entries = json.loads(GOLDENS.read_text(encoding="utf-8"))["entries"]
    if key not in entries:
        pytest.skip(f"no golden recorded for {key}")
    return entries[key]


def _run(model: str, revision: str, prompt: str, *, expert_parallel: bool) -> dict:
    kwargs = {
        "max_model_len": 8192,
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "quantization": "ascend",
        "enable_expert_parallel": expert_parallel,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "compilation_config": {"cudagraph_mode": "NONE"},
        "speculative_config": {"method": "qwen3_5_mtp", "num_speculative_tokens": 1},
        "revision": revision,
    }
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0, top_k=0,
                            logprobs=20, seed=1234)
    with patch.dict(os.environ, _ENVS):
        with VllmRunner(model, **kwargs) as runner:
            output = runner.model.generate([prompt], params)[0].outputs[0]
    return {
        "token_ids": list(output.token_ids),
        "logprobs": [{str(token_id): value.logprob for token_id, value in step.items()}
                     for step in (output.logprobs or [])],
    }


def _assert_matches(golden: dict, observed: dict, key: str) -> None:
    assert golden["token_ids"] == observed["token_ids"], (
        f"{key}: token mismatch\ngolden={golden['token_ids']}\nobserved={observed['token_ids']}"
    )
    worst_prefill = 0.0
    worst_decode = 0.0
    violations = []
    for index in range(min(len(golden["token_ids"]), len(observed["token_ids"]))):
        token_id = str(golden["token_ids"][index])
        left = golden["logprobs"][index].get(token_id) if index < len(golden["logprobs"]) else None
        right = observed["logprobs"][index].get(token_id) if index < len(observed["logprobs"]) else None
        if left is None or right is None:
            continue
        delta = abs(left - right)
        if index == 0:
            worst_prefill = max(worst_prefill, delta)
            if delta > PREFILL_ATOL:
                violations.append((index, "prefill", delta))
        else:
            worst_decode = max(worst_decode, delta)
            if delta > DECODE_ATOL:
                violations.append((index, "decode", delta))
    assert not violations, (
        f"{key}: logprob outside tolerance (prefill {worst_prefill:.4f}/{PREFILL_ATOL}, "
        f"decode {worst_decode:.4f}/{DECODE_ATOL}): {violations}"
    )


@pytest.mark.e2e_model(MODEL_MOE)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_nightly_matches_golden() -> None:
    golden = _load_golden("moe_tp1_eager")
    observed = _run(MODEL_MOE, MODEL_MOE_REVISION, golden["prompt"], expert_parallel=True)
    _assert_matches(golden, observed, "moe_tp1_eager")


@pytest.mark.e2e_model(MODEL_MOE)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs,long_sequence",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_nightly_long_prompt_matches_golden() -> None:
    golden = _load_golden("moe_tp1_eager_long")
    # Use the recorded prompt verbatim; reconstructing it can drift by a token and
    # then the very first generated token already differs.
    observed = _run(MODEL_MOE, MODEL_MOE_REVISION, golden["prompt"], expert_parallel=True)
    _assert_matches(golden, observed, "moe_tp1_eager_long")


@pytest.mark.e2e_model(MODEL_DENSE)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp,logprobs",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_dense_nightly_matches_golden() -> None:
    golden = _load_golden("dense_tp1_eager")
    observed = _run(MODEL_DENSE, MODEL_DENSE_REVISION, golden["prompt"], expert_parallel=False)
    _assert_matches(golden, observed, "dense_tp1_eager")
