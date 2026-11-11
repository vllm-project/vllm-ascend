# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Qwen3.6-35B-A3B TP=2 regression on Model Runner V2.

Lives under ``two_card/`` because it needs two NPUs: the CI routing in
``.github/workflows/scripts/test_config.yaml`` maps ``one_card/`` to the single
card partition, so a TP=2 case placed there could never run.

Same golden-based contract as the single-card suite; goldens are keyed by TP so a
TP=1 recording can never be silently reused for TP=2.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.getenv("QWEN36_MOE_MODEL", "Eco-Tech/Qwen3.6-35B-A3B-w8a8")
MODEL_REVISION = os.getenv("QWEN36_MOE_REVISION", "1a118d717bcbd59480f4a110fe22181d21711b4d")

PREFILL_ATOL = 0.0689
DECODE_ATOL = 0.1378
MAX_TOKENS = 24
SPEC_TOKENS = 1
PROMPT_MAIN = "用三句话解释什么是张量并行。"

GOLDENS = Path(__file__).parents[2] / "qwen3_6_mrv2_goldens.json"

# No deterministic HCCL/ATB flags here: they shift the numerics (a run with them
# diverges from a run without them at token 13), and the golden was recorded
# without them. See the single-card suite for the rationale.
_COMMON_ENVS = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "HCCL_BUFFSIZE": "1024",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "10",
    "TASK_QUEUE_ENABLE": "1",
}


def _runner_kwargs() -> dict:
    return {
        "max_model_len": 8192,
        "dtype": "float16",
        "tensor_parallel_size": 2,
        "quantization": "ascend",
        "enable_expert_parallel": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "compilation_config": {"cudagraph_mode": "NONE"},
        "speculative_config": {"method": "qwen3_5_mtp", "num_speculative_tokens": SPEC_TOKENS},
        "revision": MODEL_REVISION,
    }


def _load_golden(key: str) -> dict:
    if not GOLDENS.exists():
        pytest.skip(f"{GOLDENS.name} is missing; record goldens before enabling this case")
    entries = json.loads(GOLDENS.read_text(encoding="utf-8")).get("entries", {})
    if key not in entries:
        pytest.skip(f"no golden recorded for {key}; record it instead of removing the case")
    return entries[key]


def _collect(prompt: str) -> dict:
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0, top_k=0, logprobs=20, seed=1234)
    with patch.dict(os.environ, _COMMON_ENVS):
        with VllmRunner(MODEL, **_runner_kwargs()) as runner:
            output = runner.model.generate([prompt], params)[0]
    completion = output.outputs[0]
    return {
        "token_ids": list(completion.token_ids),
        "logprobs": [
            {str(token_id): value.logprob for token_id, value in step.items()}
            for step in (completion.logprobs or [])
        ],
    }


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs",
    parallel="TP,EP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A3",
    graph_mode="eager",
)
@pytest.mark.xfail(
    strict=False,
    reason="TP=2 is not reproducible run to run even for the reference runner on v0.30.0 "
    "(two MRV1 cold starts differ at token 13), so a frozen TP=2 golden cannot be asserted; "
    "run-to-run comparisons of the two runners in one session do agree",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_tp2_matches_golden() -> None:
    """TP=2 must reproduce the recorded TP=2 greedy output within the repository tolerance."""
    key = "moe_tp2_eager"
    golden = _load_golden(key)
    observed = _collect(golden.get("prompt", PROMPT_MAIN))

    assert golden["token_ids"] == observed["token_ids"], (
        f"{key}: token mismatch\ngolden={golden['token_ids']}\nobserved={observed['token_ids']}"
    )
    worst = 0.0
    for index in range(min(len(golden["token_ids"]), len(observed["token_ids"]))):
        token_id = str(golden["token_ids"][index])
        left = golden["logprobs"][index].get(token_id) if index < len(golden["logprobs"]) else None
        right = observed["logprobs"][index].get(token_id) if index < len(observed["logprobs"]) else None
        if left is None or right is None:
            continue
        atol = PREFILL_ATOL if index == 0 else DECODE_ATOL
        delta = abs(left - right)
        assert delta <= atol, f"{key}: step {index} delta {delta:.4f} > {atol}"
        worst = max(worst, delta)
    assert worst <= DECODE_ATOL
