# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Greedy Uno parity on one Ascend NPU with the original Qwen3-8B adapter."""

from pathlib import Path

import pytest
import vllm.envs as envs
from huggingface_hub.constants import HF_HUB_OFFLINE
from vllm import SamplingParams
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory
from vllm_ascend.spec_decode.uno import UPSTREAM_UNO_AVAILABLE

pytestmark = pytest.mark.skipif(
    not UPSTREAM_UNO_AVAILABLE,
    reason="requires a vLLM revision containing Uno",
)


@pytest.fixture(scope="module")
def uno_adapter_path() -> str:
    snapshot = hf_api().snapshot_download(
        repo_id="s-sahoo/uno-qwen3-8B",
        revision="8819e09ac901e7290d8d89d62c98b9f756c602fe",
        allow_patterns=["adapter/*"],
        local_files_only=HF_HUB_OFFLINE,
    )
    return str(Path(snapshot) / "adapter")


def _draft_token_count(metrics) -> int:
    return sum(
        metric.value
        for metric in metrics
        if isinstance(metric, Counter) and metric.name == "vllm:spec_decode_num_draft_tokens"
    )


def test_uno_greedy_matches_base_model(
    monkeypatch: pytest.MonkeyPatch,
    uno_adapter_path: str,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    prompts = [
        "The capital of France is",
        "Explain why the sky appears blue in two sentences.",
    ]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=32,
        ignore_eos=True,
        seed=0,
    )
    common = {
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "dtype": "bfloat16",
        "enforce_eager": True,
        "async_scheduling": False,
        "max_model_len": 2048,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 256,
        "gpu_memory_utilization": 0.8,
        "enable_lora": True,
        "max_lora_rank": 128,
        "max_loras": 2,
        "max_cpu_loras": 2,
        "disable_log_stats": False,
    }

    with VllmRunner("Qwen/Qwen3-8B", **common) as reference:
        expected = reference.model.generate(prompts, sampling, use_tqdm=False)
    cleanup_dist_env_and_memory()

    with VllmRunner(
        "Qwen/Qwen3-8B",
        **common,
        speculative_config={
            "method": "uno",
            "uno_lora_path": uno_adapter_path,
            "uno_mask_token_id": 151669,
            "num_speculative_tokens": 3,
            "enforce_eager": True,
        },
    ) as speculative:
        actual = speculative.model.generate(prompts, sampling, use_tqdm=False)
        metrics = speculative.model.get_metrics()
    cleanup_dist_env_and_memory()

    assert _draft_token_count(metrics) > 0
    assert [list(item.outputs[0].token_ids) for item in actual] == [
        list(item.outputs[0].token_ids) for item in expected
    ]
