# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""UNO Model Runner V2 smoke test on one Ascend NPU."""

from pathlib import Path

import pytest
import vllm.envs as envs
from huggingface_hub.constants import HF_HUB_OFFLINE
from vllm import SamplingParams
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner
from vllm_ascend.worker.v2.spec_decode.uno.speculator import (
    UPSTREAM_UNO_AVAILABLE,
)

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


def test_uno_v2_generates_and_verifies_drafts(
    monkeypatch: pytest.MonkeyPatch,
    uno_adapter_path: str,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)

    with VllmRunner(
        "Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        dtype="bfloat16",
        enforce_eager=True,
        async_scheduling=False,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.8,
        enable_lora=True,
        max_lora_rank=128,
        max_loras=2,
        max_cpu_loras=2,
        disable_log_stats=False,
        speculative_config={
            "method": "uno",
            "uno_lora_path": uno_adapter_path,
            "uno_mask_token_id": 151669,
            "num_speculative_tokens": 3,
        },
    ) as runner:
        outputs = runner.model.generate(
            ["The capital of France is"], sampling, use_tqdm=False
        )
        metrics = runner.model.get_metrics()

    assert len(outputs[0].outputs[0].token_ids) == 8
    draft_tokens = sum(
        metric.value
        for metric in metrics
        if isinstance(metric, Counter)
        and metric.name == "vllm:spec_decode_num_draft_tokens"
    )
    assert draft_tokens > 0
