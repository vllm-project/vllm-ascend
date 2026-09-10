# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""GLM-5.2 W4A8 DSpark with pipeline parallelism on Model Runner V2.

Eight-card case (PP2 x TP4) mirroring ``test_spec_pp_accuracy.py`` (DeepSeek-V4
DSpark + PP) for GLM-5.2, which can additionally run the target in graph mode
while keeping async scheduling, prefix caching and chunked prefill enabled.
Pooling (PD/memcache) and EPLB are intentionally out of scope of this case.

``GLM52_TP_SIZE`` / ``GLM52_PP_SIZE`` widen the parallelism for larger hosts.
The same request is repeated ``GLM52_DSPARK_PP_REPEAT`` times (default 1) so the
case doubles as a stability retest of the PP speculative path.
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest
import regex as re
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

GLM52_MODEL = os.environ.get("GLM52_W4A8_MODEL_PATH", "Eco-Tech/GLM-5.2-w4a8")
GLM52_DRAFT_MODEL = os.environ.get(
    "GLM52_DSPARK_DRAFT_PATH",
    "RedHatAI/GLM-5.2-speculator.dspark",
)
# GLM-5.2 has 78 layers and layer 42 is a full-Indexer layer, so the manual PP
# split has to land there; the last stage also carries the local drafter.
GLM52_PP_LAYER_PARTITION = os.environ.get("GLM52_PP_LAYER_PARTITION", "42,36")
# Eight-card default (PP2 x TP4); override for wider hosts.
GLM52_TP_SIZE = int(os.environ.get("GLM52_TP_SIZE", "4"))
GLM52_PP_SIZE = int(os.environ.get("GLM52_PP_SIZE", "2"))
REPEAT = int(os.environ.get("GLM52_DSPARK_PP_REPEAT", "1"))

GSM8K_PROMPT = (
    'Answer the following question. The last line of the response should follow this format: "answer:$ANSWER" '
    "(without quotes), where ANSWER is a number. Let's think step by step.\n\n"
    "Question: Ali had $21. Leila gave him half of her $100. How much does Ali have now?"
)
GSM8K_ANSWER = "71"
ANSWER_RE = re.compile(r"answer\s*:\s*\$?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text) or NUMBER_RE.findall(text)
    assert matches, f"No numeric answer found in model output: {text!r}"
    normalized = matches[0].strip().replace(",", "").rstrip(".")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _spec_decode_counts(metrics) -> tuple[int, int]:
    """Return (num_drafts, num_accepted) collected from the runtime metrics.

    ``num_accepted_tokens`` and ``num_accepted_tokens_per_pos`` describe the same
    accepted tokens, so a single source is used instead of summing both.
    """
    num_drafts = 0
    num_accepted_counter: int | None = None
    num_accepted_per_pos = 0
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_counter = metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            num_accepted_per_pos = sum(metric.values)
    num_accepted = num_accepted_counter if num_accepted_counter is not None else num_accepted_per_pos
    return num_drafts, num_accepted


@pytest.mark.e2e_model(GLM52_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dspark",
    parallel="PP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True,pinned_mem_register:True",
        "VLLM_PP_LAYER_PARTITION": GLM52_PP_LAYER_PARTITION,
        "HCCL_BUFFSIZE": "2048",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_glm5_2_dspark_pp_full_decode_only() -> None:
    """GLM-5.2 W4A8 DSpark + PP2/TP8/EP in graph mode with all features on."""
    with VllmRunner(
        GLM52_MODEL,
        quantization="ascend",
        tensor_parallel_size=GLM52_TP_SIZE,
        pipeline_parallel_size=GLM52_PP_SIZE,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        max_model_len=8192,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.92,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        async_scheduling=True,
        compilation_config=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY"),
        disable_log_stats=False,
        speculative_config={
            "method": "dspark",
            "model": GLM52_DRAFT_MODEL,
            "num_speculative_tokens": 7,
            "enforce_eager": True,
        },
        additional_config={
            "enable_dsa_cp": False,
            "enable_fused_mc2": 0,
        },
    ) as runner:
        for attempt in range(REPEAT):
            started = time.monotonic()
            outputs = runner.generate_greedy([GSM8K_PROMPT], max_tokens=512)
            elapsed = time.monotonic() - started
            assert len(outputs) == 1
            _, output_text = outputs[0]
            assert _extract_answer(output_text) == GSM8K_ANSWER, (
                f"attempt {attempt + 1}/{REPEAT} returned an unexpected answer in {elapsed:.1f}s: {output_text!r}"
            )
            print(f"[glm5.2 dspark pp] attempt {attempt + 1}/{REPEAT} ok in {elapsed:.1f}s")
        metrics = runner.model.get_metrics()

    num_drafts, num_accepted = _spec_decode_counts(metrics)
    assert num_drafts > 0, "Speculative decoding did not generate draft tokens"
    assert num_accepted > 0, "Speculative decoding did not accept any draft tokens"
    acceptance_length = 1 + num_accepted / num_drafts
    print(f"[glm5.2 dspark pp] drafts={num_drafts} accepted={num_accepted} acceptance_length={acceptance_length:.3f}")
