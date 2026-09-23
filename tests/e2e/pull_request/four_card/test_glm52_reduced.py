#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Four-card functional coverage for the layer-reduced GLM-5.2 W4A8 checkpoint.

The 78-layer source is resolved from the local ModelScope weight cache (no
network access) and cropped to 11 main layers inside the test, so the test model
construction stays local to the test. This is a functional startup/generation
check, not an accuracy gate: it does not apply the TP8 numerical baseline and
says nothing about GSM8K accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL_ID = "Eco-Tech/GLM-5.2-w4a8"
# Repo convention for a locally cached ModelScope checkpoint (see other configs).
DEFAULT_SOURCE = "/root/.cache/modelscope/hub/models/Eco-Tech/GLM-5.2-w4a8"
MAIN_LAYERS = 11
PROMPT = ["Question: What is 1 + 1?\nAnswer:\n"]


def _reduced_checkpoint() -> Path:
    """Return an 11-layer checkpoint, cropping the local source when needed."""
    provided = os.environ.get("GLM52_REDUCED_DIR")
    if provided:
        path = Path(provided)
        if not (path / "config.json").is_file():
            pytest.fail(f"GLM52_REDUCED_DIR={provided} has no config.json")
        return path

    source = Path(os.environ.get("GLM52_SOURCE_DIR", DEFAULT_SOURCE))
    if not (source / "config.json").is_file():
        pytest.skip(f"local GLM-5.2 source not found at {source}")

    from tools.glm_reduced.prepare import prepare_checkpoint

    repo = Path(__file__).resolve().parents[4]
    descriptor = repo / "tools/glm_reduced/data/glm52/source.json"
    cache = Path(os.environ.get("GLM52_CACHE_DIR", "~/.cache/vllm-ascend/glm-reduced")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    model, _ = prepare_checkpoint(descriptor, source, cache, layers=MAIN_LAYERS, download=False)
    return Path(model)


@pytest.mark.e2e_model(MODEL_ID)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_dsa",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_glm52_reduced_tp4_eager_generates() -> None:
    """The locally cropped 11-layer checkpoint starts at TP4 and generates tokens."""
    model = _reduced_checkpoint()
    with VllmRunner(
        str(model),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        quantization="ascend",
        gpu_memory_utilization=0.9,
        distributed_executor_backend="mp",
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(PROMPT, max_tokens=8)

    assert len(outputs) == 1
    assert outputs[0][1], "expected a non-empty completion"
