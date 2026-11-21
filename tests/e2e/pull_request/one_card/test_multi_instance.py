#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""
Two VllmRunner instances are nested so that the first instance's worker
process is still holding NPU memory when the second instance's worker process
starts.  Both instances must:

  1. Initialize without raising any exception (no OOM during
     determine_available_memory / KV-cache allocation).
  2. Successfully complete a short generation request.

The model is Qwen/Qwen3-0.6B (~0.5 GiB weights) and gpu_memory_utilization
is set to 0.4 per instance so that two instances comfortably fit on a single
64 GiB Ascend 910B card while leaving enough headroom to avoid the
pre-fix negative-KV-cache condition.
"""

import os
from typing import Any

import pytest

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen3-0.6B"
_PROMPTS = ["Hello, my name is"]
_MAX_TOKENS = 5
# Use a low utilization so two instances fit side-by-side on one card:
#   2 × 0.4 × card_total ≤ card_total  (holds for any card ≥ 1 GiB)
_GPU_MEM_UTIL = 0.4
_MAX_MODEL_LEN = 512


def test_two_instances_on_single_card() -> None:
    """
    Regression test for PR #7427 (multi-instance OOM on single card).

    Start a first vllm-ascend instance; while it is still running and holding
    NPU memory, start a second instance with identical settings.  Both must
    initialize correctly and produce non-empty outputs.

    Failure signature (pre-fix):
        RuntimeError / ValueError during the second instance's init, or
        "Available KV cache memory: -X.XX GiB" in the logs followed by
        zero KV blocks being allocated.
    """
    # ── First instance ──────────────────────────────────────────────────
    with VllmRunner(
        MODEL,
        max_model_len=_MAX_MODEL_LEN,
        gpu_memory_utilization=_GPU_MEM_UTIL,
        enforce_eager=True,
    ) as runner1:
        # ── Second instance starts while first is still alive ────────────
        # This is the exact scenario from PR #7427: the second worker process
        # sees a reduced init_snapshot.free_memory because the first instance's
        # worker is still holding NPU memory.
        with VllmRunner(
            MODEL,
            max_model_len=_MAX_MODEL_LEN,
            gpu_memory_utilization=_GPU_MEM_UTIL,
            enforce_eager=True,
        ) as runner2:
            outputs2 = runner2.generate_greedy(_PROMPTS, max_tokens=_MAX_TOKENS)

        outputs1 = runner1.generate_greedy(_PROMPTS, max_tokens=_MAX_TOKENS)

    # ── Assertions ───────────────────────────────────────────────────────
    assert outputs1, "First instance produced no outputs"
    assert outputs2, "Second instance produced no outputs"

    _, text1 = outputs1[0]
    _, text2 = outputs2[0]

    assert text1, "First instance output text is empty — model may have failed to run"
    assert text2, (
        "Second instance output text is empty — "
        "KV cache may have been allocated with zero blocks (pre-fix OOM regression)"
    )


class ModelLoadProbe:
    """Check extension callbacks on workers created by both executors."""

    model_runner: Any

    def before_load_model(self, *, load_dummy_weights):
        assert not load_dummy_weights
        assert self.model_runner is not None
        assert getattr(self.model_runner, "model", None) is None
        self.model_load_events = ["before"]

    def after_load_model(self, *, load_dummy_weights):
        assert not load_dummy_weights
        assert self.model_load_events == ["before"]
        parameters = list(self.model_runner.get_model().parameters())
        assert parameters and all(p.device.type == "npu" for p in parameters)
        self.model_load_events.append("after")

    def inspect_model_load_events(self):
        return self.model_load_events


@pytest.mark.parametrize("executor", ["uni", "mp"])
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="",
    parallel="",
    deploy="pd_mix",
    hardware="",
    quantization="FP16",
    graph_mode="eager",
)
def test_worker_extension_model_load(executor):
    prompts = ["Hello", "The capital of France is"]
    kwargs = dict(
        dtype="float16",
        max_model_len=128,
        max_num_seqs=2,
        enforce_eager=True,
        distributed_executor_backend=executor,
        seed=0,
    )
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    with VllmRunner(model, **kwargs) as baseline:
        expected = baseline.generate_greedy(prompts, max_tokens=4)
    with VllmRunner(
        model,
        worker_extension_cls=f"{__name__}.ModelLoadProbe",
        **kwargs,
    ) as runner:
        assert runner.model.collective_rpc("inspect_model_load_events") == [["before", "after"]]
        assert runner.generate_greedy(prompts, max_tokens=4) == expected
