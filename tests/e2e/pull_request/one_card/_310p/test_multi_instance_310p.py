#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Two 310P EP workers must retain comparable per-instance KV cache budgets."""

import pytest
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.device.device_config import is_310p
from vllm_ascend.utils import is_rc_device

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The result of 2 + 2 is"
MAX_TOKENS = 8
GPU_MEMORY_UTILIZATION = 0.3
MAX_MODEL_LEN = 2048
# Allow profiling variation while rejecting substantial second-instance capacity loss.
KV_BUDGET_RELATIVE_TOLERANCE = 0.1


class WorkerMemoryBudgetProbe:
    """Expose the already profiled worker budget through LLM.collective_rpc."""

    def get_multi_instance_memory_budget(self) -> int:
        return int(self.available_kv_cache_memory_bytes)


def _memory_budget(runner: VllmRunner) -> int:
    budgets = runner.model.collective_rpc("get_multi_instance_memory_budget")
    assert len(budgets) == 1, "This regression requires one worker per instance"
    assert budgets[0] > 0, f"Instance reported a non-positive KV cache budget: {budgets}"
    return budgets[0]


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="multi_instance,profiling",
    parallel="TP",
    deploy="pd_mix",
    hardware="310P",
    quantization="FP16",
    graph_mode="eager",
)
@wait_until_npu_memory_free(0.7)
def test_two_instances_preserve_kv_cache_capacity_310p(monkeypatch) -> None:
    """Keep both instances alive for capacity checks and greedy generation.

    At utilization 0.3, the old 310P calculation can still start the second
    instance while deducting the first instance's HBM from its cache budget.
    Comparing budgets catches that regression even when short inference succeeds.
    The device must allow both worker processes to use the same physical chip.
    """
    if not is_310p() or is_rc_device():
        pytest.skip("Requires a 310P EP device with discrete device memory")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    model_name = maybe_model_redirect(MODEL)
    settings = dict(
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_MODEL_LEN,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        worker_extension_cls=f"{__name__}.WorkerMemoryBudgetProbe",
    )
    with VllmRunner(model_name, **settings) as runner1:
        first_budget = _memory_budget(runner1)
        with VllmRunner(model_name, **settings) as runner2:
            second_budget = _memory_budget(runner2)
            print(f"310P KV cache budgets: first={first_budget} bytes, second={second_budget} bytes")
            assert second_budget == pytest.approx(first_budget, rel=KV_BUDGET_RELATIVE_TOLERANCE), (
                "Identical 310P workers should retain comparable per-instance KV cache capacity; "
                f"first={first_budget} bytes, second={second_budget} bytes. "
                "The second worker may have charged the first worker's HBM again."
            )
            outputs1 = runner1.generate_greedy([PROMPT], max_tokens=MAX_TOKENS)
            outputs2 = runner2.generate_greedy([PROMPT], max_tokens=MAX_TOKENS)
            for outputs in (outputs1, outputs2):
                assert len(outputs) == 1
                _, text = outputs[0]
                # VllmRunner includes the prompt in the returned text.
                assert text.startswith(PROMPT)
                assert text[len(PROMPT) :].strip(), "Instance generated no text after the prompt"
            assert outputs1 == outputs2, "Identical eager workers produced different greedy outputs"
