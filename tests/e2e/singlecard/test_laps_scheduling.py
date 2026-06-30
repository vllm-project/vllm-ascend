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
"""End-to-end consistency test for LAPS length-aware prefill scheduling.

LAPS only reorders *when* prefills are admitted (short vs. long, with
anti-starvation aging); it must not change the tokens a request produces. This
test runs a fixed greedy workload twice — once with the default scheduler and
once with LAPS enabled (via ``recompute_scheduler_enable`` + ``laps_config``) —
and asserts the outputs are identical. The prompts deliberately mix lengths so
both the short and long sub-queues, and the aging path, are exercised.

Requires NPU hardware; runs as part of the singlecard e2e suite.
"""

import os
from typing import Any

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"

# Mix short and long prompts so both LAPS sub-queues are populated. The long
# prompt is repeated/padded text so it clearly exceeds the (small) threshold.
SHORT_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "1 + 1 equals",
]
LONG_PROMPTS = [
    "The following is a long passage that should be classified as a long "
    "prefill by the LAPS scheduler. " * 8 + "In summary, the key point is",
    "Once upon a time in a faraway land, there lived a great many people who "
    "told stories about the stars. " * 8 + "The moral of the story is",
]
PROMPTS = SHORT_PROMPTS + LONG_PROMPTS

GREEDY = SamplingParams(temperature=0.0, max_tokens=32, min_tokens=16)

# Short-prefill threshold (tokens): chosen so SHORT_PROMPTS land in the short
# queue and LONG_PROMPTS in the long queue.
LAPS_THRESHOLD = 16

LAPS_ADDITIONAL_CONFIG = {
    "recompute_scheduler_enable": True,
    "laps_config": {
        "enabled": True,
        "threshold": LAPS_THRESHOLD,
        # Exercise the anti-starvation aging + token-bucket path.
        "long_max_wait_ms": 50.0,
        "long_token_reservation": 0.2,
        "long_burst_steps": 4,
    },
}


def _generate(additional_config):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    runner_kwargs: dict[str, Any] = dict(
        max_model_len=1024,
        enforce_eager=True,
        dtype="float16",  # avoid precision drift between the two runs
        gpu_memory_utilization=0.9,
    )
    if additional_config is not None:
        runner_kwargs["additional_config"] = additional_config
    with VllmRunner(MODEL, **runner_kwargs) as vllm_model:
        return vllm_model.generate(PROMPTS, sampling_params=GREEDY)


def test_laps_matches_default_scheduler_outputs():
    """LAPS must not change generated tokens versus the default scheduler."""
    baseline = _generate(additional_config=None)
    laps = _generate(additional_config=LAPS_ADDITIONAL_CONFIG)

    check_outputs_equal(
        outputs_0_lst=baseline,
        outputs_1_lst=laps,
        name_0="default_scheduler",
        name_1="laps_scheduler",
    )
