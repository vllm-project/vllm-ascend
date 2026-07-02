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
"""End-to-end consistency test for ShortRequestFirst prefill scheduling."""

import os
from typing import Any

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3-0.6B"

SHORT_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "1 + 1 equals",
]
LONG_PROMPTS = [
    "The following is a long passage that should be classified as a long "
    "prefill by the ShortRequestFirst scheduler. " * 8 + "In summary, the key point is",
    "Once upon a time in a faraway land, there lived a great many people who "
    "told stories about the stars. " * 8 + "The moral of the story is",
]
PROMPTS = SHORT_PROMPTS + LONG_PROMPTS

GREEDY = SamplingParams(temperature=0.0, max_tokens=32, min_tokens=16)
SHORT_REQUEST_FIRST_THRESHOLD = 16

SHORT_REQUEST_FIRST_ADDITIONAL_CONFIG = {
    "recompute_scheduler_enable": True,
    "short_request_first_config": {
        "enabled": True,
        "threshold": SHORT_REQUEST_FIRST_THRESHOLD,
        "long_max_wait_ms": 50.0,
    },
}


def _generate(additional_config):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    runner_kwargs: dict[str, Any] = dict(
        max_model_len=1024,
        enforce_eager=True,
        dtype="float16",
        gpu_memory_utilization=0.9,
    )
    if additional_config is not None:
        runner_kwargs["additional_config"] = additional_config
    with VllmRunner(MODEL, **runner_kwargs) as vllm_model:
        return vllm_model.generate(PROMPTS, sampling_params=GREEDY)


def test_short_request_first_matches_default_scheduler_outputs():
    baseline = _generate(additional_config=None)
    short_request_first = _generate(additional_config=SHORT_REQUEST_FIRST_ADDITIONAL_CONFIG)

    check_outputs_equal(
        outputs_0_lst=baseline,
        outputs_1_lst=short_request_first,
        name_0="default_scheduler",
        name_1="short_request_first_scheduler",
    )
