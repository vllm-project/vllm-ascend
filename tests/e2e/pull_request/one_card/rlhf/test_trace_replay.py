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

"""End-to-end trace replay coverage for the Ascend Model Runner V2 path."""

import math

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "Hello, my name is"
TRACE_TOKEN_IDS = [444, 2210, 13, 358, 2776, 264, 220, 17]


def test_trace_replay_outputs_requested_tokens_and_scores() -> None:
    """Replay a fixed response and return scores from the real logits."""
    sampling_params = SamplingParams(
        trace_decode_token_ids=TRACE_TOKEN_IDS,
        logprobs=5,
    )

    with VllmRunner(
        MODEL_NAME,
        max_model_len=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        enable_trace_replay=True,
    ) as runner:
        outputs = runner.generate_w_logprobs([PROMPT], sampling_params)

    output_ids, _, output_logprobs = outputs[0]
    assert output_ids == TRACE_TOKEN_IDS
    assert output_logprobs is not None
    assert len(output_logprobs) == len(TRACE_TOKEN_IDS)

    for token_id, step_logprobs in zip(TRACE_TOKEN_IDS, output_logprobs):
        assert step_logprobs is not None
        assert token_id in step_logprobs
        token_logprob = step_logprobs[token_id]
        assert math.isfinite(token_logprob.logprob)
        assert token_logprob.rank is not None
