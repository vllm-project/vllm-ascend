#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#

import math
import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
    "A short story starts with",
]


@pytest.mark.parametrize(
    ("enforce_eager", "compilation_config", "additional_config"),
    [
        pytest.param(True, {}, {}, id="eager"),
        pytest.param(
            False,
            {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4],
            },
            {
                "ascend_compilation_config": {
                    "enable_npugraph_ex": False,
                    "fuse_norm_quant": False,
                    "fuse_qknorm_rope": False,
                    "fuse_muls_add": False,
                },
            },
            id="full_decode_only",
        ),
    ],
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "PYTORCH_NPU_ALLOC_CONF": "pinned_mem_register:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_trace_replay(
    enforce_eager: bool,
    compilation_config: dict,
    additional_config: dict,
) -> None:
    """Replay distinct traces through dynamic batches on Ascend MRV2."""
    with VllmRunner(
        MODEL,
        max_model_len=1024,
        max_num_seqs=len(PROMPTS),
        enforce_eager=enforce_eager,
        async_scheduling=True,
        enable_trace_replay=True,
        compilation_config=compilation_config,
        additional_config=additional_config,
    ) as runner:
        baseline_params = SamplingParams(
            temperature=0.0,
            max_tokens=8,
            ignore_eos=True,
        )
        baseline_outputs = runner.model.generate(PROMPTS, baseline_params)
        trace_lengths = [8, 6, 4, 7]
        traces = [
            list(output.outputs[0].token_ids[:trace_len])
            for output, trace_len in zip(
                baseline_outputs,
                trace_lengths,
                strict=True,
            )
        ]

        # An EOS token inside a trace must be replayed without ending the
        # request. This also ensures the test does not only replay greedy paths.
        eos_token_id = runner.model.get_tokenizer().eos_token_id
        assert eos_token_id is not None
        traces[0][1] = eos_token_id

        replay_params = [
            SamplingParams(
                trace_decode_token_ids=trace,
                logprobs=5,
            )
            for trace in traces
        ]
        replay_outputs = runner.model.generate(PROMPTS, replay_params)

        for output, expected_trace in zip(
            replay_outputs,
            traces,
            strict=True,
        ):
            sample = output.outputs[0]
            assert list(sample.token_ids) == expected_trace
            assert sample.logprobs is not None
            assert len(sample.logprobs) == len(expected_trace)

            for token_id, step_logprobs in zip(
                expected_trace,
                sample.logprobs,
                strict=True,
            ):
                trace_logprob = step_logprobs.get(token_id)
                assert trace_logprob is not None
                assert math.isfinite(trace_logprob.logprob)
