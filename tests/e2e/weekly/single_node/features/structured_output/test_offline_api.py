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

import pytest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.weekly.single_node.features.structured_output.assertions import assert_structured_output
from tests.e2e.weekly.single_node.features.structured_output.cases import (
    STRUCTURED_OUTPUT_CASES,
    StructuredOutputCase,
)


@pytest.mark.parametrize("case", STRUCTURED_OUTPUT_CASES, ids=lambda case: case.case_id)
def test_offline_structured_output(offline_runner: VllmRunner, case: StructuredOutputCase) -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=case.max_tokens,
        structured_outputs=StructuredOutputsParams(**case.structured_outputs_kwargs()),
    )
    inputs = offline_runner.get_inputs([case.prompt, case.prompt])

    outputs = offline_runner.model.generate(inputs, sampling_params=sampling_params)

    assert len(outputs) == 2
    for output in outputs:
        assert output.outputs
        assert_structured_output(output.outputs[0].text, case)


def test_offline_switches_constraints_without_state_leak(offline_runner: VllmRunner) -> None:
    for case in STRUCTURED_OUTPUT_CASES:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=case.max_tokens,
            structured_outputs=StructuredOutputsParams(**case.structured_outputs_kwargs()),
        )
        outputs = offline_runner.model.generate(
            offline_runner.get_inputs([case.prompt]),
            sampling_params=sampling_params,
        )

        assert outputs[0].outputs
        assert_structured_output(outputs[0].outputs[0].text, case)
