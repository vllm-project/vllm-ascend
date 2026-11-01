# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks for the reduced GLM-5.1 IPC case."""

from tests.e2e.pull_request.rlhf.weight_transfer.models.glm import GLM51_CASE, GLM_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.registry import pytest_model_cases


def test_glm51_case_configuration():
    case = GLM51_CASE
    assert case.model == "ZhipuAI/GLM-5.1"
    assert case.hf_overrides == {"num_hidden_layers": 4, "n_routed_experts": 8}
    assert case.meta_config_attribute is None
    assert not case.full_transfer_matrix
    assert not case.server_env
    assert not case.extra_server_args
    assert (case,) == GLM_CASES
    assert [parameter.values for parameter in pytest_model_cases("npu_ipc", model_cases=GLM_CASES)] == [(case, False)]
    assert pytest_model_cases("hccl", model_cases=GLM_CASES) == []
