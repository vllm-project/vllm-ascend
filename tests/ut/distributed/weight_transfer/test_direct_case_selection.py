# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep legacy skips while making Direct coverage explicit and runnable."""

from dataclasses import replace

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import QWEN35_CASE, pytest_model_cases


def test_direct_packed_is_not_hidden_by_layerwise_skip():
    legacy = pytest_model_cases("npu_ipc", model_cases=(QWEN35_CASE,))
    direct = pytest_model_cases("npu_ipc", model_cases=(QWEN35_CASE,), reload_mode="direct")
    assert any(mark.name == "skip" for mark in legacy[1].marks)
    assert not any(mark.name == "skip" for mark in direct[1].marks)


def test_direct_preserves_model_prerequisite_skip():
    case = replace(QWEN35_CASE, skip_reason="missing model assets")
    parameters = pytest_model_cases("npu_ipc", model_cases=(case,), reload_mode="direct")
    assert all(any(mark.name == "skip" for mark in parameter.marks) for parameter in parameters)


def test_invalid_reload_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported reload mode"):
        pytest_model_cases("npu_ipc", reload_mode="typo")
