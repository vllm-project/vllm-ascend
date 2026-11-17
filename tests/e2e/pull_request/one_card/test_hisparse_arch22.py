# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2/A3 Target/draft mirror integration coverage."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.fixture(autouse=True)
def _require_arch22():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("requires the arch22 kernel")
    if not hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"):
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-not-found, import-untyped]  # noqa: F401
    assert hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"), "rebuild the arch22 extension"


@pytest.mark.parametrize("graph_producer,shared_group", [(False, True), (True, False)])
def test_direct_target_mirror_survives_draft_writes(graph_producer, shared_group, record_property):
    from tests.e2e.pull_request.one_card.test_hisparse_integration import (
        test_direct_target_mirror_survives_legal_draft_writes,
    )

    test_direct_target_mirror_survives_legal_draft_writes(graph_producer, shared_group, record_property)
