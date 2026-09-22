# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qualify the target/draft head counts and packed page sizes on A5."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.flash_mla_qualification import Qualification
from vllm_ascend.device.device_config import is_950


@pytest.mark.parametrize("heads", [12, 8])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("graph", [False, True])
def test_packed_mla_device_lengths_and_cache_views(heads, page_size, graph):
    if not torch.npu.is_available() or not is_950():
        pytest.skip("Requires an A5 NPU and the RFC 16464 pinned runtime.")
    args = SimpleNamespace(
        device=0,
        dtype="bfloat16",
        heads=heads,
        page_size=page_size,
        graph=graph,
        graph_only=graph,
        smoke=False,
        atol=0.02,
        rtol=0.03,
    )
    qualification = Qualification(args)
    qualification.writer()
    if graph:
        qualification.graph()
    else:
        qualification.eager()
