# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.manual.check_ple_gva import check_private


def test_ple_gva_lookup_graph_and_pool_isolation():
    memfabric = pytest.importorskip("memfabric_hybrid")
    if not hasattr(memfabric.offload, "Offload"):
        pytest.skip("requires MemFabric independent Offload instances")
    check_private(torch.npu.current_device(), check_adapter=True)
