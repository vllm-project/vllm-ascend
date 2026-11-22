# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check in-place INT64 representative selection on Ascend 950."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.hisparse import scatter_min_representatives
from vllm_ascend.utils import vllm_version_is

pytestmark = [
    pytest.mark.skipif(vllm_version_is("0.33.0"), reason="HiSparse requires a supported vLLM main revision"),
    pytest.mark.skipif(not is_950(), reason="Representative selection is qualified on Ascend 950"),
]


@torch.inference_mode()
def test_representatives_match_int64_amin_after_graph_replay():
    high = torch.iinfo(torch.int64).max
    low = torch.iinfo(torch.int64).min
    representatives = torch.tensor([high, 13, high, 19], device="npu")
    indices = torch.tensor([0, 0, 1, 2], device="npu")
    values = torch.tensor([7, 3, 5, 17], device="npu")
    assert scatter_min_representatives(representatives, indices, values) is representatives
    assert representatives.cpu().tolist() == [3, 5, 17, 19]

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scatter_min_representatives(representatives, indices, values)

    representatives.copy_(torch.tensor([high, 13, high, 19], device="npu"))
    values.copy_(torch.tensor([high, low, 17, 23], device="npu"))
    graph.replay()
    torch.npu.synchronize()
    assert representatives.cpu().tolist() == [low, 13, 23, 19]
