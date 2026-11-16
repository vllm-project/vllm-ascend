# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.glm5next.kda import PCPRowParallelLinear


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("use_mapping", [False, True])
def test_pcp_output_reduction_matches_allreduce_then_local_mapping(rank, use_mapping):
    layer = PCPRowParallelLinear.__new__(PCPRowParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.input_is_parallel = True
    layer.tp_rank = rank
    layer.tp_size = 4
    layer.skip_bias_add = False
    layer.bias = None
    layer.reduce_results = True
    layer.return_bias = True
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    layer.quant_method = SimpleNamespace(apply=lambda _, inputs, bias: inputs @ weight)
    group = SimpleNamespace(
        all_reduce=Mock(side_effect=lambda values: values * 4),
        reduce_scatter=Mock(side_effect=lambda values, dim: (values * 4).chunk(4, dim=dim)[rank]),
    )
    layer.tp_group = group
    inputs = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    mapping = torch.tensor([0, 7, 1, 6, 2, 5, 3, 4])
    output, bias = layer(inputs, pcp_output_indices=mapping if use_mapping else None)
    full_output = (inputs @ weight) * 4
    expected = full_output[mapping].chunk(4)[rank] if use_mapping else full_output
    assert torch.equal(output, expected)
    assert bias is None
    assert group.reduce_scatter.call_count == int(use_mapping)
    assert group.all_reduce.call_count == int(not use_mapping)
