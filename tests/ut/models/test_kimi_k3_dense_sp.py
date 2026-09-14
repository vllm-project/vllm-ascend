# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_ascend.models import kimi_k3


@pytest.mark.parametrize("rank", range(16))
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_dense_adapter_preserves_token_rows(rank, sequence_parallel):
    x = torch.arange(16 * 8, dtype=torch.float64).view(16, 8)
    calls = []

    def init(module, **kwargs):
        nn.Module.__init__(module)
        assert kwargs == {}

    def upstream(module, value):
        calls.append("tp_mlp")
        return value.square() + 2

    def gather(local):
        calls.append("gather")
        assert torch.equal(local, x[rank : rank + 1])
        return x

    def shard(value):
        calls.append("shard")
        return value[rank : rank + 1]

    with (
        patch.object(kimi_k3.KimiMLP, "__init__", init),
        patch.object(kimi_k3.KimiMLP, "forward", upstream),
        patch.object(kimi_k3, "sp_all_gather", gather),
        patch.object(kimi_k3, "sp_shard", shard),
    ):
        model = kimi_k3.AscendKimiMLP(use_sequence_parallel=sequence_parallel)
        given = x[rank : rank + 1] if sequence_parallel else x
        result = model(given)
        assert torch.equal(result, given.square() + 2)
        assert calls == (["gather", "tp_mlp", "shard"] if sequence_parallel else ["tp_mlp"])
