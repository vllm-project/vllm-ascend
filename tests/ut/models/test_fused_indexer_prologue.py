# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer


def test_update_keys_dispatches_fused_a2_a3_indexer_prologue(monkeypatch):
    indexer = object.__new__(DeepseekV41Indexer)
    nn.Module.__init__(indexer)
    indexer.owns_k = True
    indexer.use_fused_indexer_prologue_int8 = True
    indexer.rope_width = 64

    wk_weight = nn.Parameter(torch.zeros((128, 512), dtype=torch.bfloat16), requires_grad=False)
    norm_weight = nn.Parameter(torch.ones(128, dtype=torch.bfloat16), requires_grad=False)
    indexer.wk = SimpleNamespace(weight=wk_weight)
    indexer.k_norm = SimpleNamespace(weight=norm_weight, variance_epsilon=1e-6)

    k_cache = torch.zeros((2, 16, 1, 128), dtype=torch.int8)
    scale_cache = torch.zeros((2, 16, 1, 1), dtype=torch.float16)
    indexer.k_cache = SimpleNamespace(kv_cache=[(k_cache, scale_cache)])

    latent = torch.zeros((3, 512), dtype=torch.bfloat16)
    slots = torch.tensor([[0, 0], [0, 1], [-1, -1]], dtype=torch.int32)
    cos = torch.ones((3, 1, 1, 64), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    fused_op = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "fused_indexer_prologue", fused_op, raising=False)

    indexer.update_keys(latent, slots, cos, sin)

    fused_op.assert_called_once_with(
        latent,
        wk_weight,
        norm_weight,
        cos,
        sin,
        slots,
        k_cache,
        scale_cache,
        1e-6,
    )
