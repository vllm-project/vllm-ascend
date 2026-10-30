# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target CP must not replicate the ordinary MLA Draft's TP projections."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.models.glm5_dspark import DeepseekV2MLAAttention, Glm5DSparkMLAAttention
from vllm_ascend.ops import linear_op


@pytest.mark.parametrize("constructor_error", [False, True])
def test_glm_draft_keeps_local_projection_heads_with_target_cp(monkeypatch, constructor_error):
    group = SimpleNamespace(world_size=16, rank_in_group=3)
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: group)
    monkeypatch.setattr(linear_op, "enable_dsa_cp", lambda: True)
    observed = []

    def parent_init(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.mla_attn = SimpleNamespace(mla_attn=object())
        for name in ("q_b_proj", "kv_b_proj"):
            op, rank, size = linear_op.get_parallel_op(
                False, f"model.layers.78.self_attn.{name}", MagicMock(), "column"
            )
            observed.append((op, rank, size))
        if constructor_error:
            raise RuntimeError("constructor failed after projection selection")

    monkeypatch.setattr(DeepseekV2MLAAttention, "__init__", parent_init)
    config = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        q_lora_rank=1536,
        kv_lora_rank=512,
        max_position_embeddings=131072,
    )

    def construct():
        return Glm5DSparkMLAAttention(
            vllm_config=SimpleNamespace(cache_config=None), config=config, prefix="model.layers.78.self_attn"
        )

    if constructor_error:
        with pytest.raises(RuntimeError, match="constructor failed"):
            construct()
    else:
        construct()
    assert observed == [(None, 3, 16), (None, 3, 16)]
    # The target's projection retains its existing CP replication after the
    # draft constructor; this is not a global disable or a prefix rename.
    op, rank, size = linear_op.get_parallel_op(False, "model.layers.0.self_attn.kv_b_proj", MagicMock(), "column")
    assert isinstance(op, linear_op.ShardedCPColumnParallelOp) and (rank, size) == (0, 1)
