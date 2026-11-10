# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.ut.ops.test_gdn_attn_builder import (
    BatchSpec,
    _make_builder,
    _patch_missing_runtime_cdiv,
    create_common_attn_metadata,
)
from vllm_ascend.ops import gdn_attn_builder, kda_pcp


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("for_capture", [False, True])
def test_real_builder_pcp8_context_restores_and_scatters(rank, for_capture, monkeypatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    group = SimpleNamespace(rank_in_group=rank, world_size=8)

    def make_kimi(spec, layers, config, device):
        config.parallel_config.prefill_context_parallel_size = 8
        return kda_pcp.KimiKDAHostMetadataBuilder(spec, layers, config, device)

    with (
        patch.object(kda_pcp, "get_pcp_group", return_value=group),
        patch.object(gdn_attn_builder, "get_pcp_group", return_value=group),
    ):
        builder = _make_builder(
            device=torch.device("cpu"), num_heads=32, num_speculative_tokens=0, builder_cls=make_kimi
        )
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=[15], query_lens=[15]), block_size=16, device=torch.device("cpu")
        )
        # Rank-major all-gather of cyclic shards must restore causal order.
        restore = torch.tensor([2 * (i % 8) + i // 8 for i in range(15)])
        local_indices = torch.arange(rank, 15, 8)
        context = SimpleNamespace(local_num_tokens=2, hidden_restore_idx=restore, local_token_indices=local_indices)
        common.pcp_context = context
        if for_capture:
            metadata = builder.build_for_cudagraph_capture(common)
        else:
            metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    gathered = torch.tensor([[r, r + 8 if r < 7 else -1] for r in range(8)]).float().reshape(-1, 1)
    group.all_gather = lambda x, dim: gathered
    forward = SimpleNamespace(
        attn_metadata={"kda": metadata}, no_compile_layers={"kda": SimpleNamespace(pcp_group=group)}
    )
    with patch.object(kda_pcp, "get_forward_context", return_value=forward):
        restored = kda_pcp.kda_pcp_gather(gathered[2 * rank : 2 * rank + 2], 8, "kda")
        torch.testing.assert_close(restored[:15, 0], torch.arange(15).float())
        assert restored[15, 0] == 0
        output = kda_pcp.kda_pcp_scatter(restored * 10 + 1, 8, "kda")
    expected = torch.zeros(2)
    expected[: len(local_indices)] = local_indices.float() * 10 + 1
    torch.testing.assert_close(output[:, 0], expected)
