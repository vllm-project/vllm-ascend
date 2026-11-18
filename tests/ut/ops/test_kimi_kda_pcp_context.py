# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from tests.ut.ops.test_gdn_attn_builder import (
    BatchSpec,
    _make_builder,
    _patch_missing_runtime_cdiv,
    create_common_attn_metadata,
)
from vllm_ascend.ops import gdn_attn_builder, kda_pcp
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


def _build_runner_metadata(builder, common, for_capture, pcp_context=None):
    group = SimpleNamespace(layer_names=["kda"], get_metadata_builder=lambda _: builder)
    return build_attn_metadata(
        attn_groups=[[group]],
        num_reqs=common.num_reqs,
        num_tokens=common.num_actual_tokens,
        query_start_loc_gpu=common.query_start_loc,
        query_start_loc_cpu=common.query_start_loc_cpu,
        max_query_len=common.max_query_len,
        seq_lens=common.seq_lens,
        max_seq_len=common.max_seq_len,
        block_tables=(common.block_table_tensor,),
        slot_mappings=(common.slot_mapping,),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=builder.kv_cache_spec)]),
        seq_lens_np=common.seq_lens_cpu.numpy(),
        is_prefilling=common.is_prefilling,
        pcp_context=pcp_context,
        for_cudagraph_capture=for_capture,
    )["kda"]


@pytest.mark.parametrize("num_reqs", [1, 8, 16])
@pytest.mark.parametrize("for_capture", [False, True])
def test_kimi_decode_without_pcp_through_runner(num_reqs, for_capture, monkeypatch):
    """Exercise the FULL decode path used by TP8/PCP1 startup."""
    _patch_missing_runtime_cdiv(monkeypatch)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        builder_cls=kda_pcp.KimiKDAHostMetadataBuilder,
        mamba_cache_mode="align",
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8] * num_reqs, query_lens=[1] * num_reqs),
        block_size=16,
        device=torch.device("cpu"),
    )

    metadata = _build_runner_metadata(builder, common, for_capture)

    assert metadata.pcp_context is None
    assert metadata.num_decodes == num_reqs
    assert metadata.num_decode_tokens == num_reqs
    assert not hasattr(metadata, "pcp_restore_idx")


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
        metadata = _build_runner_metadata(builder, common, for_capture, context)
        assert metadata.pcp_context is context

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
