# SPDX-License-Identifier: Apache-2.0
"""Ordering and fallback contracts for the optional decode overlap path."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3SparseAttention
from vllm_ascend.models.minimax_m3.ops.indexer_dcp import gather_index_queries_async, index_shard_scores


def test_overlap_forward_keeps_projections_inside_opaque_op():
    hidden, positions = torch.empty(4, 6144), torch.arange(4)
    layer = SimpleNamespace(
        dcp_overlap=True,
        _can_fuse_sparse_prepare=Mock(return_value=True),
        qkv_proj=Mock(side_effect=AssertionError("projection escaped overlap op")),
        indexer_proj=Mock(side_effect=AssertionError("projection escaped overlap op")),
        q_size=2048,
        layer_name="main",
        o_proj=Mock(side_effect=lambda x: (x, None)),
    )
    with patch.object(torch.ops.vllm, "minimax_m3_dcp_overlap_forward") as op:
        output = MiniMaxM3SparseAttention.forward(layer, positions, hidden)
    op.assert_called_once_with(hidden, positions, output, "main")
    assert output.shape == (4, 2048)


def layer_and_context():
    norm = SimpleNamespace(weight=torch.zeros(128), variance_epsilon=1e-6)
    cache = SimpleNamespace(prefix="index", kv_cache=(torch.empty(2, 32, 128),))
    layer = SimpleNamespace(
        layer_name="main",
        num_idx_heads=1,
        idx_head_dim=128,
        num_heads=16,
        num_kv_heads=1,
        q_norm=norm,
        k_norm=norm,
        index_q_norm=norm,
        index_k_norm=norm,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(32, 64)),
        kv_cache=(torch.empty(2, 128, 1, 128), torch.empty(2, 128, 1, 128)),
        qkv_proj=Mock(return_value=(torch.empty(4, 2304), None)),
        indexer_proj=Mock(return_value=(torch.empty(4, 256), None)),
        indexer=SimpleNamespace(
            index_cache=cache, impl=SimpleNamespace(decode_dcp=Mock(return_value=("topk", "counts")))
        ),
        impl=SimpleNamespace(forward=Mock()),
        _run_fused_sparse_attention=Mock(),
    )
    slots = torch.tensor([1, 2, 3])
    md = SimpleNamespace(num_prefills=0, num_decodes=3, num_decode_tokens=3, num_actual_tokens=3, slot_mapping=slots)
    return layer, SimpleNamespace(attn_metadata={"main": md, "index": md})


def test_allgather_wait_follows_main_prepare_and_precedes_score_consumption():
    layer, context = layer_and_context()
    context.attn_metadata["index"].num_decode_tokens = 4
    order = []
    layer.indexer_proj.side_effect = lambda _: (order.append("index_projection") or torch.empty(4, 256), None)
    layer.qkv_proj.side_effect = lambda _: (order.append("main_projection") or torch.empty(4, 2304), None)
    work = SimpleNamespace(wait=Mock(side_effect=lambda: order.append("wait")))
    gathered = torch.empty(4, 3, 1, 128)
    layer.indexer.impl.decode_dcp.side_effect = lambda *a, **kw: (order.append("score") or "topk", "counts")
    layer.impl.forward.side_effect = lambda *a: order.append("attention")
    module = "vllm_ascend.ops.triton.msa.minimax_dcp_prepare"
    with (
        patch("vllm_ascend.models.minimax_m3.minimax_m3.get_forward_context", return_value=context),
        patch(
            "vllm_ascend.models.minimax_m3.minimax_m3.get_tp_group",
            return_value=SimpleNamespace(world_size=4, rank_in_group=2),
        ),
        patch(
            module + ".minimax_dcp_index_prepare",
            side_effect=lambda *a: order.append("index_prepare") or torch.empty(4, 128),
        ),
        patch(
            module + ".minimax_dcp_main_prepare",
            side_effect=lambda *a: order.append("main_prepare") or torch.empty(4, 2048),
        ),
        patch(
            "vllm_ascend.models.minimax_m3.ops.indexer_dcp.gather_index_queries_async",
            side_effect=lambda *a: (order.append("gather") or gathered, work),
        ) as gather,
    ):
        MiniMaxM3SparseAttention._run_dcp_overlap(layer, torch.empty(4, 6144), torch.arange(4), torch.empty(4, 2048))
    assert order == [
        "index_projection",
        "index_prepare",
        "gather",
        "main_projection",
        "main_prepare",
        "wait",
        "score",
        "attention",
    ]
    # Padding queries are excluded from communication, while main prepare retains them.
    assert gather.call_args.args[0].shape == (3, 1, 128)
    layer.indexer.impl.decode_dcp.assert_called_once_with(gathered, context.attn_metadata["index"], gathered_query=True)
    work.wait.assert_called_once()


@pytest.mark.parametrize("mode", ["prefill", "mixed", "empty", "profile"])
def test_non_decode_batches_keep_original_fused_path(mode):
    layer, context = layer_and_context()
    md = context.attn_metadata["index"]
    if mode in ("prefill", "mixed"):
        md.num_prefills = 1
    if mode in ("prefill", "empty"):
        md.num_decodes = 0
    if mode == "profile":
        context.attn_metadata = None
    output = torch.ones(4, 2048)
    with (
        patch("vllm_ascend.models.minimax_m3.minimax_m3.get_forward_context", return_value=context),
        patch(
            "vllm_ascend.models.minimax_m3.ops.indexer_dcp.gather_index_queries_async",
            side_effect=AssertionError("unexpected gather"),
        ),
    ):
        MiniMaxM3SparseAttention._run_dcp_overlap(layer, torch.empty(4, 6144), torch.arange(4), output)
    layer.qkv_proj.assert_called_once()
    layer.indexer_proj.assert_called_once()
    if mode == "profile":
        assert torch.count_nonzero(output) == 0
        layer._run_fused_sparse_attention.assert_not_called()
    else:
        layer._run_fused_sparse_attention.assert_called_once()


def test_async_gather_retains_rank_major_layout_and_leaves_wait_to_caller():
    query = torch.randn(3, 2, 128)
    work = Mock()

    def gather(output, value, **kwargs):
        output.copy_(torch.cat([value, value + 1], dim=0))
        assert kwargs == {"group": "tp", "async_op": True}
        return work

    with patch("vllm_ascend.models.minimax_m3.ops.indexer_dcp.dist.all_gather_into_tensor", side_effect=gather):
        output, pending = gather_index_queries_async(query, SimpleNamespace(world_size=2, device_group="tp"))
    assert output.shape == (2, 3, 2, 128)
    torch.testing.assert_close(output[0], query)
    torch.testing.assert_close(output[1], query + 1)
    assert pending is work
    work.wait.assert_not_called()


@pytest.mark.parametrize("size", [2, 4])
def test_empty_gathered_scores_preserve_global_head_ownership(size):
    result = index_shard_scores(
        torch.empty(size, 0, 4 // size, 128, dtype=torch.bfloat16),
        torch.empty(2, 128 // size, 128, dtype=torch.bfloat16),
        torch.empty(0, 17, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        1,
        size,
        0,
        gathered_query=True,
    )
    assert result.shape == (4, 0, 32) and result.dtype == torch.float16
