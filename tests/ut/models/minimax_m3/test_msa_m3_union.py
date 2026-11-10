# SPDX-License-Identifier: Apache-2.0
"""Union operator ABI, exact-mask forwarding and validation on CPU mocks."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.minimax_m3.ops import msa_m3_union as union_module


def _make_attention():
    nn_ops = SimpleNamespace(union_sparse_meta_launch=MagicMock())
    transformer_ops = SimpleNamespace(
        generic_block_sparse_attention_metadata=MagicMock(), generic_block_sparse_attention=MagicMock()
    )
    modules = {"cann_ops_nn_custom": nn_ops, "cann_ops_transformer_custom": transformer_ops}
    with patch.object(union_module, "import_module", side_effect=modules.__getitem__):
        attention = union_module.UnionSparseAttention()
    return attention, nn_ops, transformer_ops


def _inputs():
    return dict(
        q=torch.empty(9, 16, 128, dtype=torch.bfloat16),
        kv_cache=torch.empty(2, 4, 128, 1, 128, dtype=torch.bfloat16),
        topk_idx=torch.tensor([[[2, -1, 0, 1]] * 9], dtype=torch.int32),
        block_table=torch.tensor([[3, 2, 1, 0]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 9], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 512], dtype=torch.int32),
        seq_lens=torch.tensor([512], dtype=torch.int32),
        num_union_groups=2,
        max_kv_blocks=4,
        sm_scale=128**-0.5,
        output=torch.empty(9, 16, 128, dtype=torch.bfloat16),
    )


def test_union_forwards_exact_mask_and_preserves_selector_sets():
    attention, nn_ops, transformer_ops = _make_attention()
    inputs = _inputs()
    original = inputs["topk_idx"].clone()
    union_idx, union_count, mask = object(), object(), object()
    nn_ops.union_sparse_meta_launch.return_value = union_idx, union_count, mask
    expected = torch.randn_like(inputs["output"])
    transformer_ops.generic_block_sparse_attention.return_value = expected, None
    attention(**inputs)

    assert torch.equal(inputs["topk_idx"], original)
    meta_args = nn_ops.union_sparse_meta_launch.call_args.args
    assert meta_args[0].tolist() == [[[0, 1, 2, -1]] * 9]
    assert meta_args[1].tolist() == [[3] * 9]
    assert meta_args[2] is inputs["cu_seqlens_q"]
    assert meta_args[3] is inputs["cu_seqlens_kv"]
    assert meta_args[-3:] == (4, 2, 4)
    builder_call = transformer_ops.generic_block_sparse_attention_metadata.call_args
    attn_call = transformer_ops.generic_block_sparse_attention.call_args
    assert builder_call.args[:2] == (union_idx, union_count)
    assert attn_call.args[3:5] == (union_idx, union_count)
    assert attn_call.kwargs["attn_mask"] is mask
    assert attn_call.kwargs["metadata"] is transformer_ops.generic_block_sparse_attention_metadata.return_value
    assert attn_call.kwargs["block_table"] is inputs["block_table"]
    for call in (builder_call, attn_call):
        assert call.kwargs["mask_mode"] == 3
        assert call.kwargs["softmax_precision"] == 0
        assert call.kwargs["cu_seqlens_q"].dtype == torch.int64
        assert call.kwargs["seqused_kv"] is inputs["seq_lens"]
    assert torch.equal(inputs["output"], expected)


@pytest.mark.parametrize("invalid", ["dtype", "heads", "indices", "groups", "strides"])
def test_union_rejects_unsupported_inputs_before_launch(invalid):
    attention, nn_ops, _ = _make_attention()
    inputs = _inputs()
    if invalid == "dtype":
        inputs["q"] = inputs["q"].float()
    elif invalid == "heads":
        inputs["q"] = torch.empty(9, 17, 128, dtype=torch.bfloat16)
    elif invalid == "indices":
        inputs["topk_idx"] = inputs["topk_idx"][0]
    elif invalid == "groups":
        inputs["num_union_groups"] = 0
    else:
        inputs["kv_cache"] = inputs["kv_cache"].transpose(1, 4)
    with pytest.raises(ValueError):
        attention(**inputs)
    nn_ops.union_sparse_meta_launch.assert_not_called()


def test_empty_union_query_does_not_launch():
    attention, nn_ops, transformer_ops = _make_attention()
    inputs = _inputs()
    inputs["q"] = inputs["q"][:0]
    attention(**inputs)
    nn_ops.union_sparse_meta_launch.assert_not_called()
    transformer_ops.generic_block_sparse_attention.assert_not_called()


def test_union_missing_or_wrong_optional_extensions_fail_at_initialization():
    with patch.object(union_module, "import_module", side_effect=ImportError), pytest.raises(RuntimeError):
        union_module.UnionSparseAttention()
    with patch.object(union_module, "import_module", return_value=SimpleNamespace()), pytest.raises(RuntimeError):
        union_module.UnionSparseAttention()
